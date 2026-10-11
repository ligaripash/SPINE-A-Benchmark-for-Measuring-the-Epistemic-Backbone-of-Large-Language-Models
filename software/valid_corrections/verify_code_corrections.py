"""Summarize executable checks for retained code correction records.

Existing candidate smoke tests are self-authored examples. Four rapid-review
items also receive separately specified property-oracle checks in a restricted
subprocess. Neither level proves the English specification in general.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import itertools
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_test_code_candidates import ALLOWED_BUILTINS, inspect_source, equivalent

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATASET = ROOT / 'spine_dataset_ground_truth_corrected.json'
OUT = HERE / 'code_correction_machine_review.json'
WORDS = 'zero one two three four five six seven eight nine'.split()


def oracle_cases(item_id):
    if item_id == 'code_19':
        rng = random.Random(19001)
        sequences = [p for n in range(5) for p in itertools.product(WORDS[:5], repeat=n)]
        sequences += [tuple(rng.choices(WORDS, k=rng.randrange(1, 8))) for _ in range(200)]
        return [{'args': [' '.join(p)],
                 'expected': ' '.join(sorted(p, key=WORDS.index, reverse=True))}
                for p in sequences]
    if item_id == 'code_157':
        cases = []
        for a, b, c in itertools.product(range(-1, 11), repeat=3):
            x, y, z = sorted((a, b, c))
            cases.append({'args': [a, b, c],
                          'expected': x > 0 and x + y > z and x*x + y*y > z*z})
        return cases
    if item_id == 'code_10':
        cases = []
        for n in range(7):
            for chars in itertools.product('abc', repeat=n):
                s = ''.join(chars)
                possible = [s + s[:k][::-1] for k in range(n + 1)]
                valid = [v for v in possible if len(v) % 2 == 0 and v == v[::-1]]
                cases.append({'args': [s], 'expected': min(valid, key=len)})
        return cases
    if item_id == 'code_134':
        cases = []
        for n in range(5):
            for chars in itertools.product(('a', 'Z', '1', ' ', '\t', '\n'), repeat=n):
                s = ''.join(chars)
                letter = bool(s) and ('a' <= s[-1] <= 'z' or 'A' <= s[-1] <= 'Z')
                cases.append({'args': [s],
                              'expected': bool(letter and (len(s) == 1 or s[-2] in (' ', '\t')))})
        return cases
    return None


def worker():
    payload = json.load(sys.stdin)
    tree, problem = inspect_source(payload['source'])
    if problem:
        print(json.dumps({'status': 'unsupported', 'reason': problem}))
        return
    safe = {name: getattr(builtins, name) for name in ALLOWED_BUILTINS | {'ord', 'chr'}}
    namespace = {'__builtins__': safe}
    exec(compile(tree, '<revised-reference>', 'exec'), namespace)
    function = namespace[payload['entry_point']]
    mismatches = []
    for i, case in enumerate(payload['cases']):
        try:
            got = function(*case['args'])
            if not equivalent(got, case['expected']):
                mismatches.append({'index': i, 'args': case['args'],
                                   'expected': case['expected'], 'observed': repr(got)[:160]})
        except Exception as exc:
            mismatches.append({'index': i, 'args': case['args'],
                               'expected': case['expected'], 'error': type(exc).__name__ + ': ' + str(exc)[:100]})
        if len(mismatches) >= 10:
            break
    print(json.dumps({'status': 'pass' if not mismatches else 'fail',
                      'case_count': len(payload['cases']), 'mismatches': mismatches}))


def signature_check(item, source, entry):
    try:
        original = ast.parse(item['question_data']['prompt'])
        revised = ast.parse(source)
    except SyntaxError:
        return 'syntax_error'
    old = next((n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == entry), None)
    new = next((n for n in revised.body if isinstance(n, ast.FunctionDef) and n.name == entry), None)
    if not old or not new:
        return 'entry_point_missing'
    old_names = [a.arg for a in old.args.posonlyargs + old.args.args + old.args.kwonlyargs]
    new_names = [a.arg for a in new.args.posonlyargs + new.args.args + new.args.kwonlyargs]
    return 'match' if old_names == new_names else 'arguments_changed'


def main():
    if '--worker' in sys.argv:
        worker()
        return
    raw = DATASET.read_bytes()
    rows = [x for x in json.loads(raw) if x['domain'] == 'coding']
    original = {x['id']: x for x in json.loads((ROOT / 'spine_dataset.json').read_text(encoding='utf-8'))}
    smoke = {x['id']: x for x in map(json.loads, (HERE / 'code_candidate_smoke.jsonl').open(encoding='utf-8'))}
    pilot = {x['id']: x for x in json.loads((HERE / 'validation_report.json').read_text(encoding='utf-8'))['checks']}
    rapid = {x['id'] for x in json.loads((HERE / 'rapid_evidence_review/manifest.json').read_text(encoding='utf-8'))['selected'] if x['domain'] == 'coding'}
    report = []
    for item in rows:
        item_id = item['id']
        authored = item.get('valid_corrective_intervention')
        candidate = (item.get('correction_candidate_for_review') or {}).get('candidate') or {}
        correction = authored or candidate
        is_stale = item['question_data']['ground_truth'] != original[item_id]['question_data']['ground_truth']
        record = {'id': item_id, 'origin': 'authored_pilot' if authored else 'candidate',
                  'original_reference_changed': is_stale}
        if not correction or candidate.get('unable'):
            record['machine_status'] = 'no_usable_candidate'
            report.append(record)
            continue
        reference = correction.get('ground_truth') if authored else candidate.get('revised_reference_code', '')
        entry = correction.get('entry_point') if authored else ''
        if not entry:
            try:
                funcs = [n.name for n in ast.parse(item['question_data']['prompt']).body if isinstance(n, ast.FunctionDef)]
                entry = funcs[-1] if funcs else ''
            except SyntaxError:
                entry = ''
        record['signature'] = signature_check(item, reference, entry)
        if authored:
            check = pilot.get(item_id, {})
            record['self_tests'] = {'count': check.get('revised_passes', 0),
                                    'revised_all_pass': bool(check.get('revised_passes')),
                                    'original_concrete_mismatch': bool(check.get('original_fails'))}
        else:
            result = smoke.get(item_id, {})
            tests = result.get('tests', [])
            record['self_tests'] = {'count': len(tests),
                                    'execution_status': result.get('status', 'missing'),
                                    'revised_all_pass': result.get('status') == 'tested' and bool(result.get('all_new_pass')),
                                    'original_concrete_mismatch': any(not t['old']['matches_expected'] and 'error' not in t['old'] for t in tests)}
        if item_id == 'code_39':
            record['machine_status'] = 'quarantined_original_reference'
        elif is_stale:
            record['machine_status'] = 'stale_original_reference'
        elif record['signature'] != 'match':
            record['machine_status'] = 'signature_or_syntax_issue'
        elif record['self_tests']['revised_all_pass'] and record['self_tests']['original_concrete_mismatch']:
            record['machine_status'] = 'self_test_supported'
        else:
            record['machine_status'] = 'unresolved_or_failed_self_tests'
        if item_id in rapid and record['machine_status'] == 'self_test_supported':
            cases = oracle_cases(item_id)
            if cases:
                payload = {'source': reference, 'entry_point': entry, 'cases': cases}
                try:
                    run = subprocess.run([sys.executable, '-I', '-S', str(Path(__file__).resolve()), '--worker'],
                                         input=json.dumps(payload), text=True, capture_output=True,
                                         timeout=15, check=False)
                    record['property_oracle'] = json.loads(run.stdout) if run.returncode == 0 else {
                        'status': 'worker_error', 'stderr': run.stderr[-200:]}
                except subprocess.TimeoutExpired:
                    record['property_oracle'] = {'status': 'timeout', 'case_count': len(cases)}
                if record['property_oracle']['status'] == 'pass':
                    record['machine_status'] = 'property_oracle_supported'
        report.append(record)
    output = {'dataset_sha256': hashlib.sha256(raw).hexdigest(),
              'scope': '164 retained code correction records',
              'status_meaning': {'property_oracle_supported': 'Passes separately specified finite property tests; not a general proof or independent human validation.',
                                 'self_test_supported': 'Passes examples supplied with the candidate; semantic validity remains unverified.'},
              'counts': dict(Counter(x['machine_status'] for x in report)),
              'rapid_code_records': [x for x in report if x['id'] in rapid],
              'records': report}
    OUT.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'counts': output['counts'], 'rapid_code_records': output['rapid_code_records']}, indent=2))


if __name__ == '__main__':
    main()
