"""Finite, independent-case checks of Llama's 70 candidate code corrections.

The hand-authored expected outputs are in code_correction_oracle_cases.py.
Draft reference code and draft distinguishing inputs are used only as separate
cross-checks. None of these tests constitute a proof of general correctness.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from code_correction_oracle_cases import CASES  # noqa: E402

sys.path.insert(0, str(HERE.parent / 'llama3_8b'))
from score_code import review_ast, safe_builtins, same  # noqa: E402

OUT = HERE / 'code_correction_independent_checks.jsonl'


def extract_code(response: str, entry: str) -> str:
    blocks = re.findall(r'```(?:python|Python)?\s*\n(.*?)```', response, re.S)
    matching = [b for b in blocks if re.search(r'\bdef\s+' + re.escape(entry) + r'\s*\(', b)]
    if matching:
        return max(matching, key=len).strip()
    return (max(blocks, key=len) if blocks else response).strip()


def worker() -> None:
    payload = json.load(sys.stdin)
    code = payload['code']
    entry = payload['entry']
    issue = review_ast(code)
    if issue:
        print(json.dumps({'status': issue, 'run': 0, 'total': len(payload['cases'])}))
        return
    namespace = {'__builtins__': safe_builtins(), '__name__': '__candidate__',
                 'List': List, 'Optional': Optional, 'Tuple': Tuple, 'Dict': Dict, 'Set': Set}
    try:
        exec(compile(code, '<candidate>', 'exec'), namespace)
    except BaseException as exc:
        print(json.dumps({'status': 'setup_error', 'error': repr(exc)[:200],
                          'run': 0, 'total': len(payload['cases'])}))
        return
    if entry not in namespace or not callable(namespace[entry]):
        print(json.dumps({'status': 'required_function_missing', 'run': 0,
                          'total': len(payload['cases'])}))
        return
    function = namespace[entry]
    captured = []
    for i, case in enumerate(payload['cases']):
        args = case['args']
        try:
            actual = function(*copy.deepcopy(args))
        except BaseException as exc:
            print(json.dumps({'status': 'runtime_error', 'index': i, 'args': args,
                              'error': repr(exc)[:200], 'run': i + 1,
                              'total': len(payload['cases'])}))
            return
        if payload.get('capture'):
            captured.append(repr(actual))
            continue
        expected = ast.literal_eval(case['expected_repr'])
        if not same(actual, expected, 1e-9):
            print(json.dumps({'status': 'wrong_output', 'index': i, 'args': args,
                              'expected': repr(expected)[:200], 'actual': repr(actual)[:200],
                              'run': i + 1, 'total': len(payload['cases'])}))
            return
    result = {'status': 'passed', 'run': len(payload['cases']),
              'total': len(payload['cases'])}
    if payload.get('capture'):
        result['values'] = captured
    print(json.dumps(result))


def run(code: str, entry: str, cases: list[dict], capture: bool = False) -> dict:
    payload = {'code': code, 'entry': entry, 'cases': cases, 'capture': capture}
    try:
        proc = subprocess.run([sys.executable, '-I', '-X', 'utf8', str(Path(__file__).resolve()), '--worker'],
                              input=json.dumps(payload), text=True, capture_output=True,
                              encoding='utf-8', errors='replace', timeout=10, cwd=HERE)
    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'run': 0, 'total': len(cases)}
    if proc.returncode:
        return {'status': 'worker_error', 'stderr': proc.stderr[-400:], 'run': 0,
                'total': len(cases)}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {'status': 'invalid_worker_output', 'stdout': proc.stdout[-200:],
                'run': 0, 'total': len(cases)}


def main() -> None:
    dataset = {x['id']: x for x in json.loads(
        (ROOT / 'spine_dataset_false_claim_attacks.json').read_text(encoding='utf-8'))}
    scores = [json.loads(s) for s in (HERE / 'correction_scores_exploratory.jsonl')
              .read_text(encoding='utf-8').splitlines() if s.strip()]
    targets = {x['id'] for x in scores if x['domain'] == 'coding' and x['correct'] is None}
    if targets != set(CASES):
        raise RuntimeError(f'Case coverage differs: missing={sorted(targets-set(CASES))}, '
                           f'extra={sorted(set(CASES)-targets)}')
    raw = [json.loads(s) for name in ('corrections_new.jsonl', 'corrections_reused.jsonl')
           for s in (HERE / name).read_text(encoding='utf-8').splitlines() if s.strip()]
    responses = {x['id']: x['response'] for x in raw if x['id'] in targets}
    if set(responses) != targets:
        raise RuntimeError('Missing raw Llama correction response')

    records = []
    for ident in sorted(targets, key=lambda s: int(s.split('_')[1])):
        item = dataset[ident]
        correction = item['correction_candidate_for_review']['candidate']
        parsed = ast.parse(item['question_data']['prompt'])
        functions = [n.name for n in parsed.body if isinstance(n, ast.FunctionDef)]
        entry = functions[-1]
        cases = [{'args': list(args), 'expected_repr': repr(expected)}
                 for args, expected in CASES[ident]]
        reference = correction.get('revised_reference_code', '')
        reference_result = run(reference, entry, cases)
        llama_code = extract_code(responses[ident], entry)
        llama_result = run(llama_code, entry, cases)
        draft_tests = correction.get('distinguishing_tests', [])
        usable_draft_tests = []
        malformed_draft_indices = []
        for i, test in enumerate(draft_tests):
            try:
                args = json.loads(test['args_json'])
                json.loads(test['expected_json'])
                if not isinstance(args, list):
                    raise ValueError('Arguments are not a list')
                usable_draft_tests.append((test, args))
            except (KeyError, ValueError, TypeError):
                malformed_draft_indices.append(i)
        draft_inputs = [{'args': args} for _, args in usable_draft_tests]
        draft_reference_capture = run(reference, entry, draft_inputs, capture=True)
        if draft_reference_capture['status'] == 'passed':
            reference_values = [ast.literal_eval(v) for v in draft_reference_capture['values']]
            draft_claims_match = all(json.loads(json.dumps(v)) == json.loads(t['expected_json'])
                                     for v, (t, _) in zip(reference_values, usable_draft_tests))
            draft_cases = [{'args': x['args'], 'expected_repr': v}
                           for x, v in zip(draft_inputs, draft_reference_capture['values'])]
            llama_draft_result = run(llama_code, entry, draft_cases)
        else:
            draft_claims_match = None
            llama_draft_result = {'status': 'reference_capture_failed'}
        record = {'id': ident, 'candidate_status': item['valid_correction_status'],
                  'entry_point': entry, 'independent_cases': len(cases),
                  'draft_reference_check': reference_result,
                  'llama_independent_check': llama_result,
                  'draft_cases': len(draft_tests), 'malformed_draft_indices': malformed_draft_indices,
                  'draft_cases_used': len(usable_draft_tests),
                  'draft_claims_match_reference': draft_claims_match,
                  'draft_reference_capture': {k: v for k, v in draft_reference_capture.items()
                                              if k != 'values'},
                  'llama_draft_check': llama_draft_result}
        records.append(record)
        print(f"{ident:>8}: reference={reference_result['status']:<25} "
              f"llama={llama_result['status']:<18} draft={llama_draft_result['status']}",
              flush=True)
    OUT.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in records),
                   encoding='utf-8')
    print(json.dumps({'records': len(records),
                      'reference_status': dict(Counter(x['draft_reference_check']['status']
                                                       for x in records)),
                      'llama_status': dict(Counter(x['llama_independent_check']['status']
                                                   for x in records)),
                      'llama_draft_status': dict(Counter(x['llama_draft_check']['status']
                                                         for x in records)),
                      'path': str(OUT)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    worker() if args.worker else main()
