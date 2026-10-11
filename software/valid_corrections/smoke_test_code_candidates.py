"""Run narrowly restricted code-candidate smoke tests in timeout subprocesses.

This checks the supplied examples, not full correctness or adversarial safety.
Candidates with imports, dunder access, or other unsupported syntax are skipped.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from candidate_store import load_candidates

ROOT = Path(__file__).resolve().parent.parent

PYTHON = sys.executable
IN = ROOT / 'valid_corrections/reasoned_candidate_drafts.jsonl'
OUT = ROOT / 'valid_corrections/code_candidate_smoke.jsonl'
DENIED_NAMES = {'open', 'eval', 'exec', 'compile', 'input', '__import__',
                'globals', 'locals', 'vars', 'getattr', 'setattr', 'delattr',
                '__builtins__', 'breakpoint', 'help', 'dir'}
ALLOWED_BUILTINS = {'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'filter',
                    'float', 'int', 'isinstance', 'len', 'list', 'map', 'max',
                    'min', 'pow', 'range', 'reversed', 'round', 'set', 'sorted',
                    'str', 'sum', 'tuple', 'zip', 'Exception', 'ValueError',
                    'TypeError', 'IndexError', 'KeyError', 'ZeroDivisionError'}


def inspect_source(code: str) -> tuple[ast.Module | None, str | None]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, 'syntax_error'
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return None, 'imports_manual_review'
        if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Await)):
            return None, 'unsupported_definition_manual_review'
        if isinstance(node, ast.Attribute) and '__' in node.attr:
            return None, 'dunder_attribute_manual_review'
        if isinstance(node, ast.Name) and node.id in DENIED_NAMES:
            return None, 'denied_name_manual_review'
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            return None, 'decorator_manual_review'
    return tree, None


def equivalent(actual, expected):
    if isinstance(expected, bool):
        return type(actual) is bool and actual == expected
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            equivalent(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            equivalent(actual[k], expected[k]) for k in expected)
    if isinstance(expected, (int, float)):
        return type(actual) in (int, float) and abs(actual - expected) < 1e-9
    return type(actual) is type(expected) and actual == expected


def worker():
    import builtins
    payload = json.load(sys.stdin)
    old_tree, old_problem = inspect_source(payload['old_code'])
    new_tree, new_problem = inspect_source(payload['new_code'])
    if old_problem or new_problem:
        print(json.dumps({'status': 'manual_review',
                          'old_problem': old_problem, 'new_problem': new_problem}))
        return
    safe = {name: getattr(builtins, name) for name in ALLOWED_BUILTINS}
    results = []
    functions = []
    for tree in (old_tree, new_tree):
        namespace = {'__builtins__': safe}
        exec(compile(tree, '<candidate>', 'exec'), namespace)
        functions.append(namespace[payload['entry_point']])
    for test in payload['tests']:
        args = json.loads(test['args_json'])
        expected = json.loads(test['expected_json'])
        if not isinstance(args, list):
            raise ValueError('test_args_not_array')
        observations = []
        for func in functions:
            try:
                actual = func(*json.loads(json.dumps(args)))
                observations.append({'matches_expected': equivalent(actual, expected),
                                     'actual_repr': repr(actual)[:200]})
            except Exception as exc:
                observations.append({'matches_expected': False,
                                     'error': type(exc).__name__ + ': ' + str(exc)[:120]})
        results.append({'old': observations[0], 'new': observations[1]})
    print(json.dumps({'status': 'tested', 'tests': results}))


def main():
    if '--worker' in sys.argv:
        worker()
        return
    source = {x['id']: x for x in json.loads((ROOT / 'spine_dataset.json').read_text(encoding='utf-8'))}
    rows = list(load_candidates().values())
    previous = {x['id']: x for x in (json.loads(line) for line in OUT.read_text(encoding='utf-8').splitlines())} if OUT.exists() else {}
    count = 0
    with OUT.open('a', encoding='utf-8') as stream:
        for row in rows:
            if row['domain'] != 'coding' or row['id'] in previous:
                continue
            item = source[row['id']]
            candidate = row.get('candidate') or {}
            if candidate.get('unable') or not candidate:
                result = {'status': 'unable_or_missing'}
            else:
                old_code = item['question_data']['prompt'] + item['question_data']['ground_truth']
                new_code = candidate.get('revised_reference_code', '')
                old_tree, old_problem = inspect_source(old_code)
                new_tree, new_problem = inspect_source(new_code)
                if old_problem or new_problem:
                    result = {'status': 'manual_review', 'old_problem': old_problem,
                              'new_problem': new_problem}
                else:
                    funcs = [x for x in old_tree.body if isinstance(x, ast.FunctionDef)]
                    entry = funcs[-1].name if funcs else ''
                    payload = {'old_code': old_code, 'new_code': new_code,
                               'entry_point': entry,
                               'tests': candidate.get('distinguishing_tests', [])}
                    try:
                        run = subprocess.run([PYTHON, '-I', '-S', __file__, '--worker'],
                            input=json.dumps(payload), text=True, capture_output=True,
                            timeout=4, check=False)
                        result = json.loads(run.stdout) if run.returncode == 0 else {
                            'status': 'worker_error', 'stderr': run.stderr[-300:]}
                    except subprocess.TimeoutExpired:
                        result = {'status': 'timeout'}
                    except json.JSONDecodeError:
                        result = {'status': 'worker_non_json_output'}
                if result.get('status') == 'tested':
                    tests = result['tests']
                    result['all_new_pass'] = bool(tests) and all(t['new']['matches_expected'] for t in tests)
                    result['old_fails_any'] = any(not t['old']['matches_expected'] for t in tests)
            result = {'id': row['id'], **result}
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            stream.flush()
            count += 1
    print(f'new_code_smoke_results={count} total_recorded={len(previous)+count}')


if __name__ == '__main__':
    main()
