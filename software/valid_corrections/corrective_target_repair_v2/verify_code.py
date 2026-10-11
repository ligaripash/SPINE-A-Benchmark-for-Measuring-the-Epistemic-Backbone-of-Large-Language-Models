"""Execute replacement acceptance cases with exact recursive output types.

Reuses the existing AST/import restrictions and child-process execution worker.
This is restricted execution, not an operating-system security sandbox.
"""
import ast
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(1, str(ROOT / 'valid_corrections/corrective_target_validation_v1'))
import audit_corrective_targets as audit


def same_result(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(same_result(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same_result(actual[k], v) for k, v in expected.items())
    return actual == expected


def execute(ident, source, entry, cases):
    payload = {'id': ident, 'source': source, 'entry': entry, 'cases': cases}
    try:
        out = subprocess.run([sys.executable, '-I', '-X', 'utf8', str(Path(__file__).resolve()), '--worker'],
            input=json.dumps(payload), capture_output=True, text=True, encoding='utf-8', timeout=12)
        if out.returncode:
            return {'status': 'execution_unresolved', 'reason': out.stderr[-400:]}
        return json.loads(out.stdout)
    except subprocess.TimeoutExpired:
        return {'status': 'execution_unresolved', 'reason': 'timeout_12s'}


if __name__ == '__main__':
    if '--worker' in sys.argv:
        audit.same_result = same_result
        audit.worker()
    else:
        # Score a supplied complete implementation, not an extracted code block.
        if len(sys.argv) != 3:
            raise SystemExit('Usage: python verify_code.py code_ID complete_implementation.py')
        ident, filename = sys.argv[1:]
        from repair_specs import CODE
        from independent_code_cases import packed_cases
        result = execute(ident, Path(filename).read_text(encoding='utf-8'),
                         CODE[ident]['entry_point'], packed_cases(ident))
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result['status'] == 'passed' else 1)
