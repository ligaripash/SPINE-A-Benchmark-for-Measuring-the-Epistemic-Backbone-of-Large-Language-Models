"""Restricted one-case execution of a HumanEval reference; subprocess only."""
from __future__ import annotations

import ast
import builtins
import collections
import hashlib
import json
import math
import sys
import typing

DENIED_NAMES = {'open', 'eval', 'exec', 'compile', 'input', '__import__',
                'globals', 'locals', 'vars', 'getattr', 'setattr', 'delattr',
                '__builtins__', 'breakpoint', 'help', 'dir'}
ALLOWED_BUILTINS = {'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'filter',
                    'float', 'int', 'isinstance', 'len', 'list', 'map', 'max',
                    'min', 'pow', 'range', 'reversed', 'round', 'set', 'sorted',
                    'str', 'sum', 'tuple', 'zip', 'Exception', 'ValueError',
                    'TypeError', 'IndexError', 'KeyError', 'ZeroDivisionError',
                    'ord', 'chr', 'bin', 'hex', 'oct', 'divmod', 'type'}


class StripSafeImports(ast.NodeTransformer):
    def visit_Import(self, node):
        if any(alias.name not in ('math', 'collections', 'hashlib') or alias.asname for alias in node.names):
            raise ValueError('import')
        return None

    def visit_ImportFrom(self, node):
        if node.module not in ('typing', 'math') or node.level or any(alias.asname for alias in node.names):
            raise ValueError('import_from')
        return None


def run():
    payload = json.load(sys.stdin)
    try:
        tree = ast.parse(payload['source'])
    except SyntaxError:
        print(json.dumps({'status': 'unsupported', 'reason': 'syntax_error'}))
        return
    try:
        tree = StripSafeImports().visit(tree)
        ast.fix_missing_locations(tree)
    except ValueError as exc:
        print(json.dumps({'status': 'unsupported', 'reason': str(exc)}))
        return
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef,
                             ast.AsyncFunctionDef, ast.Await)):
            print(json.dumps({'status': 'unsupported', 'reason': 'unsafe_syntax'}))
            return
        if isinstance(node, ast.Attribute) and '__' in node.attr:
            print(json.dumps({'status': 'unsupported', 'reason': 'dunder_attribute'}))
            return
        if isinstance(node, ast.Name) and node.id in DENIED_NAMES:
            print(json.dumps({'status': 'unsupported', 'reason': 'denied_name'}))
            return
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            print(json.dumps({'status': 'unsupported', 'reason': 'decorator'}))
            return
    safe = {name: getattr(builtins, name) for name in ALLOWED_BUILTINS}
    namespace = {'__builtins__': safe, 'math': math,
                 'collections': collections, 'hashlib': hashlib}
    for name in ('List', 'Tuple', 'Optional', 'Dict', 'Set', 'Any', 'Iterable'):
        namespace[name] = getattr(typing, name)
    for name in ('sqrt', 'floor', 'ceil', 'factorial', 'gcd', 'pi', 'isclose'):
        namespace[name] = getattr(math, name)
    try:
        compiled = compile(tree, '<accepted-reference>', 'exec')
        exec(compiled, namespace)
        args = payload['args']
        if isinstance(args, list):
            result = namespace[payload['entry_point']](*args)
        elif isinstance(args, dict):
            result = namespace[payload['entry_point']](**args)
        else:
            print(json.dumps({'status': 'unsupported', 'reason': 'args_not_list_or_object'}))
            return
        json.dumps(result)
        print(json.dumps({'status': 'executed', 'actual': result}))
    except Exception as exc:
        print(json.dumps({'status': 'execution_error',
                          'reason': type(exc).__name__ + ': ' + str(exc)[:120]}))


if __name__ == '__main__':
    run()
