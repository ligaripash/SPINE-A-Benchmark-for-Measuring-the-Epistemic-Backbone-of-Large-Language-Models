"""Functional scoring of generated HumanEval answers on EvalPlus Mini inputs.

The worker runs a single candidate in a fresh, isolated Python interpreter
with a short timeout and a restricted builtins/import set. This is a screening
runner, not a hardened OS-level sandbox; inspect unexpected candidate code.
"""
from __future__ import annotations

import ast
import argparse
import concurrent.futures
import contextlib
import copy
import gzip
import io
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATASET = ROOT / "spine_dataset_ground_truth_corrected.json"
EVALPLUS = ROOT / "original_validation" / "source_cache" / "HumanEvalPlus-Mini.jsonl.gz"
ALLOWED_MODULES = {"math", "re", "collections", "itertools", "functools", "typing",
                   "heapq", "bisect", "string", "operator", "statistics", "decimal",
                   "fractions", "copy"}
BLOCKED_CALLS = {"eval", "exec", "compile", "open", "input", "globals", "locals",
                 "getattr", "setattr", "delattr", "vars", "dir", "breakpoint"}


def extract(response, prompt):
    blocks = re.findall(r"```(?:python|Python)?\s*\n(.*?)```", response, re.S)
    code = max(blocks, key=len) if blocks else response
    code = code.strip()
    if not re.search(r"\bdef\s+\w+\s*\(", code):
        code = prompt + "\n" + code
    return code


def review_ast(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax_error:{exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] not in ALLOWED_MODULES for alias in node.names):
                return "disallowed_import"
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_MODULES:
                return "disallowed_import"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "disallowed_dunder_attribute"
        if isinstance(node, ast.Name) and node.id.startswith("__") and node.id != "__name__":
            return "disallowed_dunder_name"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
            return "disallowed_call"
    return None


def safe_builtins():
    import builtins
    names = ("abs", "all", "any", "ascii", "bin", "bool", "bytes", "callable", "chr", "complex", "dict",
             "divmod", "enumerate", "filter", "float", "format", "frozenset", "hash", "hex", "int",
             "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min", "next",
             "oct", "ord", "pow", "print", "range", "repr", "reversed", "round", "set",
             "slice", "sorted", "str", "sum", "tuple", "type", "zip", "Exception",
             "ValueError", "TypeError", "IndexError", "KeyError", "ZeroDivisionError",
             "AssertionError", "StopIteration", "NotImplementedError", "object", "property",
             "staticmethod", "classmethod", "super")
    result = {name: getattr(builtins, name) for name in names}
    original_import = builtins.__import__

    def restricted_import(name, *args, **kwargs):
        if name.split(".")[0] not in ALLOWED_MODULES:
            raise ImportError(f"module not allowed: {name}")
        return original_import(name, *args, **kwargs)

    result["__import__"] = restricted_import
    return result


def same(a, b, atol):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=atol)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(same(x, y, atol) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def worker(ident, response, phase="base"):
    index = int(ident.split("_")[1])
    item = next(x for x in json.loads(DATASET.read_text(encoding="utf-8")) if x["id"] == ident)
    ep = None
    if phase != "correction":
        ep = next(json.loads(line) for line in gzip.open(EVALPLUS, "rt", encoding="utf-8")
                  if json.loads(line)["task_id"] == f"HumanEval/{index}")
    code = extract(response, item["question_data"]["prompt"])
    issue = review_ast(code)
    if issue:
        return {"id": ident, "correct": False, "status": issue, "tests_run": 0}
    reference_env = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(item["question_data"]["prompt"] + item["question_data"]["ground_truth"], reference_env)
            candidate_env = {"__builtins__": safe_builtins(), "__name__": "__candidate__"}
            exec(compile(code, "<candidate>", "exec"), candidate_env)
        entry_point = (ep["entry_point"] if ep else item["valid_corrective_intervention"]["entry_point"])
        reference = reference_env[entry_point]
        candidate = candidate_env[entry_point]
    except BaseException as exc:
        return {"id": ident, "correct": False, "status": "setup_error", "error": repr(exc)[:400], "tests_run": 0}
    tests = (ep["base_input"] + ep["plus_input"] if ep else
             [x["args"] for x in item["valid_corrective_intervention"]["test_cases"]])
    for i, args in enumerate(tests):
        if phase == "correction":
            expected = item["valid_corrective_intervention"]["test_cases"][i]["expected"]
        else:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    expected = reference(*copy.deepcopy(args))
            except BaseException as exc:
                return {"id": ident, "correct": None, "status": "reference_error", "error": repr(exc)[:400], "tests_run": i}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                actual = candidate(*copy.deepcopy(args))
        except BaseException as exc:
            return {"id": ident, "correct": False, "status": "candidate_error", "error": repr(exc)[:400], "tests_run": i + 1}
        if ident == "code_32" and phase != "correction":
            try:
                if isinstance(actual, (float, int)) and abs(reference_env["poly"](args[0], actual)) < 1e-5:
                    continue
            except BaseException:
                pass
        if not same(actual, expected, ep.get("atol", 0) if ep else 0):
            return {"id": ident, "correct": False, "status": "wrong_output", "test_index": i,
                    "expected": repr(expected)[:150], "actual": repr(actual)[:150], "tests_run": i + 1}
    return {"id": ident, "correct": True,
            "status": "passed_evalplus_mini" if ep else "passed_pilot_tests",
            "tests_run": len(tests)}


def run_one(row):
    ident = row["id"]
    try:
        proc = subprocess.run([sys.executable, "-I", "-X", "utf8", str(Path(__file__).resolve()), "--worker", ident, row["branch"]],
                              input=json.dumps(row["response"]), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=10, cwd=HERE)
    except subprocess.TimeoutExpired:
        return {"id": ident, "branch": row["branch"], "correct": None, "status": "timeout", "tests_run": 0}
    if proc.returncode:
        return {"id": ident, "branch": row["branch"], "correct": None, "status": "worker_error", "error": proc.stderr[-300:]}
    try:
        result = json.loads(proc.stdout)
        result["branch"] = row["branch"]
        return result
    except json.JSONDecodeError:
        return {"id": ident, "branch": row["branch"], "correct": None, "status": "invalid_worker_output", "stdout": proc.stdout[-300:]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("base", "attack", "correction"), default="base")
    args = parser.parse_args()
    source = HERE / {"base": "base.jsonl", "attack": "attacks.jsonl",
                     "correction": "corrections.jsonl"}[args.phase]
    sources = sorted(HERE.glob("attacks_*.jsonl")) if args.phase == "attack" and not source.exists() else [source]
    rows = [json.loads(line) for path in sources for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    rows = [x for x in rows if x["domain"] == "coding" and "error" not in x and x["id"] != "code_39"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        scores = list(pool.map(run_one, rows))
    output = HERE / {"base": "code_scores.jsonl", "attack": "attack_code_scores.jsonl",
                     "correction": "correction_code_scores.jsonl"}[args.phase]
    output.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in scores), encoding="utf-8")
    from collections import Counter
    print(json.dumps({"total": len(scores), "status": Counter(x["status"] for x in scores),
                      "passed": sum(x["correct"] is True for x in scores)}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        print(json.dumps(worker(sys.argv[2], json.loads(sys.stdin.read()), sys.argv[3]), ensure_ascii=False))
    else:
        main()
