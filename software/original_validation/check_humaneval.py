"""Differentially exercise every SPINE HumanEval reference on EvalPlus Mini inputs.

The candidate functions are executed in short-lived Python subprocesses with a
per-task timeout. Disagreements are evidence for adjudication, not automatic
permission to replace the original reference.
"""
from __future__ import annotations

import concurrent.futures
import copy
import gzip
import json
import math
import subprocess
import sys
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = HERE / "source_cache" / "HumanEvalPlus-Mini.jsonl.gz"


def worker(task_id: str, dataset: Path):
    items = json.loads(dataset.read_text(encoding="utf-8"))
    code = next(x for x in items if x["id"] == f"code_{task_id.split('/')[-1]}")
    ep = next(json.loads(line) for line in gzip.open(SOURCE, "rt", encoding="utf-8") if json.loads(line)["task_id"] == task_id)
    env1, env2 = {}, {}
    try:
        exec(code["question_data"]["prompt"] + code["question_data"]["ground_truth"], env1)
        exec(ep["prompt"] + ep["canonical_solution"], env2)
    except Exception as exc:
        return {"id": code["id"], "status": "compile_error", "error": repr(exc)}
    function = ep["entry_point"]
    results = []
    def short_repr(value):
        if isinstance(value, int) and value.bit_length() > 14000:
            return f"<integer with {value.bit_length()} bits>"
        try:
            return repr(value)[:400]
        except ValueError:
            return f"<{type(value).__name__}: representation too large>"
    for source, inputs in (("base", ep["base_input"]), ("plus", ep["plus_input"])):
        for index, args in enumerate(inputs):
            outputs = []
            for env in (env1, env2):
                try:
                    output = env[function](*copy.deepcopy(args))
                    outputs.append(("ok", output))
                except Exception as exc:
                    outputs.append(("error", type(exc).__name__))
            a, b = outputs
            try:
                def equivalent(left, right):
                    if isinstance(left, int) and isinstance(right, int) and not isinstance(left, bool) and not isinstance(right, bool):
                        return left == right
                    if isinstance(left, (float, int)) and isinstance(right, (float, int)) and not isinstance(left, bool) and not isinstance(right, bool):
                        return math.isclose(left, right, rel_tol=1e-9, abs_tol=ep.get("atol", 0))
                    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
                        return len(left) == len(right) and all(equivalent(x, y) for x, y in zip(left, right))
                    return type(left) == type(right) and left == right
                same = a[0] == b[0] and equivalent(a[1], b[1])
                # HumanEval/32 deliberately allows multiple distinct roots.
                if not same and task_id == "HumanEval/32" and a[0] == b[0] == "ok":
                    same = all(abs(env["poly"](args[0], output[1])) < 1e-5 for env, output in ((env1, a), (env2, b)))
            except Exception:
                same = False
            if not same:
                results.append({"set": source, "index": index, "input": short_repr(args), "spine": short_repr(a), "evalplus": short_repr(b)})
    return {"id": code["id"], "status": "ok", "test_count": len(ep["base_input"]) + len(ep["plus_input"]), "disagreement_count": len(results), "disagreements": results}


def run_one(task_id: str, dataset: Path):
    try:
        result = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--worker", task_id, str(dataset)], capture_output=True, text=True, timeout=12)
        if result.returncode:
            return {"id": f"code_{task_id.split('/')[-1]}", "status": "process_error", "error": result.stderr[-1000:]}
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return {"id": f"code_{task_id.split('/')[-1]}", "status": "timeout"}


def main(dataset: Path, output_prefix: str):
    with gzip.open(SOURCE, "rt", encoding="utf-8") as handle:
        ids = [json.loads(line)["task_id"] for line in handle]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda ident: run_one(ident, dataset), ids))
    rows.sort(key=lambda x: int(x["id"].split("_")[-1]))
    (HERE / f"{output_prefix}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {"tasks": len(rows), "tested": sum(x["status"] == "ok" for x in rows), "tests": sum(x.get("test_count", 0) for x in rows), "disagreeing_tasks": sum(x.get("disagreement_count", 0) > 0 for x in rows), "disagreeing_test_cases": sum(x.get("disagreement_count", 0) for x in rows), "errors": [x for x in rows if x["status"] != "ok"], "disagreeing_ids": [x["id"] for x in rows if x.get("disagreement_count", 0) > 0]}
    (HERE / f"{output_prefix}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        print(json.dumps(worker(sys.argv[2], Path(sys.argv[3])), ensure_ascii=False))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--dataset", type=Path, default=ROOT / "spine_dataset.json")
        parser.add_argument("--output-prefix", default="humaneval_differential")
        args = parser.parse_args()
        main(args.dataset.resolve(), args.output_prefix)
