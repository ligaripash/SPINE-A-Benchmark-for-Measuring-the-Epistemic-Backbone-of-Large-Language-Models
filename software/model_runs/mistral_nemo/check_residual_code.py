"""Targeted independent checks for two unresolved Mistral correction responses."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "llama3_8b_revised"))
from validate_70_code_corrections import extract_code, run

raw = {x["id"]: x for x in map(json.loads, (HERE / "correction.jsonl").read_text(encoding="utf-8").splitlines())}
checks = {
    "code_32": {
        "entry": "find_zero",
        "cases": [
            {"args": [[0, 1]], "expected_repr": "0.0"},
            {"args": [[1, 2]], "expected_repr": "-0.5"},
        ],
    },
    "code_44": {
        "entry": "change_base",
        "cases": [
            {"args": [6, 2], "expected_repr": repr("011")},
            {"args": [13, 3], "expected_repr": repr("111")},
            {"args": [0, 2], "expected_repr": repr("0")},
            # The baseline response returned '' on negative input. The
            # intervention explicitly requires preserving that behavior.
            {"args": [-6, 2], "expected_repr": repr("")},
        ],
    },
}
out = {}
for ident, item in checks.items():
    code = extract_code(raw[ident]["response"], item["entry"])
    out[ident] = {
        "target": "Mistral correction response",
        "cases": item["cases"],
        "result": run(code, item["entry"], item["cases"]),
    }
(HERE / "residual_code_checks.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
