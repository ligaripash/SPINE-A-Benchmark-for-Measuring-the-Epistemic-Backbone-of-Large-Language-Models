"""Targeted finite checks for Gemini's two unresolved code corrections."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "llama3_8b_revised"))
from validate_70_code_corrections import extract_code, run

raw = {x["id"]: x for x in map(json.loads, (HERE / "correction.jsonl").read_text(encoding="utf-8").splitlines())}
checks = {
    "code_22": {
        "entry": "filter_integers",
        "cases": [
            {"args": [[True, 1, False, 0, 2.0, "3"]], "expected_repr": repr([1, 0])},
            {"args": [[]], "expected_repr": repr([])},
            {"args": [[1, 1, -2]], "expected_repr": repr([1, 1, -2])},
            {"args": [[False, True]], "expected_repr": repr([])},
        ],
    },
    "code_44": {
        "entry": "change_base",
        "cases": [
            {"args": [8, 2], "expected_repr": repr("0001")},
            {"args": [6, 2], "expected_repr": repr("011")},
            {"args": [0, 2], "expected_repr": repr("0")},
            # The Gemini baseline returns '' for negative x. The intervention
            # explicitly asks to preserve the original behavior for x <= 0.
            {"args": [-6, 2], "expected_repr": repr("")},
        ],
    },
}
out = {}
for ident, item in checks.items():
    code = extract_code(raw[ident]["response"], item["entry"])
    out[ident] = {"cases": item["cases"], "result": run(code, item["entry"], item["cases"])}
(HERE / "residual_code_checks.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
