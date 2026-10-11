"""Check revision minimality, corrected-code syntax, audit coverage and UI payload."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent


def main():
    old_raw = (ROOT / "spine_dataset.json").read_bytes()
    old = json.loads(old_raw)
    new = json.loads((ROOT / "spine_dataset_ground_truth_corrected.json").read_text(encoding="utf-8"))
    summary = json.loads((HERE / "validation_summary.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(old_raw).hexdigest() == summary["source_sha256"]
    excluded = set(summary["ambiguous_excluded_ids"])
    assert len(old) == 764 and len(new) == 757 == summary["corrected_dataset_seeds"]
    assert {x["id"] for x in old} - {x["id"] for x in new} == excluded
    removed = json.loads((HERE / "removed_ambiguous_entries.json").read_text(encoding="utf-8"))
    assert {x["id"] for x in removed} == excluded and len(removed) == 7
    changes = []
    retained_original = [x for x in old if x["id"] not in excluded]
    for original, revised in zip(retained_original, new):
        assert original["id"] == revised["id"] and original["domain"] == revised["domain"]
        assert original["question_data"]["prompt"] == revised["question_data"]["prompt"]
        assert original["adversarial_attacks"] == revised["adversarial_attacks"]
        if original["question_data"]["ground_truth"] != revised["question_data"]["ground_truth"]:
            changes.append(original["id"])
        if revised["domain"] == "coding":
            ast.parse(revised["question_data"]["prompt"] + revised["question_data"]["ground_truth"])
    assert set(changes) == set(summary["correction_ids"])
    assert Fraction(next(x for x in new if x["id"] == "math_93")["question_data"]["ground_truth"]) == Fraction(400, 11)
    assert all("valid_corrective_intervention" in x and "valid_correction_status" in x and "correction_candidate_for_review" in x for x in new)
    assert all((x["valid_corrective_intervention"] is None) != (x["correction_candidate_for_review"] is None) for x in new)
    assert sum(x["valid_corrective_intervention"] is not None for x in new) == 59
    assert Counter(x["valid_correction_status"] for x in new) == summary["corrective_evidence_status_counts"]
    assert summary["independently_validated_corrections"] == 0
    by_id = {x["id"]: x for x in new}
    assert by_id["code_22"]["valid_corrective_intervention"] is None
    assert by_id["code_22"]["correction_candidate_for_review"]["status"] == "invalidated_by_ground_truth_fix"
    for ident in ("code_55", "code_127", "code_132"):
        pilot = by_id[ident]["valid_corrective_intervention"]
        assert pilot["original_ground_truth"] == by_id[ident]["question_data"]["ground_truth"]
        assert pilot["validation"]["status"] == "pilot_pending_revalidation_after_ground_truth_fix"
    with (HERE / "audit_index.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 764 and {x["id"] for x in rows} == {x["id"] for x in old}
    code = json.loads((HERE / "humaneval_corrected_differential_summary.json").read_text(encoding="utf-8"))
    assert code["tested"] == 163 and code["errors"][0]["id"] == "code_39"
    assert set(code["disagreeing_ids"]) == {"code_114", "code_147"}
    html = (ROOT / "valid_corrections/full_dataset_review/review_spine.html").read_text(encoding="utf-8")
    payload = json.loads(re.search(r'<script type="application/json" id="dataset">(.*?)</script>', html, re.S).group(1))
    assert len(payload["items"]) == 764
    assert payload["ground_truth_correction_count"] == 16
    assert payload["source_adjudication_count"] == 8
    assert payload["corrected_sha256"] == summary["revised_sha256"]
    print(json.dumps({"retained_seeds": len(new), "ambiguous_removed": len(excluded), "revision_changes": len(changes), "code_syntax_checked": 164, "audit_index_rows": len(rows), "browser_items": len(payload["items"]), "corrected_code_tasks_tested": code["tested"]}, indent=2))


if __name__ == "__main__": main()
