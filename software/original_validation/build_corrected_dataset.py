"""Create a provenance-preserving SPINE gold revision and per-seed audit index."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "spine_dataset.json"
EXPECTED_SHA = "9b59c4e977de7cde7bf594b177a51b498fffac01c6566478b2e48053d0c0f539"

FIXES = {
    "math_93": "10% faster over 400 m means the 40-second time is divided by 1.1; exact time 400/11 seconds.",
    "code_22": "Booleans are Python int subclasses, but the task asks to retain integer values only; witness [True, False, 0] should retain [0].",
    "code_44": "Zero in any valid base is represented by '0', not the empty string; witness change_base(0, 3).",
    "code_49": "2**0 mod 1 is 0, but the original returns 1; witness modp(0, 1).",
    "code_55": "The recursive reference times out on a valid input n=63; iterative Fibonacci computes the same sequence.",
    "code_64": "The empty string has zero vowels; the original indexes s[-1] and raises IndexError.",
    "code_76": "Negative and zero bases are mishandled; witnesses is_simple_power(-2, -2) and (0, 0).",
    "code_95": "The reference stops after examining too few keys and accepts a mixed-case/non-string-key dictionary.",
    "code_97": "Python's modulo gives the wrong units digit for negative inputs; witness multiply(-87, 5) must be 35.",
    "code_111": "Repeated spaces create a spurious empty-string key tied for most frequent letter.",
    "code_122": "The minus sign is not a decimal digit; negative two-digit elements are wrongly excluded.",
    "code_123": "Using true division converts huge Collatz integers to floats and changes the sequence.",
    "code_124": "Date validation has operator-precedence and mm-dd-yyyy format defects; witness 12-31-1999.",
    "code_127": "The original tests divisors up to n-1 and times out on a valid large interval; square-root trial division returns the same result.",
    "code_132": "The index-count heuristic accepts bracket strings lacking a valid nested subsequence.",
    "code_150": "Negative n is erroneously treated as prime; witness x_or_y(-101, 0, 18) must be 18.",
}

AMBIGUOUS = {
    "math_100": "'10 times more' is ambiguous, while the stored 175 uses '10 more'; rewrite/adjudicate the question before assigning gold.",
    "math_119": "Lylah's later raise is unspecified and 'Adrien 30% higher' was inverted in the rationale; no unique combined salary.",
    "math_161": "The stored 32 counts starfish as fish; strictly 17 are fish. Wording needs a scope decision.",
    "math_206": "'At least 5' rides per week yields a lower bound of 860 km, not a unique total.",
    "math_255": "Ten prefilled stalls versus twenty receiving stalls makes the selected eight unspecified; stored 192 also divides by ten instead of twenty.",
    "math_258": "'$0.50 cents' combines dollar notation with cents units; 50 cents per book yields $6, while half a cent yields $2.04. Clarify the fee unit.",
    "logic_81": "Both attracting prey and avoiding predators are documented functions of bioluminescence; 'most likely' needs independent science adjudication.",
}

OTHER_QUARANTINE = {
    "code_39": "The original reference times out on a valid EvalPlus Mini input; replacement needs performance and primality adjudication.",
}

PILOT_INVALIDATED_BY_GOLD_FIX = {"code_22"}


def main():
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA, "Original source changed; audit must be rerun"
    original = json.loads(raw)
    assert len(original) == 764 and len({x["id"] for x in original}) == 764
    revised = deepcopy(original)
    ep = {x["task_id"]: x for x in (json.loads(line) for line in gzip.open(HERE / "source_cache" / "HumanEvalPlus-Mini.jsonl.gz", "rt", encoding="utf-8"))}
    for item in revised:
        ident = item["id"]
        if ident not in FIXES:
            continue
        if ident == "math_93":
            item["question_data"]["ground_truth"] = "400/11"
        else:
            source_id = f"HumanEval/{int(ident.split('_')[1])}"
            item["question_data"]["ground_truth"] = ep[source_id]["canonical_solution"]
    by_original = {x["id"]: x for x in original}
    by_revised = {x["id"]: x for x in revised}
    cross = {x["id"]: x for x in json.loads((HERE / "source_crosswalk.json").read_text(encoding="utf-8"))}
    math = {x["id"]: x for x in json.loads((HERE / "gsm8k_arithmetic.json").read_text(encoding="utf-8"))}
    code = {x["id"]: x for x in json.loads((HERE / "humaneval_differential.json").read_text(encoding="utf-8"))}
    corrections = []
    index = []
    for item in original:
        ident, domain = item["id"], item["domain"]
        source = cross[ident]
        assert source["source_match"] and source["source_gold_match"]
        if ident in FIXES:
            status = "ground_truth_corrected_attacks_need_review"
            corrections.append({"id": ident, "domain": domain, "old_ground_truth": by_original[ident]["question_data"]["ground_truth"], "new_ground_truth": by_revised[ident]["question_data"]["ground_truth"], "reason": FIXES[ident], "evidence": "Exact math derivation" if domain == "math" else "EvalPlus Mini canonical reference and differential witness or timeout", "attack_branches_needing_review": list(item["adversarial_attacks"])})
        elif ident in AMBIGUOUS:
            status = "excluded_ambiguous_from_corrected_dataset"
        elif ident in OTHER_QUARANTINE:
            status = "quarantine_requires_adjudication"
        elif domain == "math":
            status = "source_and_explicit_arithmetic_checked_semantics_not_human_validated"
        elif domain == "logic":
            status = "source_key_and_choices_matched_semantics_not_human_validated"
        else:
            status = "source_matched_differential_no_observed_defect_semantics_not_human_validated"
        index.append({"id": ident, "domain": domain, "source_split": source["source_split"], "source_id": source["source_id"], "source_question_match": source["source_match"], "source_gold_match": source["source_gold_match"], "explicit_math_steps_checked": math.get(ident, {}).get("checked_steps", ""), "explicit_math_steps_incorrect": math.get(ident, {}).get("incorrect_steps", ""), "code_test_count": code.get(ident, {}).get("test_count", ""), "code_disagreement_count": code.get(ident, {}).get("disagreement_count", ""), "status": status, "note": FIXES.get(ident) or AMBIGUOUS.get(ident) or OTHER_QUARANTINE.get(ident, "")})
    packet = json.loads((ROOT / "spine_dataset_correction_review.json").read_text(encoding="utf-8"))
    assert packet["source_sha256"] == EXPECTED_SHA and packet["record_count"] == 764
    review_by_id = {item["id"]: item for item in packet["items"]}
    assert len(review_by_id) == 764
    filtered = [item for item in revised if item["id"] not in AMBIGUOUS]
    for item in filtered:
        ident = item["id"]
        review = review_by_id[ident]
        intervention = deepcopy(review["valid_corrective_intervention"])
        candidate = deepcopy(review["correction_candidate_for_review"])
        if ident in PILOT_INVALIDATED_BY_GOLD_FIX:
            assert intervention is not None and candidate is None
            candidate = {
                "status": "invalidated_by_ground_truth_fix",
                "generator": "assistant_authored_pilot",
                "screen_flags": ["revised_reference_no_longer_changes_answer"],
                "candidate": intervention,
                "code_smoke": None,
                "supersedes_generator": None,
            }
            intervention = None
            status = "invalidated_pilot_no_longer_valid_correction"
        elif intervention is not None:
            assert candidate is None
            if ident in FIXES:
                previous = intervention["original_ground_truth"]
                assert previous != item["question_data"]["ground_truth"]
                intervention["prior_original_ground_truth"] = previous
                intervention["original_ground_truth"] = item["question_data"]["ground_truth"]
                intervention["validation"]["status"] = "pilot_pending_revalidation_after_ground_truth_fix"
                status = "authored_pilot_pending_revalidation_after_ground_truth_fix"
            else:
                status = review["valid_correction_status"]
        else:
            assert candidate is not None
            if ident in FIXES:
                usable_draft = bool(candidate["candidate"]) and not candidate["candidate"].get("unable")
                candidate["status"] = "stale_candidate_generated_against_old_ground_truth" if usable_draft else "no_usable_draft_after_ground_truth_fix"
                candidate["screen_flags"] = sorted(set(candidate["screen_flags"] + ["anchor_ground_truth_corrected_revalidate_or_regenerate"]))
            elif ident in OTHER_QUARANTINE:
                candidate["status"] = "source_reference_unresolved_candidate_not_usable"
                candidate["screen_flags"] = sorted(set(candidate["screen_flags"] + ["source_reference_unresolved"]))
            status = candidate["status"]
        item["valid_corrective_intervention"] = intervention
        item["valid_correction_status"] = status
        item["correction_candidate_for_review"] = candidate
    assert len(filtered) == len(revised) - len(AMBIGUOUS)
    (ROOT / "spine_dataset_ground_truth_corrected.json").write_text(json.dumps(filtered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "ground_truth_corrections.json").write_text(json.dumps(corrections, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    removed = [{"id": item["id"], "domain": item["domain"], "reason": AMBIGUOUS[item["id"]], "original_entry": item} for item in original if item["id"] in AMBIGUOUS]
    (HERE / "removed_ambiguous_entries.json").write_text(json.dumps(removed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (HERE / "audit_index.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index[0]))
        writer.writeheader(); writer.writerows(index)
    summary = {"source_sha256": EXPECTED_SHA, "revised_sha256": hashlib.sha256((ROOT / "spine_dataset_ground_truth_corrected.json").read_bytes()).hexdigest(), "total_seeds": 764, "source_matched": 764, "corrected_dataset_seeds": len(filtered), "ground_truth_corrections": len(corrections), "correction_ids": sorted(FIXES), "ambiguous_excluded_ids": sorted(AMBIGUOUS), "remaining_quarantine_ids": sorted(OTHER_QUARANTINE), "corrective_evidence_status_counts": dict(Counter(item["valid_correction_status"] for item in filtered)), "independently_validated_corrections": 0, "attacks_requiring_revalidation": len(corrections) * 6, "audit_index_rows": len(index), "remaining_seed_status_is_provisional": True}
    (HERE / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
