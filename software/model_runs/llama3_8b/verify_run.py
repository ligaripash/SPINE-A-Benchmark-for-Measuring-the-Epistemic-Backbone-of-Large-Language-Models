"""Check coverage, input provenance, and independent-branch conversations."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from run import DATASET, HERE, PERSONAS


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    raw = DATASET.read_bytes()
    items = json.loads(raw)
    by_id = {x["id"]: x for x in items}
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(raw).hexdigest() == manifest["dataset_sha256"]
    bases = read(HERE / "base.jsonl")
    base_ok = {(x["id"], x["branch"]): x for x in bases if "error" not in x}
    eligible = set(json.loads((HERE / "eligible_ids.json").read_text(encoding="utf-8")))
    attack_rows = [x for path in sorted(HERE.glob("attacks_*.jsonl")) for x in read(path)]
    attacks = {(x["id"], x["branch"]): x for x in attack_rows if "error" not in x}
    duplicate_successes = {f"{a}/{b}": n for (a, b), n in
                           Counter((x["id"], x["branch"]) for x in attack_rows if "error" not in x).items() if n > 1}
    expected = {(ident, branch) for ident in eligible for branch in PERSONAS}
    errors = []
    for item in items:
        ident = item["id"]
        base = base_ok.get((ident, "base"))
        if not base:
            errors.append(f"missing base {ident}")
            continue
        if base["messages"] != [{"role": "user", "content": item["question_data"]["prompt"]}]:
            errors.append(f"base messages mismatch {ident}")
        for branch in PERSONAS:
            attack = attacks.get((ident, branch))
            if not attack:
                continue
            wanted = base["messages"] + [
                {"role": "assistant", "content": base["response"]},
                {"role": "user", "content": item["adversarial_attacks"][branch]["attack_text"]},
            ]
            if attack["messages"] != wanted:
                errors.append(f"attack messages mismatch {ident}/{branch}")
    corrections = read(HERE / "corrections.jsonl")
    expected_corrections = {x["id"] for x in items if x["id"] in eligible and x.get("valid_corrective_intervention")}
    correction_ok = {x["id"]: x for x in corrections if "error" not in x}
    for ident in expected_corrections:
        base = base_ok[(ident, "base")]
        wanted = base["messages"] + [
            {"role": "assistant", "content": base["response"]},
            {"role": "user", "content": by_id[ident]["valid_corrective_intervention"]["intervention_text"]},
        ]
        if ident not in correction_ok or correction_ok[ident]["messages"] != wanted:
            errors.append(f"correction messages mismatch {ident}")
    summary = {
        "dataset_sha256": manifest["dataset_sha256"],
        "base_successes": len(base_ok), "base_expected": len(items),
        "attack_successes": len(attacks), "attack_expected": len(expected),
        "attack_missing": sorted([f"{a}/{b}" for a, b in expected - attacks.keys()]),
        "attack_unexpected": sorted([f"{a}/{b}" for a, b in attacks.keys() - expected]),
        "attack_error_rows": sum("error" in x for x in attack_rows),
        "attack_duplicate_successes": duplicate_successes,
        "correction_successes": len(correction_ok), "correction_expected": len(expected_corrections),
        "errors": errors,
    }
    (HERE / "integrity.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("attack_missing", "attack_unexpected", "errors")}, indent=2))
    print(f"missing={len(summary['attack_missing'])}, unexpected={len(summary['attack_unexpected'])}, errors={len(errors)}")
    if summary["attack_missing"] or summary["attack_unexpected"] or duplicate_successes or errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
