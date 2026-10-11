"""Recheck all accepted code-attack counterexamples in restricted execution."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from repair_false_claim_attacks import DATASET, OUT, approved_up_to, code_check


def main():
    raw = DATASET.read_bytes()
    data = {x['id']: x for x in json.loads(raw)}
    approved = approved_up_to(2)
    records = []
    for (item_id, persona), draft in sorted(approved.items()):
        if draft['domain'] != 'coding':
            continue
        check = code_check(data[item_id], draft)
        records.append({'id': item_id, 'persona': persona,
                        'attack_sha256': hashlib.sha256(draft['attack_text'].encode('utf-8')).hexdigest(),
                        'check': check})
    counts = Counter(x['check']['status'] for x in records)
    bad = [x for x in records if x['check'].get('claim_is_false') is not True
           or x['check'].get('generator_actual_matches') is not True]
    report = {'source_sha256': hashlib.sha256(raw).hexdigest(),
              'coding_attacks_checked': len(records),
              'status_counts': dict(counts),
              'false_claims_confirmed': len(records)-len(bad),
              'failed_checks': bad,
              'scope_note': 'Finite concrete counterexamples only; not a proof of all code behavior.',
              'records': records}
    path = OUT / 'code_attack_machine_audit.json'
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))
    if bad:
        raise SystemExit('Some code attack claims did not check out')


if __name__ == '__main__':
    main()
