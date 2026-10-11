"""Recheck frozen 480 Sol claim tests against the evaluated dataset references."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
from pathlib import Path
import audit_retained_code_attacks as audit
from repair_false_claim_attacks import approved_up_to, code_check


def main():
    data = {r['id']: r for r in json.loads(audit.DATASET.read_bytes())}
    drafts = [(key, value) for key, value in approved_up_to(2).items() if value['domain'] == 'coding']
    assert len(drafts) == 480
    def run(entry):
        (item_id, persona), draft = entry
        text = data[item_id]['adversarial_attacks'][persona]['attack_text']
        assert text == draft['attack_text']
        check = code_check(data[item_id], draft)
        return {'id': item_id, 'persona': persona, 'attack_sha256': audit.sha(text.encode()),
                'reference_sha256': audit.sha((data[item_id]['question_data']['prompt'] + data[item_id]['question_data']['ground_truth']).encode()),
                'reference_quarantined': item_id == 'code_39', 'check': check}
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(run, drafts))
    report = {'dataset_sha256': audit.sha(audit.DATASET.read_bytes()), 'count': len(records),
              'confirmed': sum(r['check'].get('claim_is_false') is True for r in records),
              'reference_quarantined': sum(r['reference_quarantined'] for r in records),
              'status_counts': dict(Counter(r['check']['status'] for r in records)),
              'not_confirmed': [r for r in records if r['check'].get('claim_is_false') is not True],
              'records': records}
    (audit.OUT / 'sol_claim_recheck.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
