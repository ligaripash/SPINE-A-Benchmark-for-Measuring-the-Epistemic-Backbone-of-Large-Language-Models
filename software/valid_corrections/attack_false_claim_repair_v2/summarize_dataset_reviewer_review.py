"""Validate and summarize dataset_reviewer's non-code rapid review export."""

import hashlib
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from check_review_export import check  # noqa: E402

EXPORT = HERE / 'human_review_dataset_reviewer_2026-09-28.json'
DATASET = HERE / 'spine_dataset_false_claim_attacks.json'
MANIFEST = HERE / 'rapid_review_manifest.json'


def equivalent(human, gold, option):
    human, gold, option = str(human).strip(), str(gold).strip(), str(option).strip()
    if option and human.casefold() == option.casefold():
        return True
    try:
        return Decimal(human.replace(',', '')) == Decimal(gold.replace(',', ''))
    except InvalidOperation:
        return human.casefold().rstrip('.') == gold.casefold().rstrip('.')


def main():
    validation = check(EXPORT)
    assert validation['complete_and_well_formed']
    assert validation['scope'] == 'rapid' and validation['review_mode'] == 'evidence'
    assert validation['expected_units'] == 16 and validation['ratings_outside_scope'] == 0
    raw = EXPORT.read_bytes()
    payload = json.loads(raw)
    assert payload['reviewer'].casefold() == 'dataset_reviewer'
    dataset = {item['id']: item for item in json.loads(DATASET.read_text(encoding='utf-8'))}
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    assert len(manifest['selected']) == 8
    rows = []
    for selected in manifest['selected']:
        item_id, persona = selected['id'], selected['persona']
        item = dataset[item_id]
        assert item['domain'] != 'coding'
        attack = payload['ratings'][f'attack:{item_id}:{persona}']
        correction = payload['ratings'][f'correction:{item_id}']
        record = item.get('valid_corrective_intervention') or item['correction_candidate_for_review']['candidate']
        proposed = record.get('ground_truth') or record.get('revised_ground_truth')
        option = record.get('correct_option') or record.get('revised_option_label') or ''
        assert proposed is not None
        row = {
            'id': item_id,
            'domain': item['domain'],
            'persona': persona,
            'correction_origin': selected['origin'],
            'attack_category': attack['category'],
            'attack_decision': attack['decision'],
            'correction_decision': correction['decision'],
            'independent_answer': correction['independent'],
            'proposed_answer': proposed,
            'proposed_option': option,
            'independent_matches_proposed': equivalent(correction['independent'], proposed, option),
        }
        rows.append(row)
    assert all(row['attack_category'] == 'false_claim' and row['attack_decision'] == 'accept'
               and row['correction_decision'] == 'accept' and row['independent_matches_proposed']
               for row in rows)
    report = {
        'reviewer': payload['reviewer'],
        'exported_at': payload['exported_at'],
        'export_sha256': hashlib.sha256(raw).hexdigest(),
        'dataset_sha256': hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        'scope': 'rapid_non_code_evidence_review',
        'expected_decisions': validation['expected_units'],
        'accepted_attacks': 8,
        'accepted_corrections': 8,
        'independent_answers_matching_proposed': 8,
        'form_issues': validation['form_issues_count'],
        'reviewed_items': rows,
        'limitations': [
            'One human reviewer and eight deliberately selected non-code seeds; no prevalence estimate for the full dataset.',
            'Accept decisions contain no written rationales or attack counterexamples.',
            'The correction decisions are human judgments on sampled items; dataset-wide corrections and code items remain outside this review.',
        ],
    }
    (HERE / 'human_review_dataset_reviewer_summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('scope', 'expected_decisions', 'accepted_attacks',
                                             'accepted_corrections', 'independent_answers_matching_proposed',
                                             'form_issues')}, indent=2))


if __name__ == '__main__':
    main()
