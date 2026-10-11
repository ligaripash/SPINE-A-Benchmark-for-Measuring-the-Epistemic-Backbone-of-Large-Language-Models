"""Prepare gold-blind item cards and separate reviewer forms for the 60-item pilot.

The cards omit assistant-authored reference answers, rationales, code solutions,
test cases, attack text, model identities, and model outputs. Reviewers should
record independent answers before opening pilot_dataset.json to check the key.
"""
import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
pilot = json.loads((HERE / 'pilot_dataset.json').read_text(encoding='utf-8'))
cards = []
for item in pilot:
    correction = item['valid_corrective_intervention']
    cards.append({
        'id': item['id'],
        'domain': item['domain'],
        'original_question': item['question_data']['prompt'],
        'followup_user_message': correction['intervention_text'],
    })

random.Random(20260927).shuffle(cards)
out = HERE / 'human_review'
out.mkdir(exist_ok=True)
(out / 'blind_item_cards.json').write_text(json.dumps(cards, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

fields = [
    'id', 'reviewer_id', 'original_answer_independent',
    'revised_answer_independent', 'original_answer_correct_under_original',
    'new_information_or_specification_is_explicit',
    'original_answer_invalid_under_revision', 'revised_answer_unique_and_checkable',
    'internally_consistent', 'minimal_change', 'overwrites_multiple_choice_options',
    'kind', 'supported_claims', 'decision', 'issue_codes', 'explanation',
]
for reviewer in ('A', 'B'):
    path = out / f'item_review_{reviewer}.csv'
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for card in cards:
            writer.writerow({'id':card['id'], 'reviewer_id':reviewer})
print(f'Wrote {len(cards)} blind cards and two separate review forms to {out}')
