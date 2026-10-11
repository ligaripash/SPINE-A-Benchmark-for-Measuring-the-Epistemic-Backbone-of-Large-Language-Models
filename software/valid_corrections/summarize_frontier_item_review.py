"""Summarize a blinded OpenAI-model pilot without promoting AI labels to gold."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from summarize_ai_item_review import normalized_number, normalized_text, proposed_gold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOCAL = HERE / 'ai_review_rapid_comparison.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='gpt-6-luna')
    args = p.parse_args()
    frontier = HERE / f'ai_review_{args.model}_rapid/labels.jsonl'
    out = HERE / f'ai_review_{args.model}_rapid_comparison.json'
    data = {item['id']: item for item in json.loads(
        (ROOT / 'spine_dataset_ground_truth_corrected.json').read_text(encoding='utf-8'))}
    local = {(row['id'], row['unit'], row['persona']): row
             for row in json.loads(LOCAL.read_text(encoding='utf-8'))['items']}
    rows = [json.loads(line) for line in frontier.read_text(encoding='utf-8').splitlines()
            if line.strip()]
    items = []
    for row in rows:
        key = (row['id'], row['unit'], row['persona'])
        item = data[row['id']]
        flags = []
        matching_gold = None
        if row['unit'] == 'attack':
            if row['label'] not in {'false_claim', 'unsupported_pressure'}:
                flags.append('attack_not_confirmed_false_or_unsupported')
        else:
            if row['label'] != 'valid_change':
                flags.append('correction_not_confirmed_valid')
            answer = row['parsed']['revised_answer']
            if item['domain'] == 'math':
                matching_gold = normalized_number(answer) == normalized_number(proposed_gold(item))
            elif item['domain'] == 'logic':
                matching_gold = normalized_text(answer) == normalized_text(proposed_gold(item))
                if not row['parsed']['visible_options_sufficient']:
                    flags.append('target_visible_options_insufficient')
            if matching_gold is False:
                flags.append('independent_answer_mismatch')
        existing = local.get(key)
        if existing:
            labels = {judge['label'] for judge in existing['judges'].values()}
            if row['label'] not in labels:
                flags.append('frontier_differs_from_both_local_judges')
        items.append({'id': row['id'], 'domain': row['domain'], 'unit': row['unit'],
                      'persona': row['persona'], 'frontier_label': row['label'],
                      'frontier_answer': row['parsed'].get('revised_answer'),
                      'matches_proposed_gold': matching_gold,
                      'proposed_gold': proposed_gold(item) if row['unit'] == 'correction' and item['domain'] != 'coding' else None,
                      'frontier_reason': row['parsed']['reason'],
                      'frontier_check': row['parsed']['concrete_check'],
                      'local_labels': {model: value['label'] for model, value in existing['judges'].items()} if existing else {},
                      'review_flags': flags})
    report = {'frontier_model': args.model, 'rows_completed': len(rows),
              'priority_count': sum(bool(x['review_flags']) for x in items),
              'label_counts': dict(Counter(f"{x['unit']}:{x['frontier_label']}" for x in items)),
              'warning': 'AI agreement is screening evidence, not human validation or semantic proof.',
              'items': items}
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'items'}, indent=2))
    for x in items:
        if x['review_flags']:
            print(x['id'], x['unit'], x['persona'], x['review_flags'])


if __name__ == '__main__':
    main()
