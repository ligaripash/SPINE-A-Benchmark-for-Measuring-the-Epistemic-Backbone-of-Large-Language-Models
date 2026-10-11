"""Compare two AI item screens and route disagreements to human review."""
from __future__ import annotations

import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS = ('mistral-nemo_latest', 'qwen2.5_latest')
ELIGIBLE_ATTACK = {'false_claim', 'unsupported_pressure'}


def normalized_text(value):
    text = re.sub(r'^\s*[\(\[]?[A-D][\)\].:]\s*', '', str(value or ''), flags=re.I)
    return re.sub(r'\s+', ' ', text.lower().strip().rstrip('. '))


def normalized_number(value):
    match = re.fullmatch(r'\s*\$?\s*([-+]?\d+(?:\.\d+)?)\s*(?:dollars?|cups?|gallons?)?\s*', str(value or ''), flags=re.I)
    if not match:
        return None
    try:
        return Decimal(match.group(1))
    except InvalidOperation:
        return None


def proposed_gold(item):
    pilot = item.get('valid_corrective_intervention')
    if pilot:
        return pilot.get('ground_truth', '')
    candidate = (item.get('correction_candidate_for_review') or {}).get('candidate') or {}
    return candidate.get('revised_ground_truth') or candidate.get('ground_truth') or ''


def main():
    data = {x['id']: x for x in json.loads((ROOT / 'spine_dataset_ground_truth_corrected.json').read_text(encoding='utf-8'))}
    labels = {}
    for model in MODELS:
        path = HERE / f'ai_review_{model}_rapid/labels.jsonl'
        labels[model] = {(row['id'], row['unit'], row['persona']): row
                         for row in (json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip())}
    keys = sorted(set().union(*(set(v) for v in labels.values())))
    results = []
    for key in keys:
        item_id, unit, persona = key
        item = data[item_id]
        entries = {model: labels[model].get(key) for model in MODELS}
        reasons = []
        judge_summary = {}
        for model, row in entries.items():
            if not row:
                reasons.append(f'{model}:missing')
                continue
            output = {'label': row['label'], 'reason': row['parsed'].get('reason', '')}
            if unit == 'correction':
                answer = row['parsed'].get('revised_answer', '')
                output['revised_answer'] = answer
                if item['domain'] == 'math':
                    match = normalized_number(answer) == normalized_number(proposed_gold(item))
                elif item['domain'] == 'logic':
                    match = normalized_text(answer) == normalized_text(proposed_gold(item))
                else:
                    match = None
                output['answer_matches_proposed_gold'] = match
                if match is False:
                    reasons.append(f'{model}:revised_answer_mismatch')
                if row['label'] != 'valid_change':
                    reasons.append(f'{model}:nonvalid_correction_label')
            elif row['label'] not in ELIGIBLE_ATTACK:
                reasons.append(f'{model}:attack_not_confirmed_false_or_unsupported')
            judge_summary[model] = output
        if unit == 'attack' and len(judge_summary) == 2:
            a, b = (judge_summary[model]['label'] for model in MODELS)
            if a != b:
                reasons.append('judge_category_disagreement')
        if unit == 'correction':
            flags = (item.get('correction_candidate_for_review') or {}).get('screen_flags', [])
            if 'new_answer_not_original_option_text' in flags:
                reasons.append('proposed_answer_not_original_option_text')
        results.append({'id': item_id, 'domain': item['domain'], 'unit': unit,
                        'persona': persona,
                        'proposed_revised_gold': proposed_gold(item) if unit == 'correction' and item['domain'] != 'coding' else None,
                        'triage': 'human_priority' if reasons else 'AI_agreement_only',
                        'reasons': sorted(set(reasons)), 'judges': judge_summary})
    summary = {'model_ids': MODELS, 'total_items': len(results),
               'triage_counts': dict(Counter(x['triage'] for x in results)),
               'human_priority_by_unit': dict(Counter(x['unit'] for x in results if x['triage'] == 'human_priority')),
               'warning': 'Agreement is not validation. Derived-answer mismatches are review flags, not automatic gold errors.',
               'items': results}
    out = HERE / 'ai_review_rapid_comparison.json'
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'items'}, indent=2))
    print('Priority IDs:', [(x['id'], x['unit'], x['persona'], x['reasons']) for x in results if x['triage'] == 'human_priority'])


if __name__ == '__main__':
    main()
