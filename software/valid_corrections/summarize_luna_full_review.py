"""Triage the full Luna AI screen; do not promote labels to validated gold."""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

from ai_item_review import correction
from summarize_ai_item_review import normalized_text, proposed_gold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LABELS = HERE / 'ai_review_gpt-6-luna_full/labels.jsonl'
OUT = HERE / 'ai_review_gpt-6-luna_full_summary.json'
ELIGIBLE_ATTACK = {'false_claim', 'unsupported_pressure'}


def number(value):
    s = str(value or '').strip().replace(',', '')
    match = re.fullmatch(r'\$?\s*([-+]?\d+(?:\.\d+)?)\s*(?:[a-zA-Z%²³/ -]+)?', s)
    if not match:
        return None
    try:
        return Decimal(match.group(1))
    except InvalidOperation:
        return None


def option_text(value):
    s = unicodedata.normalize('NFKC', str(value or ''))
    s = s.replace('→', '->').replace('⟶', '->')
    return normalized_text(s)


def main():
    data = {x['id']: x for x in json.loads(
        (ROOT / 'spine_dataset_ground_truth_corrected.json').read_text(encoding='utf-8'))}
    rows = [json.loads(line) for line in LABELS.read_text(encoding='utf-8').splitlines()
            if line.strip()]
    keys = [(r['id'], r['unit'], r['persona']) for r in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit('Duplicate judgment keys')
    flags = []
    summary = Counter()
    for r in rows:
        item = data[r['id']]
        domain, unit, label = r['domain'], r['unit'], r['label']
        summary[(domain, unit, label)] += 1
        reasons = []
        candidate_flags = []
        expected = proposed_gold(item) if unit == 'correction' else None
        derived = r['parsed'].get('revised_answer') if unit == 'correction' else None
        answer_match = None
        if unit == 'attack':
            if label not in ELIGIBLE_ATTACK:
                reasons.append('attack_not_false_or_unsupported_pressure')
        else:
            if label != 'valid_change':
                reasons.append('correction_not_confirmed_valid')
            if domain == 'math':
                actual_num, gold_num = number(derived), number(expected)
                if actual_num is not None and gold_num is not None:
                    answer_match = actual_num == gold_num
                else:
                    reasons.append('answer_not_machine_comparable')
            elif domain == 'logic':
                answer_match = option_text(derived) == option_text(expected)
                if not r['parsed']['visible_options_sufficient']:
                    reasons.append('visible_options_insufficient')
            if answer_match is False:
                reasons.append('derived_answer_not_string_equivalent_to_proposal')
            candidate_flags = (item.get('correction_candidate_for_review') or {}).get('screen_flags', [])
        if reasons or candidate_flags:
            flags.append({'id': r['id'], 'domain': domain, 'unit': unit,
                          'persona': r['persona'], 'label': label,
                          'model_review_flags': reasons,
                          'existing_candidate_screen_flags': candidate_flags,
                          'derived_answer': derived,
                          'proposed_answer': expected if domain != 'coding' else None,
                          'reason': r['parsed'].get('reason'),
                          'concrete_check': r['parsed'].get('concrete_check')})
    report = {
        'model': 'gpt-6-luna',
        'status': 'AI triage only; not human or exhaustive semantic validation',
        'comparison_note': 'Answer string differences include paraphrases and option-label formatting; inspect before treating as a gold mismatch.',
        'seed_count': len(data),
        'seeds_without_correction_text': [x['id'] for x in data.values() if not correction(x)],
        'judgments_completed': len(rows),
        'expected_judgments': 5298,
        'missing_judgments': 5298-len(rows),
        'label_counts': {'/'.join(k): v for k, v in sorted(summary.items())},
        'model_flag_count': sum(bool(x['model_review_flags']) for x in flags),
        'existing_candidate_screen_flag_count': sum(bool(x['existing_candidate_screen_flags']) for x in flags),
        'model_flag_reasons': dict(Counter(reason for x in flags for reason in x['model_review_flags'])),
        'existing_candidate_screen_reasons': dict(Counter(reason for x in flags for reason in x['existing_candidate_screen_flags'])),
        'flags': flags,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'flags'}, indent=2))


if __name__ == '__main__':
    main()
