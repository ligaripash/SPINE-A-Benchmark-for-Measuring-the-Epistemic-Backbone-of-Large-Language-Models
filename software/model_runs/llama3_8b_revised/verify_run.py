"""Audit exact revised-dataset inputs and response coverage."""

import hashlib
import json
from collections import Counter

from run import DATASET, HERE, PERSONAS, correction, messages, rows


def main():
    raw = DATASET.read_bytes()
    dataset = json.loads(raw)
    items = {item['id']: item for item in dataset}
    manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(raw).hexdigest() == manifest['dataset_sha256']
    base_rows = rows(HERE / 'base.jsonl')
    base = {row['id']: row for row in base_rows if 'error' not in row}
    followup_ids = set(json.loads((HERE / 'followup_ids.json').read_text(encoding='utf-8')))
    eligible = set(json.loads((HERE / 'eligible_ids.json').read_text(encoding='utf-8')))
    assert eligible <= followup_ids and len(followup_ids) == len(dataset)
    attack_rows = rows(HERE / 'attacks_reused.jsonl') + rows(HERE / 'attacks_new.jsonl')
    correction_rows = rows(HERE / 'corrections_reused.jsonl') + rows(HERE / 'corrections_new.jsonl')
    attacks = {(row['id'], row['branch']): row for row in attack_rows if 'error' not in row}
    corrections = {row['id']: row for row in correction_rows if 'error' not in row}
    expected_attacks = {(ident, persona) for ident in followup_ids for persona in PERSONAS}
    expected_corrections = {ident for ident in followup_ids if correction(items[ident])}
    duplicate_attacks = {f'{key[0]}/{key[1]}': count for key, count in
                         Counter((row['id'], row['branch']) for row in attack_rows if 'error' not in row).items()
                         if count > 1}
    duplicate_corrections = {key: count for key, count in
                             Counter(row['id'] for row in correction_rows if 'error' not in row).items()
                             if count > 1}
    errors = []
    for item in dataset:
        ident = item['id']
        anchor = base.get(ident)
        if not anchor or anchor['messages'] != [{'role': 'user', 'content': item['question_data']['prompt']}]:
            errors.append(f'baseline mismatch: {ident}')
            continue
        if ident not in followup_ids:
            continue
        for persona in PERSONAS:
            row = attacks.get((ident, persona))
            if row and row['messages'] != messages(item, anchor, persona):
                errors.append(f'attack conversation mismatch: {ident}/{persona}')
        if ident in expected_corrections:
            row = corrections.get(ident)
            if row and row['messages'] != messages(item, anchor, 'correction'):
                errors.append(f'correction conversation mismatch: {ident}')
            if row and row.get('correction_status') != item['valid_correction_status']:
                errors.append(f'correction status mismatch: {ident}')
    result = {
        'dataset_sha256': manifest['dataset_sha256'],
        'model_digest': manifest['model_digest'],
        'baseline_successes': len(base),
        'eligible_anchors': len(eligible),
        'followup_seeds': len(followup_ids),
        'attack_successes': len(attacks),
        'attack_expected': len(expected_attacks),
        'attack_missing': sorted(f'{a}/{b}' for a, b in expected_attacks - attacks.keys()),
        'attack_unexpected': sorted(f'{a}/{b}' for a, b in attacks.keys() - expected_attacks),
        'attack_duplicate_successes': duplicate_attacks,
        'attack_error_rows': sum('error' in row for row in attack_rows),
        'correction_successes': len(corrections),
        'correction_expected': len(expected_corrections),
        'correction_missing': sorted(expected_corrections - corrections.keys()),
        'correction_unexpected': sorted(corrections.keys() - expected_corrections),
        'correction_duplicate_successes': duplicate_corrections,
        'correction_error_rows': sum('error' in row for row in correction_rows),
        'conversation_errors': errors,
    }
    (HERE / 'integrity.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in result.items()}, indent=2))
    if (len(base) != len(dataset) or result['attack_missing'] or result['attack_unexpected'] or
            result['attack_duplicate_successes'] or result['correction_missing'] or
            result['correction_unexpected'] or result['correction_duplicate_successes'] or errors):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
