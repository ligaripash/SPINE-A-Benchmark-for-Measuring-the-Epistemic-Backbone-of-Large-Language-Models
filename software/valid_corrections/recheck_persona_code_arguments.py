"""Offline recheck of saved candidates under signature-bound argument comparison."""
import copy
import json
from collections import Counter

import run_persona_attack_fixer as fix
from verify_persona_attack_fixes import alive, verify


def main():
    if alive():
        raise SystemExit('Fixer is active; stop concurrent rechecking.')
    _, data, items, rows, config = fix.prepare()
    selected, _, _ = fix.selection(rows)
    before = fix.approvals(selected, fix.MAX_ROUNDS)
    protocol = {
        'policy': fix.call_args.POLICY,
        'source_dataset_sha256': config['source_dataset_sha256'],
        'normalizer_sha256': fix.persona.sha((fix.ROOT / 'valid_corrections/code_argument_normalization.py').read_bytes()),
        'scope': 'Offline argument normalization only. Original API receipts, attacks, factual and persona judgments remain unchanged. New acceptance requires both existing Luna validations and executable disproof.',
    }
    protocol_path = fix.OUT / 'code_argument_recheck_policy_v2.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding='utf-8')) == protocol
    else:
        fix.persona.write_json(protocol_path, protocol)
    count = 0
    for row in selected:
        if row['review_id'] in before or row['domain'] != 'coding':
            continue
        for number in range(1, fix.MAX_ROUNDS + 1):
            original = fix.outcome_path(row['review_id'], number)
            if not original.exists():
                continue
            path = fix.OUT / f'code_argument_rechecks_v2_r{number}' / (row['review_id'] + '.json')
            if path.exists():
                fix.read_outcome(row['review_id'], number)
                continue
            result = json.loads(original.read_text(encoding='utf-8'))
            if not result.get('draft') or 'factual_validation' not in result:
                continue
            updated = copy.deepcopy(result)
            updated['code_validation'] = fix.verify_code(items[row['seed_id']], result['draft'], result['factual_validation'])
            updated['accepted'] = bool(result['factual_validation']['false_claim_accepted']
                                       and result['persona_validation']['alignment'] == 'aligned'
                                       and updated['code_validation']['accepted'])
            updated['acceptance_recheck'] = {'policy': fix.call_args.POLICY,
                'original_outcome_sha256': fix.persona.sha(original.read_bytes()),
                'timestamp_utc': fix.persona.now()}
            updated['recheck_file'] = str(path.relative_to(fix.ROOT))
            path.parent.mkdir(exist_ok=True)
            fix.persona.write_json(path, updated)
            count += 1
        if count and count % 30 == 0:
            print(f'Offline saved-candidate checks: {count}', flush=True)
    after = fix.approvals(selected, fix.MAX_ROUNDS)
    summary = fix.publish(data, rows, config, fix.MAX_ROUNDS)
    record = {'accepted_before': len(before), 'accepted_after': len(after),
              'newly_accepted_ids': sorted(after.keys() - before.keys()),
              'pending_repairs': summary['pending_repairs'], 'rechecked_candidates': count,
              'timestamp_utc': fix.persona.now()}
    fix.persona.append(fix.OUT / 'code_argument_recheck_runs.jsonl', record)
    verify()
    failures = Counter()
    for row in selected:
        if row['review_id'] in after:
            continue
        numbers = [n for n in range(1, fix.MAX_ROUNDS + 1) if fix.outcome_path(row['review_id'], n).exists()]
        result = fix.read_outcome(row['review_id'], max(numbers))
        if not result.get('factual_validation', {}).get('false_claim_accepted', False):
            failures['factual'] += 1
        if result.get('persona_validation', {}).get('alignment') != 'aligned':
            failures['persona'] += 1
        if not result.get('code_validation', {}).get('accepted', False):
            failures['code'] += 1
    print(json.dumps({**record, 'remaining_check_failures_overlap': dict(failures)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
