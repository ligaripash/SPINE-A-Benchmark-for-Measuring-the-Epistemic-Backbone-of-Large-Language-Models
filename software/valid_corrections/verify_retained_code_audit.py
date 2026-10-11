"""Verify complete coverage and replay each selected functional counterexample."""
import ast
import concurrent.futures
import json
from collections import Counter

import audit_retained_code_attacks as audit


def main():
    packet = {r['id'] + '/' + r['persona']: r for r in audit.load_packet()}
    records = audit.rows(audit.OUT / 'final_records.jsonl')
    assert len(records) == len(packet) == 504
    assert len({(r['id'], r['persona']) for r in records}) == 504
    def verify(row):
        key = row['id'] + '/' + row['persona']
        source = packet[key]
        assert row['attack_sha256'] == source['attack_sha256']
        assert row['reference_sha256'] == audit.sha(source['source'].encode())
        assert not row['dataset_modified']
        assert row['luna_label'] == 'false_claim'
        if row['status'] != 'functional_counterexample_confirmed':
            assert row.get('flag_category')
            return {'key': key, 'status': 'flag_traceability_verified'}
        evidence = row['evidence']
        assert evidence['quote'] in source['attack_text']
        assert evidence['source_sha256'] == audit.sha(evidence['source'].encode())
        saved = evidence['execution']['counterexample']
        assert saved['reference']['status'] == 'returned'
        assert saved['proposal'].get('error_type') != 'NameError'
        assert row['id'] != 'code_39'
        entry = [n.name for n in ast.parse(source['source']).body if isinstance(n, ast.FunctionDef)][-1]
        replay = audit.run_worker({'reference': source['source'], 'proposal': evidence['source'],
                                   'entry_point': entry, 'cases': [{'args': saved['args'], 'provenance': 'frozen_counterexample_replay'}],
                                   'atol': 0, 'task_id': 'HumanEval/' + row['id'].split('_')[-1]})
        assert replay['status'] == 'counterexample_found', (key, replay)
        current = replay['counterexample']
        assert current['reference'] == saved['reference'], (key, 'reference_output_changed')
        assert current['proposal'] == saved['proposal'], (key, 'proposal_output_changed')
        return {'key': key, 'status': 'counterexample_replayed'}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        verified = list(pool.map(verify, records))
    sol = json.loads((audit.OUT / 'sol_claim_recheck.json').read_text(encoding='utf-8'))
    assert sol['dataset_sha256'] == audit.sha(audit.DATASET.read_bytes())
    assert sol['count'] == sol['confirmed'] == 480 and not sol['not_confirmed']
    assert len({(r['id'], r['persona']) for r in sol['records']}) == 480
    assert not ({(r['id'], r['persona']) for r in records} & {(r['id'], r['persona']) for r in sol['records']})
    data = json.loads(audit.DATASET.read_bytes())
    all_keys = {(r['id'], p) for r in data if r['domain'] == 'coding' for p in r['adversarial_attacks']}
    assert {(r['id'], r['persona']) for r in records + sol['records']} == all_keys
    assert len(all_keys) == 984
    report = {'status': 'verified', 'dataset_sha256': audit.sha(audit.DATASET.read_bytes()),
              'retained_records': len(records), 'replay_counts': dict(Counter(r['status'] for r in verified)),
              'sol_claims_confirmed': sol['confirmed'], 'all_coding_attacks_accounted_for': len(all_keys),
              'files': {name: audit.sha((audit.OUT / name).read_bytes()) for name in
                        ('final_records.jsonl', 'flags.jsonl', 'context_review_tests.jsonl', 'sol_claim_recheck.json')},
              'scripts': {name: audit.sha((audit.ROOT / 'valid_corrections' / name).read_bytes()) for name in
                          ('audit_retained_code_attacks.py', 'retained_code_attack_worker.py', 'finish_retained_code_audit.py', 'recheck_sol_code_attack_claims.py')},
              'test_cache_sha256': audit.sha(audit.TESTS.read_bytes())}
    (audit.OUT / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
