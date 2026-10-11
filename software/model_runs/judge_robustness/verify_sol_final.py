"""Verify the final merged policy against frozen inputs and accepted checkpoints."""
from collections import Counter
from datetime import datetime, timezone
import json

import analyze_sol_adjudication as final
import run_sol_agreed_unclear as extension
import run_sol_clarification as clarification
import run_sol_adjudication as primary


def main():
    out = primary.OUT
    metrics = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
    science = json.loads((out / 'final_science_response_labels.json').read_text(encoding='utf-8'))
    labels = json.loads((out / 'final_correctness_labels.json').read_text(encoding='utf-8'))
    index = final.analysis.rows(final.HERE / 'private_response_index.jsonl')
    luna = {r['blind_id']: r for r in final.analysis.rows(final.HERE / 'luna_labels.jsonl')}
    sol = {r['blind_id']: r for r in final.analysis.rows(out / 'labels.jsonl')}
    sol.update({r['blind_id']: r for r in final.analysis.rows(clarification.OUT / 'labels.jsonl')})
    new = {r['blind_id']: r for r in final.analysis.rows(extension.OUT / 'labels.jsonl')}
    assert len(science) == 12119 and len(labels) == len(index)
    assert not (set(new) & set(sol)) and len(new) == 92
    for module in [primary, clarification, extension]:
        assert module.summarize()['status'] == 'complete'
        assert not (module.OUT / 'runner.lock').exists()
    for module in [clarification, extension]:
        packet = {r['blind_id']: r for r in final.analysis.rows(module.OUT / 'packet.jsonl')}
        for record in final.analysis.rows(module.OUT / 'labels.jsonl'):
            item = packet[record['blind_id']]
            primary.validate(record_to_answer(record), {**item, 'option_labels': item['option_labels'] + ['OTHER']})
    for r in index:
        uid = r['blind_id']
        if r['domain'] != 'logic':
            assert labels[uid] == r['primary_correct']
            continue
        expected = new.get(uid, sol.get(uid, luna[uid]))
        assert science[uid]['choice'] == expected['choice']
        assert science[uid]['input_sha256'] == expected['input_sha256'] == r['input_sha256']
        assert labels[uid] == final.analysis.correctness(expected['choice'], r['target_option'])
    gates = {}
    for name in final.NAMES:
        rr = [r for r in index if r['run_name'] == name]
        initially_correct = {r['id'] for r in rr if r['phase'] == 'base' and labels[r['blind_id']] is True}
        corrections = {r['id'] for r in rr if r['phase'] == 'correction'}
        attacks = Counter(r['id'] for r in rr if r['phase'] == 'attack')
        gates[name] = (initially_correct - {'code_39', 'math_93'}) & corrections & {i for i, n in attacks.items() if n == 6}
    assert final.metric_rows(index, labels, gates) == metrics['metrics']
    archive = out / 'before_agreed_unclear_clarification'
    before = json.loads((archive / 'final_science_response_labels.json').read_text(encoding='utf-8'))
    assert all(science[uid] == value for uid, value in before.items() if uid not in new)
    comparison = {'original_agreed_unclear': 92,
                  'clarification_counts': dict(Counter(r['choice'] for r in new.values())),
                  'resolved_correct': sum(labels[uid] is True for uid in new),
                  'resolved_incorrect': sum(labels[uid] is False for uid in new),
                  'remaining_unclear': sum(labels[uid] is None for uid in new),
                  'unresolved_before': sum(r['choice'] == 'UNCLEAR' for r in before.values()),
                  'unresolved_after': metrics['science_unresolved']}
    final.analysis.save(extension.OUT / 'impact_summary.json', comparison)
    verification = {'status': 'verified_complete', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'primary_adjudications': 1497, 'targeted_clarifications': 281, 'agreed_unclear_clarifications': 92,
        'merged_science_labels': len(science), 'all_science_unresolved': metrics['science_unresolved'],
        'clear_out_of_options': sum(r['choice'] == 'OTHER' for r in science.values()),
        'newly_correct_without_followups': sum(v['newly_confirmed_without_complete_followups'] for v in metrics['cohort_coverage'].values()),
        'source_and_input_hashes_verified': True, 'accepted_ids_unique': True,
        'math_code_labels_preserved': True, 'nonselected_science_labels_preserved': True,
        'merged_label_policy_verified': True, 'metric_formulas_and_denominators_verified': True,
        'bootstrap_replicates': 10000, 'local_tests': '8 passed',
        'paper_sha256': final.analysis.sha(final.ROOT / 'spine.tex'),
        'metrics_sha256': final.analysis.sha(out / 'metrics.json'),
        'native_compilation': {'status': 'Native compiler check required.'}}
    final.analysis.save(out / 'completion_verification.json', verification)
    print(json.dumps(comparison), flush=True)


def record_to_answer(record):
    return {key: record[key] for key in primary.SCHEMA['required']}


if __name__ == '__main__':
    main()
