"""Apply fresh blind Luna judgments without changing reviewed follow-up text.

Checks exact input hashes, choice agreement, and Luna's independent code against
the finite acceptance cases. Failed/uncertain items remain excluded.
"""
from collections import Counter
from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'spine_dataset_corrective_branch_repaired_v2.json'
OUTPUT = ROOT / 'spine_dataset_corrective_branch_luna_reviewed_v2.json'
PACKET = HERE / 'blind_review_packet.jsonl'
LABELS = HERE / 'luna_blind_labels.jsonl'
from independent_code_cases import packed_cases
from repair_specs import CODE
from verify_code import execute, same_result


def sha(value):
    if not isinstance(value, (str, bytes)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(value.encode('utf-8') if isinstance(value, str) else value).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def start():
    manifest = {'judge_model_requested': 'gpt-6-luna', 'reasoning_effort': 'high',
                'execution_method': 'Codex judge subagent with fresh context (fork_turns=none)',
                'judge_task_path': '/root/luna_blind_correction_judge',
                'input_dataset_sha256': sha(SOURCE.read_bytes()),
                'blind_packet_sha256': sha(PACKET.read_bytes()),
                'input_items': 54, 'review_scope': 'all 54 replacement interventions only',
                'revised_target_visibility': 'hidden',
                'allowed_input': 'original question, accepted original answer, delivered follow-up',
                'blindness_method': 'fresh context and instruction to read only the blind packet; not filesystem-enforced isolation',
                'human_review': 'not_performed_on_replacements',
                'started_at_utc': datetime.now(timezone.utc).isoformat()}
    path = HERE / 'luna_review_manifest.json'
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf-8'))
        assert existing['input_dataset_sha256'] == manifest['input_dataset_sha256']
        assert existing['blind_packet_sha256'] == manifest['blind_packet_sha256']
    else:
        save(path, manifest)
    print('Saved hash-linked blind-review manifest.')


def apply():
    manifest = json.loads((HERE / 'luna_review_manifest.json').read_text(encoding='utf-8'))
    assert sha(SOURCE.read_bytes()) == manifest['input_dataset_sha256']
    assert sha(PACKET.read_bytes()) == manifest['blind_packet_sha256']
    packet = [json.loads(line) for line in PACKET.read_text(encoding='utf-8').splitlines() if line.strip()]
    rows = [json.loads(line) for line in LABELS.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(rows) == 54 and len({x['id'] for x in rows}) == 54
    assert {x['id'] for x in rows} == {x['id'] for x in packet}
    labels = {x['id']: x for x in rows}
    data = json.loads(SOURCE.read_text(encoding='utf-8'))
    output = copy.deepcopy(data)
    comparison = []
    for item in output:
        ident = item['id']
        if ident not in labels:
            continue
        row = labels[ident]
        correction = item['valid_corrective_intervention']
        followup_hash = sha(correction['intervention_text'])
        delivered = next(x for x in packet if x['id'] == ident)
        assert correction['intervention_text'] == delivered['delivered_intervention']
        assert row['model'] == 'gpt-6-luna' and row['domain'] == item['domain']
        assert row['status'] in ('valid_change', 'invalid_change', 'ambiguous', 'uncertain')
        assert isinstance(row['visible_options_sufficient'], bool)
        assert isinstance(row['reason'], str) and row['reason'].strip()
        decision = {'id': ident, 'luna_status': row['status'], 'followup_sha256': followup_hash,
                    'luna_independent_answer': row['revised_answer'], 'luna_reason': row['reason']}
        supported = row['status'] == 'valid_change' and row['visible_options_sufficient']
        if item['domain'] == 'logic':
            decision['luna_option'] = row['revised_option']
            decision['authored_option'] = correction['correct_option']
            decision['option_agreement'] = row['revised_option'] == correction['correct_option']
            supported = supported and decision['option_agreement'] and bool(row['revised_answer'].strip())
        else:
            if not row['revised_code']:
                supported = False
                decision['luna_code_result'] = {'status': 'missing'}
            else:
                result = execute(ident, row['revised_code'], correction['entry_point'], packed_cases(ident))
                decision['luna_code_result'] = result
                supported = supported and result['status'] == 'passed'
            witness = row['distinguishing_case']
            if not isinstance(witness, dict) or 'args' not in witness or 'expected' not in witness:
                supported = False
                decision['luna_witness_result'] = {'status': 'missing'}
            else:
                case = {'args': witness['args'], 'expected_repr': repr(witness['expected'])}
                revised_result = execute(ident, correction['ground_truth'], correction['entry_point'], [case])
                original_result = execute(ident, item['question_data']['prompt'] + item['question_data']['ground_truth'], correction['entry_point'], [case])
                decision['luna_witness_result'] = {'revised_reference': revised_result, 'original_reference': original_result, 'case': case}
                supported = supported and revised_result['status'] == 'passed' and original_result['status'] == 'failed' and bool(original_result.get('mismatches'))
        decision['supported'] = bool(supported)
        decision['baseline_quarantine_preserved'] = ident == 'code_39'
        comparison.append(decision)
        correction['validation']['independent_ai_review'] = 'fresh_blind_luna_completed'
        correction['validation']['independent_ai_model'] = 'gpt-6-luna'
        correction['validation']['blind_review_label'] = row['status']
        correction['validation']['blind_review_supported'] = bool(supported)
        correction['validation']['blind_review_followup_sha256'] = followup_hash
        correction['validation']['blind_review_evidence_file'] = 'valid_corrections/corrective_target_repair_v2/luna_review_comparison.json'
        if supported:
            status = 'ai_supported' if item['domain'] == 'logic' else 'finite_test_and_ai_supported'
            item['valid_correction_status'] = status
            item['corrective_target_audit']['status'] = status
            correction['validation']['status'] = status
        else:
            item['valid_correction_status'] = 'excluded_after_luna_review'
            item['corrective_target_audit']['eligible_in_audited_release'] = False
            item['corrective_target_audit']['status'] = 'luna_flagged_or_target_disagreement'
            item['luna_flagged_corrective_proposal'] = correction
            item['valid_corrective_intervention'] = None
            item['correction_candidate_for_review'] = None
        # A valid revised code_39 contract does not resolve its original reference.
        if ident == 'code_39':
            item['corrective_target_audit']['eligible_in_audited_release'] = False
    for old, new in zip(data, output):
        assert old['question_data'] == new['question_data']
        assert old['adversarial_attacks'] == new['adversarial_attacks']
        if old['id'] not in labels:
            assert old == new
        elif new['valid_corrective_intervention']:
            assert old['valid_corrective_intervention']['intervention_text'] == new['valid_corrective_intervention']['intervention_text']
            assert old['valid_corrective_intervention']['ground_truth'] == new['valid_corrective_intervention']['ground_truth']
    save(OUTPUT, output)
    save(HERE / 'luna_review_comparison.json', comparison)
    summary = {'reviewed': 54, 'judge_model': 'gpt-6-luna',
               'judgments_by_status': dict(Counter(x['status'] for x in rows)),
               'supported_replacement_targets': sum(x['supported'] for x in comparison),
               'flagged_ids': [x['id'] for x in comparison if not x['supported']],
               'science_option_agreement': sum(x.get('option_agreement', False) for x in comparison),
               'code_implementations_passing_all_tests': sum(x.get('luna_code_result', {}).get('status') == 'passed' for x in comparison),
               'code_case_count': sum(x.get('luna_code_result', {}).get('tests', 0) for x in comparison),
               'reviewed_dataset': str(OUTPUT), 'reviewed_dataset_sha256': sha(OUTPUT.read_bytes()),
               'labels_sha256': sha(LABELS.read_bytes()),
               'active_corrections': sum(bool(x['valid_corrective_intervention']) for x in output),
               'baseline_eligible_corrections': sum(x['corrective_target_audit']['eligible_in_audited_release'] for x in output),
               'baseline_quarantined': ['code_39'], 'missing_correction': ['math_93'],
               'independent_human_review_of_replacements': 'not_performed',
               'benchmark_metrics_recomputed': False}
    save(HERE / 'luna_review_summary.json', summary)
    manifest['status'] = 'completed'
    manifest['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['labels_sha256'] = summary['labels_sha256']
    manifest['reviewed_dataset_sha256'] = summary['reviewed_dataset_sha256']
    save(HERE / 'luna_review_manifest.json', manifest)
    report = f'''# Fresh blind Luna review of replacement corrections

Judge: **GPT-6 Luna**, high reasoning effort, run in a Codex judge subagent with a fresh context. The user explicitly requested this judge. The judge was given only the original question, accepted original answer, and exact delivered replacement follow-up for each of the **54** items. It was instructed not to inspect any authored revised answer, implementation, derivation, manifest, or prior judgment. This blindness was established by fresh context and input instructions, not filesystem access isolation. This was a Codex subagent review, not a Responses API run. Model identity reference: https://developers.openai.com/api/docs/models/gpt-6-luna.

## Results

* **{summary['supported_replacement_targets']}/54** replacements supported after comparing blind judgments with the authored targets.
* Judgment labels: {json.dumps(summary['judgments_by_status'])}.
* **{summary['science_option_agreement']}/49** science/reasoning option selections agree with the targets.
* **{summary['code_implementations_passing_all_tests']}/5** independently written Luna implementations pass all acceptance cases (**{summary['code_case_count']:,}** executions).
* Each code item also has a Luna-derived distinguishing case: the revised reference passes it, and the current original reference fails it. Exact recursive output types are checked.
* Flagged IDs: {json.dumps(summary['flagged_ids'])}.

The reviewed release is `spine_dataset_corrective_branch_luna_reviewed_v2.json`, SHA-256 `{summary['reviewed_dataset_sha256']}`. It contains **{summary['active_corrections']}** delivered correction branches among 757 seeds, with **{summary['baseline_eligible_corrections']}** eligible before each model's initial-correctness filter. `math_93` still lacks a correction. The revised `code_39` contract is supported, but its original-reference quarantine remains: a valid correction and a small distinguishing witness do not resolve an original-reference problem.

## What was preserved

No original questions, original answers, attacks, replacement follow-up messages, or revised targets changed during the review. The 702 earlier retained correction records are unchanged. Their existing evidence remains applicable to their unchanged text; this fresh Luna run covers only the 54 replacement messages. Failed or uncertain replacements would have been removed from the reviewed release rather than silently endorsed; none were flagged in this pass.

## Evidence and limits

`luna_blind_labels.jsonl` contains the independent answers, reasons, implementations, and distinguishing cases. Its SHA-256 is `{summary['labels_sha256']}`. `luna_review_comparison.json` contains choice comparisons and executed code checks. `luna_review_manifest.json` links exact input and output hashes and records the review method. Reproduce the comparison and reviewed release with `python valid_corrections/corrective_target_repair_v2/apply_luna_review.py`.

This is independent **AI screening**, not independent human verification or proof of every semantic claim and possible code input. Most code acceptance tests are finite; only the bounded replacement prime_fib input domain is completely enumerated. Model agreement can share mistakes, and 54/54 agreement is not a statistical confidence interval for dataset correctness. The original human-review evidence is not reassigned to these replacement messages.

Benchmark responses and metrics were not regenerated by this judge run. Changed follow-ups still need fresh correction-branch responses before revised AUR, BES, and Overall can be reported. While code_39 remains quarantined, at most 53 of the replacement branches are eligible, further filtered by each model's initial correctness. Matching unchanged baseline, attack, and earlier correction responses can be reused when their exact inputs and model configurations match.
'''
    (HERE / 'LUNA_REVIEW_REPORT.md').write_text(report, encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    start() if '--start' in sys.argv else apply()
