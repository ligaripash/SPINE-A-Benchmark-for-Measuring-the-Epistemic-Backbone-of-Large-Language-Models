"""Build a portable, local, single-file human review interface.

The HTML includes original data and machine flags, but never pre-fills a human
decision. Progress lives in browser local storage and can be exported as JSON.
"""
import hashlib
import json
import csv
from collections import defaultdict
from pathlib import Path
from candidate_store import load_candidates

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'full_dataset_review'
TEMPLATE = HERE / 'review_app_template.html'

raw = (ROOT/'spine_dataset.json').read_bytes()
source = json.loads(raw)
augmented = json.loads((ROOT/'spine_dataset_with_valid_corrections.json').read_text(encoding='utf-8'))
sample = json.loads((HERE/'sampled_evidence_review/manifest.json').read_text(encoding='utf-8'))
sample_ids = set(sample['sample_seed_ids'])
rapid = json.loads((HERE/'rapid_evidence_review/manifest.json').read_text(encoding='utf-8'))
rapid_attacks = {row['id']: row['persona'] for row in rapid['selected']}
assert len(source) == len(augmented) == 764
drafts = load_candidates()
smoke_path = HERE / 'code_candidate_smoke.jsonl'
smoke = {row['id']: row for row in (json.loads(line) for line in
         smoke_path.read_text(encoding='utf-8').splitlines() if line.strip())} if smoke_path.exists() else {}
pilot_report_path = HERE / 'validation_report.json'
pilot_checks = {row['id']: row for row in json.loads(pilot_report_path.read_text(encoding='utf-8')).get('checks', [])} if pilot_report_path.exists() else {}
machine_path = HERE / 'code_correction_machine_review.json'
machine_checks = {row['id']: row for row in json.loads(machine_path.read_text(encoding='utf-8')).get('records', [])} if machine_path.exists() else {}
flags = defaultdict(list)
audit_path = ROOT / 'original_validation/audit_index.csv'
audit_index = {}
if audit_path.exists():
    with audit_path.open(encoding='utf-8-sig', newline='') as f:
        audit_index = {row['id']: row for row in csv.DictReader(f)}
corrected_path = ROOT / 'spine_dataset_ground_truth_corrected.json'
corrected = {x['id']: x for x in json.loads(corrected_path.read_text(encoding='utf-8'))} if corrected_path.exists() else {}
corrected_sha256 = hashlib.sha256(corrected_path.read_bytes()).hexdigest() if corrected_path.exists() else ''
assert sample['population_sha256'] == corrected_sha256
assert rapid['source_sha256'] == corrected_sha256
correction_log_path = ROOT / 'original_validation/ground_truth_corrections.json'
gold_corrections_sha256 = hashlib.sha256(correction_log_path.read_bytes()).hexdigest() if correction_log_path.exists() else ''
with (HERE/'original_source_adjudication.csv').open(encoding='utf-8-sig', newline='') as f:
    source_concerns = {row['id']: row['audit_concern'] for row in csv.DictReader(f)}
with (OUT/'preflight_flags.csv').open(encoding='utf-8-sig',newline='') as f:
    for event in csv.DictReader(f):
        flags[(event['unit'],event['id'],event['persona'])].append(event['code'])
items=[]
for original, enriched in zip(source, augmented):
    assert original['id'] == enriched['id']
    integrated = corrected.get(original['id'])
    correction = integrated['valid_corrective_intervention'] if integrated else enriched['valid_corrective_intervention']
    correction_review = integrated['correction_candidate_for_review'] if integrated else drafts.get(original['id'])
    source_audit = audit_index.get(original['id'], {})
    original_gold = original['question_data']['ground_truth']
    revised_gold = corrected.get(original['id'], original)['question_data']['ground_truth']
    audit_status = source_audit.get('status', '')
    item = {
        'id':original['id'], 'domain':original['domain'],
        'retained': bool(integrated),
        'prompt':original['question_data']['prompt'],
        'gold':original_gold,
        'correctedGold':revised_gold if revised_gold != original_gold else '',
        'sourceAuditStatus':audit_status,
        'sourceAuditNote':source_audit.get('note', ''),
        'attacks':{p:original['adversarial_attacks'][p]['attack_text'] for p in original['adversarial_attacks']},
        'taskFlags':flags[('task',original['id'],'')] +
                    (['GROUND_TRUTH_CORRECTED_ATTACKS_NEED_REVIEW'] if revised_gold != original_gold else []) +
                    (['REMOVED_AMBIGUOUS_FROM_CORRECTED_DATASET'] if audit_status == 'excluded_ambiguous_from_corrected_dataset' else []) +
                    (['SOURCE_NEEDS_ADJUDICATION'] if audit_status == 'quarantine_requires_adjudication' else []) +
                    (['SOURCE_AUDIT_PRIORITY'] if original['id'] in source_concerns else []),
        'sourceConcern':source_concerns.get(original['id'],''),
        'attackFlags':{p:flags[('attack',original['id'],p)] for p in original['adversarial_attacks']},
        'inSample':original['id'] in sample_ids,
        'inRapid':original['id'] in rapid_attacks,
        'rapidAttack':rapid_attacks.get(original['id'], ''),
        'codeMachine':machine_checks.get(original['id'], {}),
    }
    if correction:
        correction_status = integrated['valid_correction_status'] if integrated else enriched['valid_correction_status']
        item['correction'] = {'text':correction['intervention_text'],
                              'gold':correction['ground_truth'],
                              'kind':correction['kind'], 'origin':'authored_pilot',
                              'flags':([] if correction_status == 'authored_pilot_pending_independent_review' else [correction_status]),
                              'recordStatus':correction_status,
                              'revisedQuestion':correction.get('revised_question',''),
                              'codeSmoke':({'status':'pilot_examples_passed',
                                            'revised_passes':pilot_checks[original['id']]['revised_passes'],
                                            'original_fails':pilot_checks[original['id']]['original_fails']}
                                           if original['domain']=='coding' and original['id'] in pilot_checks else {})}
    elif correction_review:
        row = correction_review
        candidate = row.get('candidate') or {}
        row_status = ('excluded_ambiguous_source' if audit_status == 'excluded_ambiguous_from_corrected_dataset'
                      else row.get('status', 'candidate_only_not_validated'))
        origin = ('invalidated_pilot' if row_status == 'invalidated_by_ground_truth_fix' else
                  'assistant_repair_draft' if row['generator']=='assistant_authored_repair' else
                  'generated_draft')
        item['correction'] = {
            'text': candidate.get('intervention_text',''),
            'gold': ((candidate.get('revised_reference_code','') or candidate.get('revised_ground_truth','') or candidate.get('ground_truth',''))
                    if original['domain']=='coding' else (candidate.get('revised_ground_truth','') or candidate.get('ground_truth',''))),
            'kind': candidate.get('kind',''),
            'origin': origin,
            'recordStatus':row_status,
            'unable': bool(candidate.get('unable')) or not candidate or row_status in {'invalidated_by_ground_truth_fix','excluded_ambiguous_source'},
            'reason': ('The pilot no longer changes the answer after the original ground-truth fix.' if row_status == 'invalidated_by_ground_truth_fix'
                       else 'This original seed was removed from the corrected dataset because its answer is ambiguous or underdetermined.' if row_status == 'excluded_ambiguous_source'
                       else candidate.get('reason','') or row.get('error') or ''),
            'revisedQuestion': candidate.get('revised_question',''),
            'derivation': candidate.get('derivation',''),
            'tests': (candidate.get('distinguishing_tests',[]) or candidate.get('test_cases',[])) if original['domain']=='coding' else [],
            'flags': row.get('screen_flags',[]) + ([row_status] if row_status not in {'candidate_only_not_validated','invalidated_by_ground_truth_fix'} else []),
            'codeSmoke': row.get('code_smoke') or smoke.get(original['id'],{}),
        }
    items.append(item)

def safe_json(obj):
    return json.dumps(obj,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')

html = TEMPLATE.read_text(encoding='utf-8')
html = html.replace('DATA_PLACEHOLDER',safe_json({'source_sha256':hashlib.sha256(raw).hexdigest(),
                                                  'corrected_sha256':corrected_sha256,
                                                  'gold_corrections_sha256':gold_corrections_sha256,
                                                  'ground_truth_correction_count':sum(bool(x['correctedGold']) for x in items),
                                                  'source_adjudication_count':sum(x['sourceAuditStatus'] in {'excluded_ambiguous_from_corrected_dataset','quarantine_requires_adjudication'} for x in items),
                                                  'sample_seed_count':len(sample_ids),
                                                  'rapid_seed_count':len(rapid_attacks),
                                                  'pilot_count':sum('correction' in x and x['correction']['origin']=='authored_pilot' for x in items),
                                                  'draft_count':sum('correction' in x and x['correction']['origin']!='authored_pilot' for x in items),
                                                  'items':items}))
(OUT/'review_spine.html').write_text(html,encoding='utf-8')
print(f"Built {OUT/'review_spine.html'} with a default rapid spot check of {len(rapid_attacks)} seeds, {len(rapid_attacks)} attacks, {rapid['human_correction_count']} human-reviewed corrections and {rapid['machine_only_code_correction_count']} machine-checked code corrections")
