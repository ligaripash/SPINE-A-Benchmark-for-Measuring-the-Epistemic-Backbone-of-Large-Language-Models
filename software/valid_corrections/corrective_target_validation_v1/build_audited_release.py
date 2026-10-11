"""Freeze eligible correction branches without modifying previous benchmark inputs.

AI-supported and stronger-evidence records are distinct. Excluded proposals are
archived outside the fields read by existing model runners. No model scores are
silently updated, and no follow-up message is repaired without a new response.
"""
from pathlib import Path
from collections import Counter
import copy
import hashlib
import json
import csv

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SUPPORTED={'human_supported','finite_test_supported','ai_supported'}
STRONGER={'human_supported','finite_test_supported'}

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def main():
    source=ROOT/'spine_dataset_false_claim_attacks.json'
    source_raw=source.read_bytes()
    data=json.loads(source_raw)
    audit=[json.loads(s) for s in (HERE/'target_audit.jsonl').read_text(encoding='utf-8').splitlines()]
    by_id={x['id']:x for x in audit}
    assert len(by_id)==len(data)==757
    release=[];excluded=[];stronger=[]
    for item in data:
        row=by_id[item['id']]
        edited=copy.deepcopy(item)
        c=item.get('valid_corrective_intervention') or (item.get('correction_candidate_for_review') or {}).get('candidate') or {}
        assert digest(c.get('intervention_text','').encode('utf-8'))==row['followup_sha256']
        if row['status'] in SUPPORTED:
            correction=copy.deepcopy(c)
            correction['ground_truth']=(c.get('revised_reference_code') or c.get('ground_truth','')) if item['domain']=='coding' else str(c.get('revised_ground_truth') or c.get('ground_truth','')).strip().lstrip('$').strip()
            if item['domain']=='coding':
                correction['entry_point']=row['code_checks']['entry_point']
                # Expected values from the independent oracle, with tuple/set
                # types preserved as repr rather than silently JSON-converted.
                correction['executable_oracle_file']='valid_corrections/corrective_target_validation_v1/independent_code_cases.py'
                correction['scoring']={'type':'independent_finite_functional_tests',
                    'require_all_tests':True,'item_key':item['id'],
                    'note':'Use the independent oracle; old candidate-derived tests are not the acceptance criterion.'}
            elif item['domain']=='logic':
                correction['correct_option']=row['option_checks']['label']
            correction['validation']={'status':row['status'],
                'independent_human_review':'accepted' if row['status']=='human_supported' else 'not_performed',
                'followup_sha256':row['followup_sha256'],'original_task_sha256':row['original_task_sha256'],
                'evidence':row['evidence'],'audit_file':'valid_corrections/corrective_target_validation_v1/target_audit.jsonl',
                'limitations':'AI support is not human validation; finite tests are not proofs.'}
            edited['valid_corrective_intervention']=correction
            edited['correction_candidate_for_review']=None
            edited['valid_correction_status']=row['status']
            if row['status'] in STRONGER:
                stronger.append(copy.deepcopy(edited))
        else:
            edited['excluded_corrective_proposal']=copy.deepcopy(c) if c else None
            edited['valid_corrective_intervention']=None
            edited['correction_candidate_for_review']=None
            edited['valid_correction_status']='excluded_'+row['status']
            excluded.append({'id':item['id'],'domain':item['domain'],'status':row['status'],
                             'issues':row['issues'],'followup_sha256':row['followup_sha256']})
        edited['corrective_target_audit']={'status':row['status'],'eligible_in_audited_release':row['status'] in SUPPORTED,
            'issues':row['issues'],'source_dataset_sha256':digest(source_raw)}
        assert edited['question_data']==item['question_data']
        assert edited['adversarial_attacks']==item['adversarial_attacks']
        release.append(edited)
    out=ROOT/'spine_dataset_corrective_branch_audited.json'
    raw=(json.dumps(release,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    out.write_bytes(raw)
    subset=ROOT/'spine_dataset_correction_evidence_subset.json'
    subset.write_text(json.dumps(stronger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (HERE/'excluded_corrections.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (HERE/'eligible_correction_ids.json').write_text(json.dumps(sorted(x['id'] for x in audit if x['status'] in SUPPORTED),indent=2)+'\n',encoding='utf-8')
    fields=('id','domain','status','eligible','human_review','independent_code_cases','issues')
    with (HERE/'review_index.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in audit:
            writer.writerow({'id':row['id'],'domain':row['domain'],'status':row['status'],
                'eligible':row['status'] in SUPPORTED,'human_review':row['status']=='human_supported',
                'independent_code_cases':row.get('code_checks',{}).get('independent_cases',0),
                'issues':';'.join(row['issues'])})
    arithmetic=json.loads((HERE/'arithmetic_evidence.json').read_text(encoding='utf-8'))
    summary={'source_dataset_sha256':digest(source_raw),'audited_release_sha256':digest(raw),
        'retained_seed_count':len(release),'eligible_corrective_branches':len(data)-len(excluded),
        'eligible_by_domain':dict(Counter(x['domain'] for x in audit if x['status'] in SUPPORTED)),
        'excluded_delivered_corrections':sum(x['status']!='missing' for x in excluded),
        'missing_corrections':sum(x['status']=='missing' for x in excluded),
        'status_counts':dict(Counter(x['status'] for x in audit)),
        'stronger_evidence_subset_count':len(stronger),
        'independent_code_cases':sum(x.get('code_checks',{}).get('independent_cases',0) for x in audit),
        'math_items_with_machine_recomputed_equations':sum(x['status']=='numeric_equations_recomputed' for x in arithmetic),
        'math_numeric_equations':sum(len(x['equations']) for x in arithmetic),
        'math_arithmetic_disagreements':sum(x['status']=='arithmetic_disagreement' for x in arithmetic),
        'new_model_or_verifier_api_requests':0,
        'previous_inputs_and_results_unchanged':source.read_bytes()==source_raw,
        'all_original_questions_references_and_attacks_preserved':True,
        'validation_claim':'AI-assisted screening with executable and sampled human evidence; not complete independent human validation.',
        'published_model_metrics_still_use_previous_correction_set':True}
    assert summary['eligible_corrective_branches']==702
    assert summary['excluded_delivered_corrections']==54
    assert summary['stronger_evidence_subset_count']==167
    (HERE/'release_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    report=f'''# Corrective branch validation audit

The current 757-seed dataset has 756 delivered corrective messages. This audit
checks those exact messages, current original references, and proposed targets.
It retains 702 correction branches in an **AI-assisted audited release** and
excludes 54 delivered branches. One further seed, `math_93`, has no usable draft.
No new target-model or verifier API requests were made.

| Domain | Retained correction branches | Excluded delivered branches | Missing |
| --- | ---: | ---: | ---: |
| Mathematics | 293 | 0 | 1 |
| Science reasoning | 250 | 49 | 0 |
| Code | 159 | 5 | 0 |
| Total | 702 | 54 | 1 |

## Evidence and its limits

* 535 records have blind AI support and structural/numerical checks; these are
  **not independently human-validated**. The existing GPT-6 Luna pass saw the
  original task/reference and actual follow-up, with revised targets hidden.
  All 756 follow-up hashes match this dataset. Its earlier full-dataset hash
  differs because subsequent repairs changed attack text, not corrections or
  original questions/references.
* 159 code records pass independently specified finite expected outputs and
  have a distinguishing change witness. Every retained code proposal was
  checked on the current references, including previously stale records.
  The oracle covers all 164 code seeds with 6,365 cases overall. Four previously
  specified property oracles are reused, and balanced whole-string brackets
  receive exhaustive checks up to length eight. Other cases are finite examples;
  passing does not prove all English requirements for every possible input.
* Eight records have dataset_reviewer's independent human answer confirmation: four math
  and four science. Human review never includes code. It is a qualitative spot
  check, not a statistical quality estimate for all records.
* A conservative arithmetic parser recomputed 326 numeric equations across
  192 math records, with zero remaining disagreements. The other 101 records
  do not contain a supported explicit numeric equation in the stored blind
  reasoning. Their answer agreement remains AI evidence; it is not claimed as
  complete independently executable semantic validation.
* The separate 167-record evidence subset includes the eight human-supported
  corrections and 159 code corrections with independent finite tests. These
  evidence types remain separately labeled even in that subset.

## Exclusions and adjudications

`code_22`, `code_49`, and `code_76` repeat requirements already satisfied by
the repaired original references. `logic_5` does not deliver the revised
question used by its proposed target. `code_39` remains quarantined because
its original reference is unresolved. `code_44` has an inconsistent revised
reference at zero and an underspecified preservation clause for negative inputs.

Science exclusions include absent revised options, incomplete measurement
questions, multiple plausible choices, or discrepancies between the delivered
task and proposed target. Examples are `logic_181` (release-rate question versus
a target about which factor changed), `logic_50` (the inherited option has a
false rationale), `logic_152` (weathering versus undelivered erosion wording),
and `logic_219` (slowing versus the inherited claim that friction stopped the ball).
These proposals are archived, not silently rewritten to match model answers.

Several flagged science records were supported after checking explicit option
replacements or equivalent wording. These assistant adjudications have written
reasons in `semantic_adjudications.json`; they are not labeled human review.

## Artifacts

* `../../spine_dataset_corrective_branch_audited.json`: all 757 original seeds
  and attacks, with only 702 active correction branches. Supported candidate
  targets are normalized into `valid_corrective_intervention`. Excluded
  proposals are outside the fields read by existing benchmark runners.
* `../../spine_dataset_correction_evidence_subset.json`: the 167-record subset
  with human or independently specified executable evidence.
* `target_audit.jsonl`: evidence and exact text/target hashes for every seed.
* `review_index.csv`: compact decisions and issues for every seed.
* `excluded_corrections.json`: all 55 seeds lacking an active correction.
* `blind_review_packet.jsonl`: all 756 delivered interventions, with proposed
  revised answers hidden, for further independent review.
* `independent_code_cases.py` and `code_execution/`: reproducible expectations
  and actual original/revised reference results.
* `arithmetic_evidence.json`: recomputed equations, with unsupported forms
  explicitly left unassessed.

## Scoring implications

The previous benchmark input, raw model outputs, comparison tables, and paper
are preserved. Their AUR/BES/Overall still use the old corrective set. Recompute
those metrics on this frozen eligibility list before reporting results for this
release. Code responses must use the independent oracle, rather than candidate
reference-derived expectations; the release marks this scoring requirement.
Repaired intervention text would require new model responses. Exclusion alone
does not require new generation. ESS on the original false-attack branches can
be reported alongside AUR, with both denominators and any difference in task
coverage disclosed. Do not compare this curated AUR directly to the old AUR as
if the evaluation sets were unchanged.

## Reproduce

Run `audit_corrective_targets.py`, then `check_arithmetic_evidence.py`, then
`build_audited_release.py` with the workspace Python runtime. Code executions
are checkpointed against the source, oracle cases, and audit-script hash.
The release summary records source and output SHA-256 hashes. Acceptance does
not depend on which evaluated model benefited from an item.
'''
    (HERE/'REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
