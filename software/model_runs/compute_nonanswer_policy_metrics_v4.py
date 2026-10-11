"""Provisional effect of the approved 94 non-answer labels; not the budget rerun."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'model_runs'))
import reevaluate_persona_v3 as runner
r=ROOT/'model_runs/output_budget_v4';old=ROOT/'model_runs/persona_revision_v3'
idx=runner.rows(old/'final_response_index.jsonl')
labels=runner.load(old/'final_correctness_labels.json')
manifest=runner.load(old/'run_manifest.json')
lookup={(v['run_name'],v['id']):v for v in idx if v['phase']=='base'}
gates={n:set(v) for n,v in manifest['eligible_ids'].items()}
changes=[]
for v in runner.load(r/'bounded_budget_nonanswer_labels.json')['labels']:
 x=lookup[(v['run_name'],v['id'])];uid=x['blind_id']
 changes.append({'run_name':v['run_name'],'id':v['id'],'previous_correct':labels[uid],'correct':False})
 labels[uid]=False
 gates[v['run_name']].discard(v['id'])
metrics=runner.final.metric_rows(idx,labels,gates)
result={'status':'provisional_policy_only','scope':'Apply 94 approved non-answer initial labels to frozen v3. Does NOT incorporate 387 unscored recovered answers or pending API/followup reruns. No new bootstrap intervals calculated.','changes':changes,'metrics':metrics}
(r/'provisional_nonanswer_policy_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('previously_correct',sum(x['previous_correct'] is True for x in changes))
for m in metrics:
 if m['domain']==m['attack_type']=='all':
  print(m['run_name'],*[round(100*m[k],2) for k in ['accuracy','ESS','AUR','BES','Overall']])
