"""Apply 14 explicitly authorized human corrections as a new immutable result version."""
from pathlib import Path
import json,hashlib
import reevaluate_persona_v3 as api
ROOT=api.ROOT;SOURCE=ROOT/'model_runs/output_budget_v4';OUT=ROOT/'model_runs/human_adjudicated_v5'
REVIEW=SOURCE/'human_science_review_100'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=api.load(SOURCE/'final_verification.json')
for f,h in v['hashes'].items():assert sha(SOURCE/f)==h
review=api.load(REVIEW/'reconciliation/comparison_results.json');cards={r['review_id']:r for r in api.load(REVIEW/'primary_blinded.json')}
index=api.rows(SOURCE/'final_response_index.jsonl');lookup={r['blind_id']:r for r in index};labels=api.load(SOURCE/'final_correctness_labels.json');original=dict(labels)
changes=[]
for h in review['remaining_ai_differences']:
 uid=h['review_id'];r=lookup[uid];c=cards[uid]
 assert r['phase']!='base' and r['domain']=='logic'
 assert r['judge_input']['final_response']==c['response'] and r['judge_input']['task']==c['task']
 assert r['judge_input']['initial_answer']==c['initial_answer'] and r['judge_input']['followup_message']==c['followup_message']
 assert labels[uid] is h['ai_correct'] and h['final_human_correct'] is not None
 new=h['final_human_correct'];assert (h['final_human_choice']==r['target_option']) is new
 labels[uid]=new;r['primary_correct']=new;r['primary_method']='final_human_review'
 changes.append({'blind_id':uid,'run_name':r['run_name'],'id':r['id'],'branch':r['branch'],'phase':r['phase'],'old_correct':original[uid],'correct':new,'human_choice':h['final_human_choice'],'human_label_source':h['label_source'],'original_review_item':h['item'],'input_sha256':r['input_sha256'],'response_sha256':r['response_sha256']})
assert len(changes)==14 and sum(original[i]!=labels[i] for i in labels)==14
OUT.mkdir(exist_ok=True)
base=api.load(SOURCE/'metrics.json');gates={n:{r['id'] for r in index if r['run_name']==n and r['phase']=='correction'} for n in api.NAMES};common=set.intersection(*gates.values())
metrics=api.final.metric_rows(index,labels,gates)
result={**base,'metrics':metrics,'bootstrap':api.bootstrap(index,labels),'matched_cohort_metrics':api.final.metric_rows(index,labels,{n:common for n in api.NAMES}),'matched_cohort_bootstrap':api.bootstrap([r for r in index if r['id'] in common],labels),'completed_at':api.now(),'scoring_policy':'Final human labels override 14 audited science followups; other labels unchanged.','human_override_count':14}
api.save(OUT/'human_overrides.json',{'changes':changes,'source_metrics_sha256':v['hashes']['metrics.json'],'review_comparison_sha256':sha(REVIEW/'reconciliation/comparison_results.json'),'reconciliation_export_sha256':review['source_export_sha256'],'authorization':'User requested applying final human labels to the 14 differences.'})
api.write_rows(OUT/'final_response_index.jsonl',index);api.save(OUT/'final_correctness_labels.json',labels);api.save(OUT/'metrics.json',result)
assert set(labels)==set(lookup)
for n in api.NAMES:
 old=next(r for r in base['metrics'] if r['run_name']==n and r['domain']==r['attack_type']=='all');new=next(r for r in metrics if r['run_name']==n and r['domain']==r['attack_type']=='all')
 assert old['accuracy']==new['accuracy'] and old['attack_counts']['total']==new['attack_counts']['total'] and old['correction_counts']['total']==new['correction_counts']['total']
lines=['# SPINE results after final human review','','Fourteen science labels updated. Initial accuracy, eligibility, generation receipts, and prior results are unchanged.','','| Model | Accuracy | ESS | AUR | BES | Overall |','|---|---:|---:|---:|---:|---:|']
for r in metrics:
 if r['domain']==r['attack_type']=='all':lines.append('| '+api.final.SHORT[r['run_name']]+' | '+' | '.join(f'{100*r[k]:.2f}' for k in ['accuracy','ESS','AUR','BES','Overall'])+' |')
(OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
api.save(OUT/'final_verification.json',{'verified_at':api.now(),'hashes':{f:sha(OUT/f) for f in ['metrics.json','final_response_index.jsonl','final_correctness_labels.json','human_overrides.json']},'unique_records':len(index),'source_contexts_checked':True,'human_override_count':14,'bootstrap_replicates':10000,'prior_results_unchanged':all(sha(SOURCE/f)==h for f,h in v['hashes'].items())})
print('\n'.join(lines))
