"""Validate reconciled human labels and compare, without modifying model metrics."""
import json,hashlib,shutil
from pathlib import Path
import analyze_science_human_review_v3 as a
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'model_runs/output_budget_v4/human_science_review_100'
OUT=BASE/'reconciliation'
load=a.load
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 packet=load(OUT/'packet.json');src=Path.home()/'Downloads/spine_science_reconciliation_dataset_reviewer.json';b=load(src)
 assert all(b[k]==packet[k] for k in ['schema','packet_id','role'])
 cards={c['review_id']:c for c in packet['cards']};assert b['sample_ids']==list(cards) and set(b['decisions'])==set(cards)
 for uid,d in b['decisions'].items():
  c=cards[uid];assert d['card_sha256']==c['card_sha256'] and d.get('saved_at') and d.get('joint_agreement') is True
  assert d['choice'] in c['option_labels']+['NONE','MULTIPLE','OTHER','UNCLEAR']
  assert d['choice']!='OTHER' or d.get('answer','').strip()
  ev=d.get('evidence','').strip();assert not ev or a.cosmetic(ev) in a.cosmetic(c['response'])
 sha=hashlib.sha256(src.read_bytes()).hexdigest();dest=OUT/'received_exports'/('reconciliation_'+sha+'.json');dest.parent.mkdir(exist_ok=True)
 if dest.exists():assert dest.read_bytes()==src.read_bytes()
 else:shutil.copy2(src,dest)
 original=load(BASE/'human_agreement_results.json');cross=load(BASE/'cross_review/comparison_results.json');private=load(BASE/'analyst_private.json')
 crossmap={r['review_id']:r for r in cross['records']};rows=[]
 for old in original['records']:
  uid=old['review_id'];d=b['decisions'].get(uid);choice=d['choice'] if d else old['human_choice'];human=a.correct(choice,private[uid]['target_option'])
  x=crossmap.get(uid)
  if x and not d:assert x['same_choice_category']
  rows.append({**old,'original_human_choice':old['human_choice'],'original_human_correct':old['human_correct'],'final_human_choice':choice,'final_human_correct':human,'final_agrees_ai':human==old['ai_correct'],'label_source':'reconciliation export' if d else 'two matching independent human labels' if x else 'original single review','reconciliation_recorder':b['reviewer'] if d else None,'second_human_choice':x['second_choice'] if x else None,'target_option':private[uid]['target_option']})
 def summary(rr):
  ids=[r['review_id'] for r in rr];hm={r['review_id']:r['final_human_correct'] for r in rr};ai={r['review_id']:r['ai_correct'] for r in rr};resolved=[i for i in ids if hm[i] is not None and ai[i] is not None]
  return {'n':len(rr),'agree':sum(r['final_agrees_ai'] for r in rr),'statistics':a.paired_stats(ids,hm,ai,{i:1 for i in ids}),'mutually_resolved':a.paired_stats(resolved,hm,ai,{i:1 for i in resolved})}
 result={'status':'reconciliation_complete','recorder_id':b['reviewer'],'all_11_joint_agreement_confirmations':True,'attribution_note':'Export records dataset_reviewer as recorder and joint-agreement confirmations on all 11. Original independent reviewers remain reviewer_a/reviewer_b; recorder identity is not rewritten.','source_export_sha256':sha,'original_agreement':original['correctness_category_agreement'],'after_reconciliation':summary(rows),'reconciled_11':summary([r for r in rows if r['review_id'] in cards]),'original_disputed_22':summary([r for r in rows if not r['agreement']]),'agreement_controls':summary([r for r in rows if r['review_id'] in crossmap and crossmap[r['review_id']]['selection']=='sampled_agreement']),'remaining_ai_differences':[r for r in rows if not r['final_agrees_ai']],'records':rows,'limitations':['Descriptive audit sample, not a corpus-wide accuracy estimate.','Post-reconciliation agreement is not independent inter-reviewer agreement.','Correctness comparison uses the existing task targets; exact AI option agreement is not measured.','No benchmark labels, metrics, or paper text changed.']}
 save(OUT/'comparison_results.json',result)
 lines=['# Human reconciliation comparison','','Recorder ID: '+b['reviewer']+'. All 11 decisions contain joint-agreement confirmation. Original independent labels remain preserved.','','| Original item | First label | Second label | Reconciled label | Human correctness | Original AI correctness |','|---|---|---|---|---|---|']
 label=lambda x:'unresolved' if x is None else 'correct' if x else 'incorrect'
 for r in rows:
  if r['review_id'] in cards:lines.append('| '+' | '.join(str(x) for x in [r['item'],r['original_human_choice'],r['second_human_choice'],r['final_human_choice'],label(r['final_human_correct']),label(r['ai_correct'])])+' |')
 lines+=['','Benchmark labels and metrics are unchanged.']
 (OUT/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k not in ['records','remaining_ai_differences','limitations']},indent=2))
 print('Remaining differences:',[(r['item'],r['final_human_choice'],r['final_human_correct'],r['ai_correct']) for r in result['remaining_ai_differences']])
 print('Reconciled:',[(r['item'],r['original_human_choice'],r['second_human_choice'],r['final_human_choice']) for r in rows if r['review_id'] in cards])
if __name__=='__main__':main()
