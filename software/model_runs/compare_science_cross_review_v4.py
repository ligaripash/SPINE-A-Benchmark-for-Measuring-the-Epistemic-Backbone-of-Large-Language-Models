"""Validate independent cross-review exports; compare without changing benchmark labels."""
import json,hashlib,shutil
from collections import Counter
from pathlib import Path
import analyze_science_human_review_v3 as a
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'model_runs/output_budget_v4/human_science_review_100'
OUT=SOURCE/'cross_review'
load=a.load
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 manifest=load(OUT/'analyst_manifest.json');original=load(SOURCE/'human_agreement_results.json');private=load(SOURCE/'analyst_private.json')
 assert hashlib.sha256((SOURCE/'human_agreement_results.json').read_bytes()).hexdigest()==manifest['source_analysis_sha256']
 prior={r['review_id']:r for r in original['records']};records=[]
 for reviewer in ['reviewer_a','reviewer_b']:
  packet=load(OUT/(reviewer.lower()+'_blinded.json'))
  src=Path.home()/'Downloads'/f'spine_science_cross_{reviewer.lower()}_{reviewer}.json'
  b=load(src);assert all(b[k]==packet[k] for k in ['schema','packet_id','role']);assert b['reviewer']==reviewer
  cards={c['review_id']:c for c in packet['cards']};assert b['sample_ids']==list(cards);assert set(b['decisions'])==set(cards)
  for uid,d in b['decisions'].items():
   c=cards[uid];assert d['card_sha256']==c['card_sha256'];assert d['choice'] in c['option_labels']+['NONE','MULTIPLE','OTHER','UNCLEAR'];assert d.get('saved_at')
   assert d['choice']!='OTHER' or d.get('answer','').strip()
   evidence=d.get('evidence','').strip();assert not evidence or a.cosmetic(evidence) in a.cosmetic(c['response'])
  sha=hashlib.sha256(src.read_bytes()).hexdigest();dest=OUT/'received_exports'/(reviewer.lower()+'_'+sha+'.json');dest.parent.mkdir(exist_ok=True)
  if dest.exists():assert dest.read_bytes()==src.read_bytes()
  else:shutil.copy2(src,dest)
  save(OUT/(reviewer.lower()+'_completion.json'),{'reviewer':reviewer,'completed':len(cards),'expected':len(cards),'source_sha256':sha,'backup':str(dest),'status':'complete'})
  for assignment in manifest['assignments'][reviewer]['items']:
   uid=assignment['review_id'];p=prior[uid];d=b['decisions'][uid];assert reviewer!=p['reviewer']
   second=a.correct(d['choice'],private[uid]['target_option'])
   records.append({**assignment,'second_reviewer':reviewer,'first_choice':p['human_choice'],'second_choice':d['choice'],'first_correct':p['human_correct'],'second_correct':second,'ai_correct':p['ai_correct'],'same_choice_category':p['human_choice']==d['choice'],'same_correctness_category':p['human_correct']==second,'second_agrees_ai_correctness':second==p['ai_correct'],'model':p['model'],'phase':p['phase'],'seed_id':p['seed_id']})
 records.sort(key=lambda r:r['original_item']);assert len(records)==32
 def summarize(rr):
  ids=[r['review_id'] for r in rr];weights={i:1 for i in ids}
  return {'n':len(rr),'same_choice_category':sum(r['same_choice_category'] for r in rr),'same_correctness_category':sum(r['same_correctness_category'] for r in rr),'second_agrees_ai_correctness':sum(r['second_agrees_ai_correctness'] for r in rr),'human_correctness_agreement':a.paired_stats(ids,{r['review_id']:r['first_correct'] for r in rr},{r['review_id']:r['second_correct'] for r in rr},weights)}
 result={'status':'both_complete','overall':summarize(records),'original_disagreements':summarize([r for r in records if r['selection']=='disagreement']),'agreement_controls':summarize([r for r in records if r['selection']=='sampled_agreement']),'records':records,'limitations':['Selected disagreement-enriched sample; raw agreement is not a population estimate.','Matching correctness categories do not necessarily mean identical endorsed answers.','Shared NONE/OTHER/MULTIPLE labels can describe different interpretations; inspect before reconciliation.','Original AI and human labels are preserved. No consensus is inferred where reviewers differ.']}
 save(OUT/'comparison_results.json',result)
 lines=['# Independent science cross-review comparison','','Original reviews and benchmark labels remain unchanged. This is a targeted audit, not a population reliability estimate.','','| Original item | First reviewer | First label | Second reviewer | Second label | AI correctness | Selection |','|---|---|---|---|---|---|---|']
 for r in records:lines.append('| '+' | '.join(str(x) for x in [r['original_item'],r['original_reviewer'],r['first_choice'],r['second_reviewer'],r['second_choice'],{True:'correct',False:'incorrect',None:'unresolved'}[r['ai_correct']],r['selection']])+' |')
 (OUT/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k not in ['records','limitations']},indent=2))
 print(json.dumps(records,indent=2))
if __name__=='__main__':main()
