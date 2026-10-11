"""API initial recovery and blind science extraction. Credentials remain in memory."""
import getpass,json,os,hashlib
from pathlib import Path
import reevaluate_persona_v3 as api
import recover_local_initial_budget_v4 as io
OUT=io.OUT
def main():
 api.KEYS['openai']=getpass.getpass('OpenAI API key (hidden): ')
 api.KEYS['gemini']=getpass.getpass('Gemini API key (hidden): ')
 api.OUT=OUT
 lock=OUT/'api_initial_scoring.lock'
 fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 try:
  rows=[r for r in json.loads((OUT/'limited_requests.json').read_text()) if r['phase']=='base' and r['config']['provider']=='openai']
  for n,r in enumerate(rows):
   folder=OUT/'initial_candidates'/r['run_name']/r['id']
   for cap in [1024,2048,4096]:
    payload={'model':r['config']['model'],'messages':r['messages'],'temperature':0,'max_tokens':cap}
    io.write(OUT/'api_initial_scoring_progress.json',{'stage':'gpt_initial_generation','position':n+1,'total':len(rows),'id':r['id'],'cap':cap,'time':io.now()})
    receipt=folder/(str(cap)+'_api.json')
    result=api.receipt(receipt,'openai','https://api.openai.com/v1/chat/completions',payload)
    c=result['choices'][0];reason=c.get('finish_reason')
    if reason=='stop':
     io.write(folder/'candidate.json',{'receipt':receipt.name,'response':c['message'].get('content') or '','status':'awaiting_scoring','time':io.now()});break
    if reason!='length':raise RuntimeError('Non-token stop saved; inspect receipt')
   else:io.write(folder/'needs_attention.json',{'reason':'length_at_4096','status':'requires_policy_review'})
  old={(x['run_name'],x['id']):x for x in api.rows(io.ROOT/'model_runs/persona_revision_v3/final_response_index.jsonl') if x['phase']=='base'}
  for p in sorted((OUT/'initial_candidates').glob('*/*/candidate.json')):
   name,ident=p.parent.parent.name,p.parent.name
   if old[(name,ident)]['domain']!='logic':continue
   dest=OUT/'initial_scores'/name/(ident+'.json')
   if dest.exists():continue
   c=json.loads(p.read_text());source=old[(name,ident)]
   fields={**source['judge_input'],'final_response':c['response']}
   uid=api.digest({'source':source['blind_id'],'fields':fields})[:24]
   record={**source,'blind_id':uid,'judge_input':fields,'input_sha256':api.judges.sha(fields),'response_sha256':api.digest(c['response'].encode()),'primary_correct':None}
   io.write(OUT/'api_initial_scoring_progress.json',{'stage':'science_initial_scoring','model':name,'id':ident,'time':io.now()})
   luna=api.judge_answer(record,'luna');flash=api.judge_answer(record,'flash_lite')
   if luna['raw_choice']==flash['raw_choice'] and luna['choice']==flash['choice'] and luna['choice']!='UNCLEAR':
    chosen=luna;method='Luna/Flash-Lite agreement'
   else:
    chosen=api.judge_answer(record,'sol');method='Sol medium adjudication'
    if chosen['choice'] in ['NONE','UNCLEAR']:chosen=api.judge_answer(record,'sol_clarification');method='Sol explicit-OTHER clarification'
   correct=api.analysis.correctness(chosen['choice'],record['target_option'])
   io.write(dest,{'id':ident,'run_name':name,'correct':correct,'method':method,'choice':chosen['choice'],'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'index_record':record})
  io.write(OUT/'api_initial_scoring_progress.json',{'stage':'initial_api_and_science_stage_complete','next':'Refresh objective scoring and dependent followup packets; generate API followups; score all revised followups','time':io.now()})
 except api.ApiFailure as e:
  io.write(OUT/'api_initial_scoring_error.json',{'error':str(e),'time':io.now()})
  print('API request stopped; sanitized diagnostics saved.')
 except Exception as e:
  io.write(OUT/'api_initial_scoring_error.json',{'type':type(e).__name__,'time':io.now()})
  print('Runner stopped; receipts preserved.')
 finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
