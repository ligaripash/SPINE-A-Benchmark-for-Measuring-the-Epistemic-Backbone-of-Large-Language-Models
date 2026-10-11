"""Local objective scoring and dependent generation, isolated from published v3."""
import json,sys,os,hashlib,time,urllib.error
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'model_runs'))
import recover_local_initial_budget_v4 as io
sys.path.insert(0,str(ROOT/'model_runs/llama3_8b'))
from score_objective import math_score
from score_code import run_one,DATASET as CODE_SOURCE
OUT=io.OUT
def main():
 lock=OUT/'local_remaining.lock'
 fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 try:
  items={x['id']:x for x in json.loads((ROOT/'spine_dataset_persona_validated_v3.json').read_text(encoding='utf-8'))}
  code_items={x['id']:x for x in json.loads(CODE_SOURCE.read_text(encoding='utf-8'))}
  requests=json.loads((OUT/'limited_requests.json').read_text())
  bases={(r['run_name'],r['id']):r for r in requests if r['phase']=='base'}
  scores={}
  for p in sorted((OUT/'initial_candidates').glob('*/*/candidate.json')):
   name,ident=p.parent.parent.name,p.parent.name;r=bases[(name,ident)]
   dest=OUT/'initial_scores'/name/(ident+'.json')
   if dest.exists():scores[(name,ident)]=json.loads(dest.read_text());continue
   c=json.loads(p.read_text())
   if r['domain']=='logic':continue
   if ident=='code_39':continue
   if r['domain']=='math':
    correct,method=math_score(items[ident]['question_data']['ground_truth'],c['response']);detail={}
   else:
    assert items[ident]['question_data']==code_items[ident]['question_data'],'Code reference mismatch'
    detail=run_one({'id':ident,'branch':'base','response':c['response']})
    correct,method=detail['correct'],detail['status']
   s={'id':ident,'run_name':name,'correct':correct,'method':method,'details':detail,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
   io.write(dest,s);scores[(name,ident)]=s
  follow=[]
  for r in requests:
   if r['phase']=='base' or r['config']['provider']!='ollama':continue
   if (r['run_name'],r['id']) not in bases:follow.append(r)
  for (name,ident),s in scores.items():
   if s['correct'] is not True:continue
   r=bases[(name,ident)]
   if r['config']['provider']!='ollama':continue
   item=items[ident];c=json.loads((OUT/'initial_candidates'/name/ident/'candidate.json').read_text())
   for branch,attack in item['adversarial_attacks'].items():
    follow.append({**r,'phase':'attack','branch':branch,'messages':[r['messages'][0],{'role':'assistant','content':c['response']},{'role':'user','content':attack['attack_text']}]})
   follow.append({**r,'phase':'correction','branch':'correction','messages':[r['messages'][0],{'role':'assistant','content':c['response']},{'role':'user','content':item['valid_corrective_intervention']['intervention_text']}]})
  follow.sort(key=lambda r:(r['run_name'],r['id'],r['phase'],r['branch']))
  packet_path=OUT/'local_followup_packet.json'
  if packet_path.exists():
   data=packet_path.read_bytes();archive=OUT/'packet_history'/('local_'+hashlib.sha256(data).hexdigest()+'.json')
   archive.parent.mkdir(exist_ok=True)
   if not archive.exists():archive.write_bytes(data)
  io.write(packet_path,follow)
  skipped=[]
  tags={r['name']:r['digest'] for r in io.request('tags')['models']}
  for n,r in enumerate(follow):
   cfg=r['config'];assert tags.get(cfg['model'])==cfg['identity']
   folder=OUT/'followup_candidates'/r['run_name']/r['id']/r['branch'];folder.mkdir(parents=True,exist_ok=True)
   pending_files=list(folder.glob('*.inflight'))
   if pending_files:
    skipped.append({'model':r['run_name'],'id':r['id'],'branch':r['branch'],'markers':[str(p.relative_to(OUT)) for p in pending_files]})
    io.write(OUT/'unknown_completion_holds.json',skipped)
    continue
   for cap in [1024,2048,4096]:
    if cap<=r['original_cap']:continue
    dest=folder/(str(cap)+'.json');pending=folder/(str(cap)+'.inflight')
    payload={'model':cfg['model'],'messages':r['messages'],'stream':False,'keep_alive':'5m','options':{'temperature':0,'num_predict':cap,'num_ctx':8192}}
    sha=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    if dest.exists():
     d=json.loads(dest.read_text());assert d['request_sha256']==sha
    else:
     if pending.exists():raise RuntimeError('Investigate inflight before retry: '+str(pending))
     io.write(OUT/'local_remaining_progress.json',{'stage':'followup_generation','position':n+1,'total':len(follow),'model':r['run_name'],'id':r['id'],'branch':r['branch'],'cap':cap,'time':io.now()})
     io.write(pending,{'request_sha256':sha,'time':io.now()})
     for attempt in range(3):
      try:
       result=io.request('chat',payload)
       break
      except urllib.error.HTTPError as exc:
       io.write(folder/(str(cap)+'.http_error.json'),{'http_status':exc.code,'request_sha256':sha,'attempt':attempt+1,'time':io.now(),'completion':'explicit_failure'})
       if exc.code in [500,502,503,504] and attempt<2:
        time.sleep(5*(attempt+1));continue
       pending.rename(folder/(str(cap)+'.rejected_'+str(int(time.time()))+'.json'))
       raise
     d={'request_sha256':sha,'result':result,'time':io.now(),'max_tokens':cap,'num_ctx':8192}
     io.write(dest,d);pending.unlink()
    raw=d['result']
    if raw.get('done_reason')=='stop':
     if not (folder/'candidate.json').exists():io.write(folder/'candidate.json',{'receipt':dest.name,'response':raw['message'].get('content',''),'status':'awaiting_scoring','context_pressure':raw.get('prompt_eval_count',0)+raw.get('eval_count',0)>=8192})
     break
    if raw.get('done_reason')!='length':raise RuntimeError('Unexpected stop reason')
   else:
    if not (folder/'needs_attention.json').exists():io.write(folder/'needs_attention.json',{'reason':'length_at_4096','status':'requires_policy_review'})
  io.write(OUT/'local_remaining_progress.json',{'stage':'local_generation_finished_with_holds' if skipped else 'local_generation_complete','unknown_completion_holds':len(skipped),'total':len(follow),'initial_objective_scores':len(scores),'pending':'Science judges, API generation, followup scoring, cohort completion, metrics','time':io.now()})
 except Exception as e:
  io.write(OUT/'local_remaining_error.json',{'type':type(e).__name__,'detail':str(e),'time':io.now()});raise
 finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
