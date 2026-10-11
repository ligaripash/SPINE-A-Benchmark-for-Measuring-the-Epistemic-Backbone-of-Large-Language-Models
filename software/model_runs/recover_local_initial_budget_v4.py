"""Resume-safe local initial-answer budget recovery. Original results are untouched.
Stops on unknown request completion; never silently retries an inflight request.
This stage generates candidates only. Scoring and dependent followups must follow.
"""
import json,hashlib,os,urllib.request
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'model_runs/output_budget_v4'
def now():return datetime.now(timezone.utc).isoformat()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2),encoding='utf-8');os.replace(t,p)
def request(path,payload=None):
 data=None if payload is None else json.dumps(payload).encode()
 with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:11434/api/'+path,data=data,headers={'Content-Type':'application/json'}),timeout=1800) as r:return json.load(r)
def main():
 lock=OUT/'local_initial_runner.lock'
 fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 try:
  rows=json.loads((OUT/'limited_requests.json').read_text())
  rows=[r for r in rows if r['phase']=='base' and r['config']['provider']=='ollama']
  tags={r['name']:r['digest'] for r in request('tags')['models']}
  for n,r in enumerate(rows):
   cfg=r['config'];assert tags.get(cfg['model'])==cfg['identity'],'Model identity mismatch'
   folder=OUT/'initial_candidates'/r['run_name']/r['id'];folder.mkdir(parents=True,exist_ok=True)
   caps=[c for c in [1024,2048,4096] if c>r['original_cap']]
   for cap in caps:
    receipt=folder/(str(cap)+'.json');pending=folder/(str(cap)+'.inflight')
    payload={'model':cfg['model'],'messages':r['messages'],'stream':False,'keep_alive':'5m','options':{'temperature':0,'num_predict':cap,'num_ctx':8192}}
    fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    if receipt.exists():
     result=json.loads(receipt.read_text());assert result['request_sha256']==fingerprint
    else:
     if pending.exists():raise RuntimeError('Unknown completion requires investigation: '+str(pending))
     write(OUT/'local_initial_progress.json',{'time':now(),'item':n+1,'total':len(rows),'model':r['run_name'],'id':r['id'],'cap':cap,'status':'running'})
     write(pending,{'request_sha256':fingerprint,'started':now()})
     result={'request_sha256':fingerprint,'source_response_sha256':r['response_sha256'],'max_tokens':cap,'num_ctx':8192,'result':request('chat',payload),'time':now()}
     write(receipt,result);pending.unlink()
    raw=result['result']
    if raw.get('done_reason')=='stop':
     write(folder/'candidate.json',{'receipt':receipt.name,'response':raw['message'].get('content',''),'status':'awaiting_scoring','time':now()});break
    if raw.get('done_reason')!='length':raise RuntimeError('Unexpected finish reason; saved receipt requires review')
   else:
    write(folder/'needs_attention.json',{'reason':'still_length_limited_at_4096','time':now()})
  write(OUT/'local_initial_progress.json',{'status':'generation_stage_complete','total':len(rows),'time':now(),'next':'Score candidates, rebuild dependent followups; inspect needs_attention files'})
 except Exception as e:
  write(OUT/'local_initial_error.json',{'type':type(e).__name__,'detail':str(e),'time':now()});raise
 finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
