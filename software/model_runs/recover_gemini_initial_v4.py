"""Gemini initial recovery; key stays in process memory, receipts are resumable."""
import getpass,json,os
from pathlib import Path
import reevaluate_persona_v3 as api
import recover_local_initial_budget_v4 as io
OUT=io.OUT
def main():
 api.KEYS['gemini']=getpass.getpass('Gemini API key (hidden): ')
 api.OUT=OUT
 lock=OUT/'gemini_initial.lock'
 fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 try:
  rows=[r for r in json.loads((OUT/'limited_requests.json').read_text()) if r['phase']=='base' and r['config']['provider']=='gemini']
  for n,r in enumerate(rows):
   folder=OUT/'initial_candidates'/r['run_name']/r['id']
   for cap in [4096,8192]:
    payload={'contents':[{'role':'user','parts':[{'text':r['messages'][0]['content']}]}],'generationConfig':{'maxOutputTokens':cap,'temperature':1.0,'thinkingConfig':{'thinkingLevel':r['config']['thinking_level']}}}
    io.write(OUT/'gemini_initial_progress.json',{'status':'running','position':n+1,'total':len(rows),'id':r['id'],'cap':cap,'time':io.now()})
    receipt=folder/(str(cap)+'_api.json')
    result=api.receipt(receipt,'gemini','https://generativelanguage.googleapis.com/v1beta/models/'+r['config']['model']+':generateContent',payload)
    candidates=result.get('candidates') or []
    if not candidates:raise RuntimeError('Saved response has no candidate; inspect receipt')
    c=candidates[0];reason=c.get('finishReason')
    if reason=='STOP':
     content=c.get('content',{})
     io.write(folder/'candidate.json',{'receipt':receipt.name,'response':''.join(p.get('text','') for p in content.get('parts',[]) if not p.get('thought')),'gemini_model_content':content,'status':'awaiting_scoring','time':io.now()})
     break
    if reason!='MAX_TOKENS':raise RuntimeError('Non-token stop saved; inspect receipt')
   else:io.write(folder/'needs_attention.json',{'reason':'length_at_8192','status':'requires_policy_review'})
  io.write(OUT/'gemini_initial_progress.json',{'status':'generation_complete','total':len(rows),'next':'Science scoring and dependent followups','time':io.now()})
 except api.ApiFailure as e:
  io.write(OUT/'gemini_initial_error.json',{'error':str(e),'time':io.now()})
  print('Gemini request stopped; sanitized diagnostics saved.')
 except Exception as e:
  io.write(OUT/'gemini_initial_error.json',{'error_type':type(e).__name__,'time':io.now()})
  print('Runner stopped; receipts preserved.')
 finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
