"""Resume-safe API followups; completed initial stages are never repeated."""
import json,os,getpass
from concurrent.futures import ThreadPoolExecutor
import reevaluate_persona_v3 as api
import recover_local_initial_budget_v4 as io
OUT=io.OUT
def lane(provider,packet,original):
 name='api_followups_'+provider;lock=OUT/(name+'.lock')
 fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 try:
  for n,r in enumerate(packet):
   folder=OUT/'followup_candidates'/r['run_name']/r['id']/r['branch']
   caps=[1024,2048,4096] if provider=='openai' else [4096,8192]
   for cap in caps:
    if cap<=r['original_cap']:continue
    if provider=='openai':
     url='https://api.openai.com/v1/chat/completions'
     payload={'model':r['config']['model'],'messages':r['messages'],'temperature':0,'max_tokens':cap}
    else:
     base=OUT/'initial_candidates'/r['run_name']/r['id']/'candidate.json'
     content=json.loads(base.read_text())['gemini_model_content'] if base.exists() else original[r['id']]['gemini_model_content']
     assert ''.join(p.get('text','') for p in content.get('parts',[]) if not p.get('thought'))==r['messages'][1]['content']
     url='https://generativelanguage.googleapis.com/v1beta/models/'+r['config']['model']+':generateContent'
     payload={'contents':[{'role':'user','parts':[{'text':r['messages'][0]['content']}]},content,{'role':'user','parts':[{'text':r['messages'][2]['content']}]}],'generationConfig':{'temperature':1.0,'maxOutputTokens':cap,'thinkingConfig':{'thinkingLevel':r['config']['thinking_level']}}}
    io.write(OUT/(name+'_progress.json'),{'status':'running','position':n+1,'total':len(packet),'id':r['id'],'branch':r['branch'],'cap':cap,'time':io.now()})
    receipt=folder/(str(cap)+'_api.json')
    raw=api.receipt(receipt,provider,url,payload)
    if provider=='openai':
     c=raw['choices'][0];reason=c.get('finish_reason');text=c['message'].get('content') or ''
    else:
     c=raw.get('candidates',[{}])[0];reason=c.get('finishReason');text=''.join(p.get('text','') for p in c.get('content',{}).get('parts',[]) if not p.get('thought'))
    if reason in ['stop','STOP']:
     if not (folder/'candidate.json').exists():io.write(folder/'candidate.json',{'receipt':receipt.name,'response':text,'status':'awaiting_scoring','time':io.now()})
     break
    if reason not in ['length','MAX_TOKENS']:raise RuntimeError('Unexpected finish reason; inspect receipt')
   else:
    if not (folder/'needs_attention.json').exists():io.write(folder/'needs_attention.json',{'reason':'length_at_max_budget','status':'requires_policy_review'})
  io.write(OUT/(name+'_progress.json'),{'status':'generation_complete','total':len(packet),'time':io.now()})
 except api.ApiFailure as e:io.write(OUT/(name+'_error.json'),{'error':e.info,'time':io.now()})
 except Exception as e:io.write(OUT/(name+'_error.json'),{'type':type(e).__name__,'detail':str(e),'time':io.now()})
 finally:lock.unlink(missing_ok=True)
def main():
 api.KEYS['openai']=getpass.getpass('OpenAI key (hidden): ')
 api.KEYS['gemini']=getpass.getpass('Gemini key (hidden): ')
 api.OUT=OUT
 rows=json.loads((OUT/'limited_requests.json').read_text())
 bases={(r['run_name'],r['id']):r for r in rows if r['phase']=='base'}
 items={x['id']:x for x in json.loads((io.ROOT/'spine_dataset_persona_validated_v3.json').read_text(encoding='utf-8'))}
 original={x['id']:x for x in api.rows(io.ROOT/'model_runs/gemini_3_flash/base.jsonl')}
 packets={}
 for provider in ['openai','gemini']:
  follow=[r for r in rows if r['phase']!='base' and r['config']['provider']==provider and (r['run_name'],r['id']) not in bases]
  for k,r in bases.items():
   if r['config']['provider']!=provider or r['id']=='code_39':continue
   score=OUT/'initial_scores'/k[0]/(k[1]+'.json')
   if not score.exists():
    if (OUT/'initial_candidates'/k[0]/k[1]/'candidate.json').exists():raise RuntimeError('Initial candidate still needs scoring: '+str(k))
    continue
   if json.loads(score.read_text())['correct'] is not True:continue
   candidate=json.loads((OUT/'initial_candidates'/k[0]/k[1]/'candidate.json').read_text())
   item=items[k[1]]
   branches=[(b,a['attack_text'],'attack') for b,a in item['adversarial_attacks'].items()]
   branches.append(('correction',item['valid_corrective_intervention']['intervention_text'],'correction'))
   for branch,text,phase in branches:
    follow.append({**r,'phase':phase,'branch':branch,'messages':[r['messages'][0],{'role':'assistant','content':candidate['response']},{'role':'user','content':text}]})
  follow.sort(key=lambda r:(r['id'],r['branch']))
  path=OUT/(provider+'_followup_packet.json')
  if path.exists():assert json.loads(path.read_text())==follow,'Frozen packet changed'
  else:io.write(path,follow)
  packets[provider]=follow
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(lane,p,packets[p],original) for p in packets]
  for f in futures:f.result()
if __name__=='__main__':main()
