"""Extend only DeepSeek's 212 capped followups; preserve earlier receipts."""
import json,os,hashlib,time,urllib.error
import recover_local_initial_budget_v4 as io
OUT=io.OUT
def main():
 root=OUT/'deepseek_extended';root.mkdir(exist_ok=True)
 lock=root/'runner.lock'
 fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 try:
  packet=json.loads((OUT/'local_followup_packet.json').read_text())
  selected=[r for r in packet if r['run_name']=='deepseek_r1_1_5b' and (OUT/'followup_candidates'/r['run_name']/r['id']/r['branch']/'needs_attention.json').exists()]
  assert len(selected)==212
  manifest={'requests':selected,'output_caps':[8192,16384],'num_ctx':32768,'scope':'212 capped DeepSeek followups only; original 94 initial non-answer policy unchanged'}
  if (root/'manifest.json').exists():assert json.loads((root/'manifest.json').read_text())==manifest
  else:io.write(root/'manifest.json',manifest)
  tags={m['name']:m['digest'] for m in io.request('tags')['models']}
  for n,r in enumerate(selected):
   assert tags[r['config']['model']]==r['config']['identity']
   folder=root/r['id']/r['branch'];folder.mkdir(parents=True,exist_ok=True)
   for cap in [8192,16384]:
    path=folder/(str(cap)+'.json');pending=folder/(str(cap)+'.inflight')
    payload={'model':r['config']['model'],'messages':r['messages'],'stream':False,'keep_alive':'5m','options':{'temperature':0,'num_predict':cap,'num_ctx':32768}}
    sha=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    if path.exists():
     d=json.loads(path.read_text());assert d['request_sha256']==sha
    else:
     if pending.exists():raise RuntimeError('Unknown completion: '+str(pending))
     io.write(root/'progress.json',{'status':'running','position':n+1,'total':len(selected),'id':r['id'],'branch':r['branch'],'cap':cap,'time':io.now()})
     io.write(pending,{'request_sha256':sha,'time':io.now()})
     for attempt in range(3):
      try:result=io.request('chat',payload);break
      except urllib.error.HTTPError as e:
       io.write(folder/(str(cap)+'.http_error.json'),{'status':e.code,'attempt':attempt+1,'time':io.now()})
       if e.code in [500,502,503,504] and attempt<2:time.sleep(5*(attempt+1));continue
       pending.rename(folder/(str(cap)+'.rejected_'+str(int(time.time()))+'.json'));raise
     d={'request_sha256':sha,'result':result,'time':io.now(),'max_tokens':cap,'num_ctx':32768}
     io.write(path,d);pending.unlink()
    result=d['result']
    if result.get('done_reason')=='stop':
     if not (folder/'candidate.json').exists():io.write(folder/'candidate.json',{'receipt':path.name,'response':result['message'].get('content',''),'status':'awaiting_scoring','time':io.now()})
     break
    if result.get('done_reason')!='length':raise RuntimeError('Unexpected finish reason')
   else:
    if not (folder/'needs_attention.json').exists():io.write(folder/'needs_attention.json',{'reason':'length_at_16384','time':io.now()})
  io.write(root/'progress.json',{'status':'generation_complete','total':len(selected),'time':io.now(),'next':'Score successful replacements and inspect persistent caps'})
 except Exception as e:
  io.write(root/'error.json',{'type':type(e).__name__,'detail':str(e),'time':io.now()});raise
 finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
