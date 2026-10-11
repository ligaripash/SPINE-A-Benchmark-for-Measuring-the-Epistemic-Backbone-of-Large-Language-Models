"""Wait for immutable scoring checkpoints, then verify and compute metrics once."""
import json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'model_runs/output_budget_v4'
def save(obj):
 (OUT/'metrics_completion_progress.json').write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':
 lock=OUT/'metrics_completion.lock'
 fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 try:
  packets=sum([json.loads((OUT/(n+'_followup_packet.json')).read_text(encoding='utf-8')) for n in ['local','openai','gemini']],[])
  expected=[OUT/'followup_scores'/r['run_name']/(r['id']+'__'+r['branch']+'.json') for r in packets if (r['run_name'],r['id'],r['branch'])!=('deepseek_r1_1_5b','math_158','bureaucrat')]
  while True:
   remaining=sum(not p.exists() for p in expected)
   active=any((OUT/(n+'_scoring.lock')).exists() for n in ['objective','science'])
   save({'status':'waiting_for_scoring','remaining':remaining,'expected':len(expected),'scoring_lock_present':active,'pid':os.getpid()})
   if not remaining and not active:break
   if not active:
    save({'status':'scoring_stopped_incomplete','remaining':remaining});raise SystemExit(2)
   time.sleep(30)
  save({'status':'computing_metrics','pid':os.getpid()})
  result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'model_runs/finalize_output_budget_v4.py'),'--lane','analyze'],cwd=ROOT)
  save({'status':'complete' if result.returncode==0 else 'analysis_failed','returncode':result.returncode})
 finally:lock.unlink(missing_ok=True)
