"""Audit exact published response text against immutable raw generation records."""
import json,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 root=ROOT/'model_runs';out=root/'output_budget_v4';out.mkdir(exist_ok=True)
 manifest=json.loads((root/'persona_revision_v3/run_manifest.json').read_text())
 index_path=root/'persona_revision_v3/final_response_index.jsonl'
 idx=[json.loads(x) for x in index_path.read_text(encoding='utf-8').splitlines()]
 lookup={}
 for name in manifest['models']:
  paths=[root/name/(p+'.jsonl') for p in ['base','attack','correction']]
  paths+=list((root/name).glob('attacks_*.jsonl'))+list((root/name).glob('corrections_*.jsonl'))
  paths+=list((root/'corrective_branch_v2'/name).glob('correction_*.jsonl'))
  paths+=list((root/'persona_revision_v3'/name/'raw').glob('*.json'))
  for path in paths:
   if not path.exists() or 'score' in path.name:continue
   rs=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()] if path.suffix=='.jsonl' else [json.loads(path.read_text(encoding='utf-8'))]
   for r in rs:
    if 'response' not in r or 'done_reason' not in r:continue
    key=(name,r.get('id'),(None if r.get('branch')=='base' else r.get('branch')),sha(r['response'].encode()))
    lookup[key]=(r,str(path.relative_to(ROOT)))
 audit=[];requests=[]
 for i in idx:
  k=(i['run_name'],i['id'],i['branch'],sha(i['judge_input']['final_response'].encode()))
  raw,path=lookup.get(k,({},None));reason=raw.get('done_reason')
  row={x:i[x] for x in ['blind_id','run_name','id','phase','domain','branch','response_sha256']}
  row.update(source=path,done_reason=reason,status='limit' if str(reason).lower() in ['length','max_tokens'] else 'unknown' if reason is None else 'other')
  audit.append(row)
  if row['status']=='limit':
   config=manifest['models'][i['run_name']]['config']
   requests.append({**row,'messages':raw['messages'],'config':config,'original_cap':config['coding_cap'] if i['domain']=='coding' else config['math_logic_cap']})
 summary={'index_sha256':sha(index_path.read_bytes()),'total':len(idx),'status':dict(Counter(r['status'] for r in audit)),'limited_by_model_phase':dict(Counter(r['run_name']+'/'+r['phase'] for r in requests))}
 (out/'audit.json').write_text(json.dumps({'summary':summary,'records':audit},indent=2),encoding='utf-8')
 (out/'limited_requests.json').write_text(json.dumps(requests,indent=2),encoding='utf-8')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
