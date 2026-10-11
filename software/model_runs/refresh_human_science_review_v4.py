"""Refresh the existing audit sample against final outputs; preserve all old work."""
import json,hashlib,random,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'model_runs/output_budget_v4';OLD=ROOT/'model_runs/persona_revision_v3/human_science_review';OUT=SRC/'human_science_review'
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda x:hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
v=load(SRC/'final_verification.json')
for f,h in v['hashes'].items():assert hashlib.sha256((SRC/f).read_bytes()).hexdigest()==h
idx=[json.loads(x) for x in (SRC/'final_response_index.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()];idx=[r for r in idx if r['domain']=='logic']
lookup={(r['run_name'],r['id'],r['branch']):r for r in idx};labs=load(SRC/'final_correctness_labels.json');oldprivate=load(OLD/'analyst_private.json');oldcards=load(OLD/'primary_blinded.json');oldmanifest=load(OLD/'manifest.json')
rng=random.Random(20261011);chosen=[];used=set();mapping=[];private={}
for c in oldcards:
 old=oldprivate[c['review_id']];k=(old['model'],old['seed_id'],old['persona']);r=lookup.get(k)
 if r is None:
  pool=[r for r in idx if r['run_name']==old['model'] and r['phase']==old['phase'] and r['blind_id'] not in used and (r['run_name'],r['id'],r['branch']) not in {(x['model'],x['seed_id'],x['persona']) for x in oldprivate.values()}]
  r=rng.choice(sorted(pool,key=lambda r:r['blind_id']))
 f=r['judge_input'];uid=r['blind_id'];assert uid not in used;used.add(uid)
 card={'review_id':uid,'phase':r['phase'],'task':f['task'],'original_task':f.get('original_task',f['task']),'initial_answer':f.get('initial_answer',''),'followup_message':f.get('followup_message',''),'response':f['final_response'],'option_labels':list(dict.fromkeys(re.findall(r'\(([A-E1-5])\)',f['task'].split('Options:',1)[-1])))}
 assert len(card['option_labels'])>=2
 card['card_sha256']=sha(card);chosen.append(card)
 identical=all(card[k]==c[k] for k in card if k not in ['review_id','card_sha256'])
 mapping.append({'old_review_id':c['review_id'],'new_review_id':uid,'identical_content':identical,'same_response_slot':k in lookup})
 private[uid]={'review_id':uid,'seed_id':r['id'],'model':r['run_name'],'phase':r['phase'],'persona':r['branch'],'target_option':r['target_option'],'final_correct':labs[uid],'input_sha256':r['input_sha256'],'card_sha256':card['card_sha256']}
secondids={next(x['new_review_id'] for x in mapping if x['old_review_id']==uid) for uid in oldmanifest['secondary_ids']}
second=[c for c in chosen if c['review_id'] in secondids];rng.shuffle(second)
packetid=sha(chosen);schema='spine-science-human-audit-v4'
if (OUT/'manifest.json').exists():assert load(OUT/'manifest.json')['packet_id']==packetid
save(OUT/'primary_blinded.json',chosen);save(OUT/'secondary_blinded.json',second);save(OUT/'analyst_private.json',private)
save(OUT/'manifest.json',{'schema':schema,'packet_id':packetid,'primary_n':len(chosen),'secondary_n':len(second),'population_n':len(idx),'source_hashes':v['hashes'],'mapping':mapping,'sampling':'Continuation of the v3 audit sample: same model/task/branch where eligible, replaced within model and phase otherwise. Descriptive audit; old design weights do not apply to this refreshed sample.','primary_ids':[c['review_id'] for c in chosen],'secondary_ids':[c['review_id'] for c in second]})
template=(ROOT/'model_runs/science_human_review_template.html').read_text(encoding='utf-8')
template=template.replace('<h1>Label one response per item</h1>','<h1>Review the final model outputs</h1><p>Updated after output-budget recovery. The earlier questionnaire and its saved work remain preserved.</p>')
# A visible JSON fallback works even when file downloads are blocked by the browser.
extra='''<script>
const oldExportHandler=document.getElementById('export').onclick;
document.getElementById('export').onclick=()=>{
oldExportHandler();if(!state.reviewer)return;
let box=document.getElementById('exportBackup');if(!box){box=document.createElement('details');box.id='exportBackup';box.innerHTML='<summary>JSON backup — copy this if the download does not appear</summary><textarea readonly style="height:180px" aria-label="JSON backup"></textarea>';document.querySelector('header').appendChild(box);}
box.open=true;box.querySelector('textarea').value=JSON.stringify({schema:packet.schema,packet_id:packet.packet_id,role:packet.role,reviewer:state.reviewer,exported_at:new Date().toISOString(),sample_ids:cards.map(c=>c.review_id),decisions:state.decisions},null,2);
};
</script>'''
for role,cards in [('primary',chosen),('secondary',second)]:
 body=json.dumps({'schema':schema,'packet_id':packetid,'role':role,'cards':cards},ensure_ascii=False).replace('<','\\u003c')
 html=template.replace('__PACKET_JSON__',body).replace('</body>',extra+'</body>');(OUT/(role+'_review.html')).write_text(html,encoding='utf-8')
migrated=[]
for p in (OLD/'received_exports').glob('*.json'):
 b=load(p);cards=chosen if b['role']=='primary' else second;newcards={c['review_id']:c for c in cards};decisions={}
 for mid in mapping:
  d=b.get('decisions',{}).get(mid['old_review_id']);uid=mid['new_review_id']
  oc=next(c for c in oldcards if c['review_id']==mid['old_review_id'])
  if d and mid['identical_content'] and uid in newcards and d.get('card_sha256')==oc['card_sha256']:
   decisions[uid]={**d,'card_sha256':newcards[uid]['card_sha256']}
 revised={**b,'schema':schema,'packet_id':packetid,'sample_ids':list(newcards),'decisions':decisions,'migration_source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 dest=OUT/'migrated_exports'/p.name;save(dest,revised);migrated.append({'reviewer':b['reviewer'],'retained':len(decisions),'previous':len(b.get('decisions',{})),'file':str(dest)})
save(OUT/'refresh_report.json',{'identical_items':sum(x['identical_content'] for x in mapping),'updated_items':sum(not x['identical_content'] for x in mapping),'export_migrations':migrated})
print(json.dumps(load(OUT/'refresh_report.json'),indent=2))
