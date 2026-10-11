"""Blinded cross-review: all 22 discrepancies plus ten sampled agreements."""
import hashlib,json,random,re,subprocess
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'model_runs/output_budget_v4/human_science_review_100'
OUT=SOURCE/'cross_review'
SEED=20261012
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
digest=lambda x:hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
 result=load(SOURCE/'human_agreement_results.json')
 cards={c['review_id']:c for c in load(SOURCE/'primary_blinded.json')}
 assert len(result['records'])==100 and len(result['disagreements'])==22
 rng=random.Random(SEED);assignments={};packets={}
 for reviewer,first in [('reviewer_a','reviewer_b'),('reviewer_b','reviewer_a')]:
  disputed=[r for r in result['records'] if r['reviewer']==first and not r['agreement']]
  pool=[r for r in result['records'] if r['reviewer']==first and r['agreement']]
  controls=rng.sample(sorted(pool,key=lambda r:r['item']),5)
  selected=disputed+controls;rng.shuffle(selected)
  selected_cards=[cards[r['review_id']] for r in selected]
  for c,r in zip(selected_cards,selected):
   assert c['card_sha256']==r['card_sha256']
   assert digest({k:v for k,v in c.items() if k!='card_sha256'})==c['card_sha256']
  role='cross_'+reviewer.lower()
  packet={'schema':'spine-science-cross-review-v4','packet_id':digest([role,selected_cards]),'role':role,'assigned_reviewer':reviewer,'cards':selected_cards}
  packets[reviewer]=packet
  assignments[reviewer]={'total':len(selected),'disagreements':len(disputed),'agreement_controls':5,'agreement_pool_n':len(pool),'control_inclusion_probability':5/len(pool),'items':[{'position':i+1,'review_id':r['review_id'],'original_item':r['item'],'original_reviewer':first,'selection':'disagreement' if not r['agreement'] else 'sampled_agreement'} for i,r in enumerate(selected)]}
 manifest={'sampling_seed':SEED,'source_packet_id':result['packet_id'],'source_analysis_sha256':hashlib.sha256((SOURCE/'human_agreement_results.json').read_bytes()).hexdigest(),'assignments':assignments,'packet_ids':{n:p['packet_id'] for n,p in packets.items()},'policy':'All 22 correctness-category disagreements plus five random agreements from each original reviewer. Each person reviews only the other person\'s items. Status hidden and order shuffled. No previous decisions imported.','interpretation':'Targeted audit; raw agreement on these 32 selected cases is not an overall population reliability estimate.'}
 if (OUT/'analyst_manifest.json').exists():assert load(OUT/'analyst_manifest.json')==manifest
 save(OUT/'analyst_manifest.json',manifest)
 template=(ROOT/'model_runs/science_human_review_template.html').read_text(encoding='utf-8')
 for reviewer,packet in packets.items():
  body=json.dumps(packet,ensure_ascii=False).replace('<','\\u003c')
  html=template.replace('__PACKET_JSON__',body).replace('<h1>Label one response per item</h1>',f'<h1>{reviewer} · Independent second review</h1><p>Review each response independently before discussing it with the other reviewer. Earlier decisions and item-selection reasons are hidden.</p>')
  extra='''<script>
const assigned=packet.assigned_reviewer;
$('reviewer').value=assigned;$('reviewer').readOnly=true;$('reviewer').dispatchEvent(new Event('change'));
$('role').textContent=`Independent second review · ${cards.length} responses · ${assigned}`;
const resumeOriginal=$('import').onchange;
$('import').onchange=async e=>{try{const f=e.target.files[0];if(!f)return;const b=JSON.parse(await f.text());if(b.reviewer!==assigned)throw Error('This questionnaire is assigned to '+assigned);return resumeOriginal(e)}catch(err){$('message').textContent='Cannot import: '+err.message}};
const downloadOriginal=$('export').onclick;
$('export').onclick=()=>{downloadOriginal();if(!state.reviewer)return;
let box=$('exportBackup');if(!box){box=document.createElement('details');box.id='exportBackup';box.innerHTML='<summary>JSON backup — copy if the download does not appear</summary><textarea readonly style="height:180px" aria-label="JSON backup"></textarea>';document.querySelector('header').appendChild(box);}
box.open=true;box.querySelector('textarea').value=JSON.stringify({schema:packet.schema,packet_id:packet.packet_id,role:packet.role,reviewer:state.reviewer,exported_at:new Date().toISOString(),sample_ids:cards.map(c=>c.review_id),decisions:state.decisions},null,2);
};
</script>'''
  html=html.replace('</body>',extra+'</body>')
  dest=OUT/(reviewer.lower()+'_review.html');dest.write_text(html,encoding='utf-8')
  save(OUT/(reviewer.lower()+'_blinded.json'),packet)
  scripts=re.findall(r'<script(?:\s[^>]*)?>(.*?)</script>',html,re.S)
  js=OUT/(reviewer.lower()+'_syntax_check.js');js.write_text('\n'.join(s for s in scripts if not s.lstrip().startswith('{')),encoding='utf-8')
  subprocess.run([str(Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'),'--check',str(js)],check=True)
  assert all(not ({'human_choice','ai_correct','final_correct','selection','target_option','model'} & set(c)) for c in packet['cards'])
 allids=[i['review_id'] for a in assignments.values() for i in a['items']]
 assert len(allids)==len(set(allids))==32
 assert {r['review_id'] for r in result['disagreements']} <= set(allids)
 (OUT/'index.html').write_text('''<!doctype html><html><head><meta charset="utf-8"><title>SPINE second review</title><style>body{font:18px/1.6 system-ui;max-width:680px;margin:60px auto;padding:20px}a{display:block;padding:18px;margin:16px 0;background:#e4f1eb;color:#164a3c;border-radius:10px}p{color:#394c55}</style></head><body><h1>SPINE · Independent second review</h1><p>Open only your assigned questionnaire. Label the response in the green box, save each decision, and export your JSON when finished. Do not discuss labels until both reviewers finish.</p><a href="reviewer_a_review.html">reviewer_a — '''+str(assignments['reviewer_a']['total'])+''' items</a><a href="reviewer_b_review.html">reviewer_b — '''+str(assignments['reviewer_b']['total'])+''' items</a></body></html>''',encoding='utf-8')
 print(json.dumps({n:{k:a[k] for k in ['total','disagreements','agreement_controls']} for n,a in assignments.items()}))

if __name__=='__main__':main()
