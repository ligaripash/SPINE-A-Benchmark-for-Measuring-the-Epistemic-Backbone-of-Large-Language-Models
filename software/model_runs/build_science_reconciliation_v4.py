"""Build an explicit joint reconciliation form; never overwrite independent labels."""
import json,hashlib,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'model_runs/output_budget_v4/human_science_review_100'
CROSS=BASE/'cross_review';OUT=BASE/'reconciliation'
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda x:hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def main():
 comparison=load(CROSS/'comparison_results.json');selected=[r for r in comparison['records'] if not r['same_choice_category']];assert len(selected)==11
 original=load(BASE/'human_agreement_results.json')
 export=load(BASE/'received_exports'/('primary_'+original['source_export_sha256']+'.json'))
 cards={r['review_id']:r for r in load(BASE/'primary_blinded.json')}
 second={}
 for person in ['reviewer_a','reviewer_b']:
  completion=load(CROSS/(person+'_completion.json'));b=load(CROSS/'received_exports'/(person+'_'+completion['source_sha256']+'.json'));second.update(b['decisions'])
 new=[]
 for r in selected:
  uid=r['review_id'];c={**cards[uid]};c.pop('card_sha256')
  c['original_item']=r['original_item']
  c['human_labels']=[{'reviewer':r['original_reviewer'],'choice':r['first_choice'],'answer':export['decisions'][uid].get('answer',''),'reason':export['decisions'][uid].get('reason','')},{'reviewer':r['second_reviewer'],'choice':r['second_choice'],'answer':second[uid].get('answer',''),'reason':second[uid].get('reason','')}]
  c['card_sha256']=sha(c);new.append(c)
 packet={'schema':'spine-science-reconciliation-v4','packet_id':sha(new),'role':'reconciliation','cards':new}
 OUT.mkdir(exist_ok=True)
 if (OUT/'packet.json').exists():assert load(OUT/'packet.json')==packet
 (OUT/'packet.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 template=(ROOT/'model_runs/science_human_review_template.html').read_text(encoding='utf-8')
 template=template.replace('SPINE · Independent human audit','SPINE · Joint reconciliation').replace('<h1>Label one response per item</h1>','<h1>Reconcile 11 differing human labels</h1><p>reviewer_a and reviewer_b: discuss each response and agree on one final label. Your earlier independent labels remain unchanged. If agreement is not reached, leave the item unsaved.</p>')
 template=template.replace('Review independently. Do not consult AI tools, another reviewer, AI labels, or the analyst files. The second reviewer should not see the first reviewer\'s export.','Discuss the two human interpretations together and check the exact response. Do not use AI labels or the task reference to decide which answer the response endorses.')
 template=template.replace('Reviewer ID</label>','Reviewers participating in this reconciliation</label>').replace('placeholder="Your name or reviewer code"','placeholder="For example: reviewer_a and reviewer_b"')
 template=template.replace('<section class="card"><h2>Your label for the green response</h2>','<section class="card"><h2>Independent human labels</h2><p id="originalItem"></p><div id="humanLabels"></div></section><section class="card"><h2>Agreed final label for the green response</h2>')
 template=template.replace('Brief explanation (optional)','Reason for the reconciled label (optional)').replace('Add a note if useful','Briefly note what resolved the disagreement')
 template=template.replace('<div><button id="save"','<label style="font-weight:400"><input id="jointAgreement" type="checkbox" style="width:auto"> Both reviewers agree on this final label.</label><div><button id="save"')
 template=template.replace("function show(){const c=cards[pos],d=state.decisions[c.review_id]||{};","""function show(){const c=cards[pos],d=state.decisions[c.review_id]||{};
 $('originalItem').textContent='Original questionnaire item '+c.original_item;
 $('humanLabels').replaceChildren();for(const h of c.human_labels){const p=document.createElement('p');p.textContent=h.reviewer+': '+h.choice+(h.answer?' — '+h.answer:'')+(h.reason?' (Note: '+h.reason+')':'');$('humanLabels').append(p)}
 $('jointAgreement').checked=d.joint_agreement===true;""")
 template=template.replace("if(!choice){", "if(!$('jointAgreement').checked){$('message').textContent='Confirm both reviewers agree before saving. Otherwise leave this item pending.';return false}\n if(!choice){")
 template=template.replace('d.card_sha256=c.card_sha256;','d.joint_agreement=true;d.card_sha256=c.card_sha256;')
 template=template.replace("$('role').textContent=packet.role==='primary'?`Primary review · ${cards.length} science responses`:`Second independent review · ${cards.length} overlapping science responses`;","$('role').textContent=`Joint reconciliation · ${cards.length} responses`; $('jointAgreement').onchange=()=>setDirty(true);")
 body=json.dumps(packet,ensure_ascii=False).replace('<','\\u003c');html=template.replace('__PACKET_JSON__',body)
 extra='''<script>
const originalExport=$('export').onclick;
$('export').onclick=()=>{originalExport();if(!state.reviewer)return;let box=$('exportBackup');if(!box){box=document.createElement('details');box.id='exportBackup';box.innerHTML='<summary>JSON backup — copy if download does not appear</summary><textarea readonly style="height:180px" aria-label="JSON backup"></textarea>';document.querySelector('header').appendChild(box);}box.open=true;box.querySelector('textarea').value=JSON.stringify({schema:packet.schema,packet_id:packet.packet_id,role:packet.role,reviewer:state.reviewer,exported_at:new Date().toISOString(),sample_ids:cards.map(c=>c.review_id),decisions:state.decisions},null,2);};
</script>'''
 html=html.replace('</body>',extra+'</body>');(OUT/'review.html').write_text(html,encoding='utf-8')
 scripts=re.findall(r'<script(?:\s[^>]*)?>(.*?)</script>',html,re.S);js=OUT/'syntax_check.js';js.write_text('\n'.join(s for s in scripts if not s.lstrip().startswith('{')),encoding='utf-8')
 subprocess.run([str(Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'),'--check',str(js)],check=True)
 assert all(not ({'ai_correct','target_option','final_correct','model'} & set(c)) for c in new)
 manifest={'packet_id':packet['packet_id'],'n':11,'original_items':[c['original_item'] for c in new],'source_comparison_sha256':hashlib.sha256((CROSS/'comparison_results.json').read_bytes()).hexdigest(),'policy':'Joint human reconciliation only; no automatic label changes; independent judgments preserved.'}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');print(json.dumps(manifest))
if __name__=='__main__':main()
