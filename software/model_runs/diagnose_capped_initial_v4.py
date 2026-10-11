"""Read-only diagnosis of the 94 capped initial-answer recovery receipts."""
import json,re,hashlib
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'model_runs/output_budget_v4'
rows=[]
for p in sorted((OUT/'initial_candidates').glob('*/*/needs_attention.json')):
 receipt=p.parent/'4096.json'
 d=json.loads(receipt.read_text());r=d['result'];m=r.get('message',{})
 text=m.get('thinking','') or m.get('content','')
 words=re.findall(r"\S+",text[-4000:].lower())
 grams=[tuple(words[i:i+12]) for i in range(max(0,len(words)-11))]
 counts=Counter(grams)
 rows.append({'model':p.parent.parent.name,'id':p.parent.name,'domain':p.parent.name.split('_')[0],
 'receipt':str(receipt.relative_to(ROOT)),'sha256':hashlib.sha256(receipt.read_bytes()).hexdigest(),
 'finish_reason':r.get('done_reason'),'generated_tokens':r.get('eval_count'),'prompt_tokens':r.get('prompt_eval_count'),
 'context_budget':d.get('num_ctx'),'content_characters':len(m.get('content','')),'thinking_characters':len(m.get('thinking','')),
 'tail_repeated_12word_fraction':round(sum(n-1 for n in counts.values())/max(1,len(grams)),4),
 'tail_max_12word_repetitions':max(counts.values(),default=0)})
summary={'count':len(rows),'by_model':dict(Counter(r['model'] for r in rows)),
 'by_domain':dict(Counter(r['domain'] for r in rows)),
 'empty_final_content':sum(r['content_characters']==0 for r in rows),
 'all_explicit_length_at_4096':all(r['finish_reason']=='length' and r['generated_tokens']==4096 for r in rows),
 'prompt_token_range':[min(r['prompt_tokens'] for r in rows),max(r['prompt_tokens'] for r in rows)],
 'prompt_plus_output_below_context':all(r['prompt_tokens']+r['generated_tokens']<r['context_budget'] for r in rows),
 'repetition_screen_at_least_5_copies_of_a_12word_sequence':sum(r['tail_max_12word_repetitions']>=5 for r in rows),
 'interpretation':'Repetition screen is a diagnostic heuristic, not a correctness label. DeepSeek thinking is kept separate from final response. Short prompts plus output remain below configured context; explicit termination reason is output length.',
 'examples':{'llama3_8b_revised/math_125':'Repeats the same spoon-problem restatement.',
 'qwen2_5_7b/math_236':'Continues numbered pen-recycling calculations through cycle 28 without terminating.',
 'deepseek_r1_1_5b/code_10':'Repeats the same attempted palindrome construction without returning final content.'}}
(OUT/'capped_initial_diagnosis.json').write_text(json.dumps({'summary':summary,'records':rows},indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
if __name__=='__main__':pass
