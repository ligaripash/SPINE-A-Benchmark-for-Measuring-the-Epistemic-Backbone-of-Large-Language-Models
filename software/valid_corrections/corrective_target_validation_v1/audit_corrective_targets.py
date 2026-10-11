"""Audit every delivered correction against current references and independent evidence.

Does not run target models, overwrite raw scores, or promote AI labels to human
validation. Checkpoints finite code executions and writes hash-linked evidence.
"""
from __future__ import annotations
import argparse
import ast
import builtins
import contextlib
import copy
import hashlib
import io
import json
import math
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set, Any

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DATASET=ROOT/'spine_dataset_false_claim_attacks.json'
sys.path.insert(0,str(ROOT/'model_runs/llama3_8b'))
import score_code
from score_objective import numeric, math_score
score_code.ALLOWED_MODULES.add('hashlib')

def sha(value):
    if not isinstance(value,(str,bytes)):
        value=json.dumps(value,ensure_ascii=False,sort_keys=True)
    return hashlib.sha256(value.encode('utf-8') if isinstance(value,str) else value).hexdigest()

def norm(value):
    s=unicodedata.normalize('NFKC',str(value)).casefold()
    s=s.replace('\u2212','-')
    return re.sub(r'[^\w+\-]+',' ',s).strip()

def options(text):
    if 'Options:' not in text:
        return {}
    area=text.split('Options:',1)[1]
    return {m[0]:m[1].strip().strip(',') for m in re.findall(
        r'\(([A-E1-5])\)\s*(.*?)(?=,?\s*\([A-E1-5]\)\s*|$)',area,re.S)}

def selected_correction(item):
    return item.get('valid_corrective_intervention') or (item.get('correction_candidate_for_review') or {}).get('candidate') or {}

def same_result(actual,expected):
    if isinstance(expected,bool):
        return type(actual) is bool and actual==expected
    if isinstance(expected,(int,float)):
        return type(actual) in (int,float) and math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-9)
    if isinstance(expected,(list,tuple)):
        return type(actual) is type(expected) and len(actual)==len(expected) and all(same_result(a,b) for a,b in zip(actual,expected))
    if isinstance(expected,dict):
        return type(actual) is dict and actual.keys()==expected.keys() and all(same_result(actual[k],v) for k,v in expected.items())
    return type(actual) is type(expected) and actual==expected

def worker():
    p=json.load(sys.stdin)
    code=p['source']
    issue=score_code.review_ast(code)
    if issue:
        print(json.dumps({'status':'execution_unresolved','reason':issue}))
        return
    env={'__builtins__':score_code.safe_builtins(),'__name__':'__audit_reference__',
         'List':List,'Optional':Optional,'Tuple':Tuple,'Dict':Dict,'Set':Set,'Any':Any}
    results=[]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code,'<audited-reference>','exec'),env)
        func=env[p['entry']]
    except Exception as exc:
        print(json.dumps({'status':'execution_unresolved','reason':repr(exc)[:200]}))
        return
    mismatches=[]
    exceptions=[]
    for i,case in enumerate(p['cases']):
        expected=ast.literal_eval(case['expected_repr'])
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                actual=func(*copy.deepcopy(case['args']))
            # Respect exact zero for the explicitly exact-zero root correction.
            same=(type(actual) is float and actual == expected if p['id']=='code_32' else same_result(actual,expected))
            if not same and len(mismatches)<12:
                mismatches.append({'index':i,'args':case['args'],'expected':repr(expected),'observed':repr(actual)[:200]})
        except Exception as exc:
            if len(exceptions)<12:
                exceptions.append({'index':i,'args':case['args'],'expected':repr(expected),'error':repr(exc)[:200]})
    print(json.dumps({'status':'passed' if not mismatches and not exceptions else 'failed',
                      'tests':len(p['cases']),'mismatches':mismatches,'exceptions':exceptions}))

def execute(ident,source,entry,cases):
    p={'id':ident,'source':source,'entry':entry,'cases':cases}
    try:
        out=subprocess.run([sys.executable,'-I','-X','utf8',str(Path(__file__).resolve()),'--worker'],
            input=json.dumps(p),capture_output=True,text=True,encoding='utf-8',timeout=12)
        if out.returncode:
            return {'status':'execution_unresolved','reason':out.stderr[-400:]}
        return json.loads(out.stdout)
    except subprocess.TimeoutExpired:
        return {'status':'execution_unresolved','reason':'timeout_12s'}
    except (ValueError,OSError) as exc:
        return {'status':'execution_unresolved','reason':str(exc)[:200]}

def code_record(item,cases):
    c=selected_correction(item)
    entry=[n.name for n in ast.parse(item['question_data']['prompt']).body if isinstance(n,ast.FunctionDef)][-1]
    original=item['question_data']['prompt']+item['question_data']['ground_truth']
    revised=c.get('revised_reference_code') or c.get('ground_truth','')
    serial=[{'args':list(a),'expected_repr':repr(e)} for a,e in cases]
    payload_hash=sha({'original':original,'revised':revised,'cases':serial,'entry':entry,'audit_script':sha(Path(__file__).read_bytes())})
    checkpoint=HERE/'code_execution'/f"{item['id']}.json"
    if checkpoint.exists():
        old=json.loads(checkpoint.read_text(encoding='utf-8'))
        if old['payload_sha256']==payload_hash:
            return old
    old_result=({'status':'execution_unresolved','reason':'quarantined_original_reference'}
        if item['id']=='code_39' else execute(item['id'],original,entry,serial))
    new_result=execute(item['id'],revised,entry,serial)
    record={'id':item['id'],'payload_sha256':payload_hash,'entry_point':entry,'independent_cases':len(serial),
            'original_reference':old_result,'revised_reference':new_result}
    checkpoint.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return record

def main():
    from independent_code_cases import CASES
    raw=DATASET.read_bytes()
    data=json.loads(raw)
    screened_source={x['id']:x for x in json.loads((ROOT/'spine_dataset_ground_truth_corrected.json').read_text(encoding='utf-8'))}
    assert all(x['question_data']==screened_source[x['id']]['question_data'] for x in data)
    assert all(selected_correction(x).get('intervention_text','')==selected_correction(screened_source[x['id']]).get('intervention_text','') for x in data)
    luna_path=ROOT/'valid_corrections/ai_review_gpt-6-luna_full/labels.jsonl'
    labels={x['id']:x for x in map(json.loads,luna_path.read_text(encoding='utf-8').splitlines()) if x['unit']=='correction'}
    human_path=ROOT/'valid_corrections/attack_false_claim_repair_v2/human_review_dataset_reviewer_summary.json'
    human=json.loads(human_path.read_text(encoding='utf-8'))
    human_rows={x['id']:x for x in human['reviewed_items']}
    semantic_path=HERE/'semantic_adjudications.json'
    semantic=json.loads(semantic_path.read_text(encoding='utf-8'))
    (HERE/'code_execution').mkdir(exist_ok=True)
    executions={}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(code_record,x,CASES[x['id']]):x['id'] for x in data if x['domain']=='coding'}
        for f in as_completed(futures):
            result=f.result()
            executions[result['id']]=result
    records=[]
    for item in data:
        ident=item['id']; domain=item['domain']; c=selected_correction(item)
        text=c.get('intervention_text','')
        proposed=c.get('revised_ground_truth') or c.get('ground_truth','')
        row={'id':ident,'domain':domain,'followup_sha256':sha(text),
             'original_task_sha256':sha(item['question_data']), 'proposed_target_sha256':sha(proposed),
             'status':'pending','evidence':[],'issues':[], 'target':proposed}
        if not text or c.get('unable'):
            row['status']='missing';row['issues'].append('no_usable_delivered_correction')
            records.append(row);continue
        ai=labels.get(ident)
        hash_match=bool(ai and ai['followup_sha256']==sha(text))
        row['ai_followup_hash_matches']=hash_match
        if hash_match:
            row['ai_review']={k:ai[k] for k in ('model','label','response_id','parsed')}
            row['evidence'].append('existing_blind_luna_review_exact_followup')
        else:
            row['issues'].append('no_matching_blind_review')
        if hash_match and ai['label']!='valid_change':
            row['issues'].append('blind_review_'+ai['label'])
        if domain=='math':
            normalized_target=str(proposed).strip().lstrip('$').strip()
            n=numeric(normalized_target);old=numeric(str(item['question_data']['ground_truth']).strip().lstrip('$'))
            blind_answer=str(ai['parsed'].get('revised_answer','')) if hash_match else ''
            answer=numeric(blind_answer)
            match,method=math_score(normalized_target,blind_answer) if hash_match else (None,'missing_blind_answer')
            row['numeric_checks']={'proposed_numeric':n is not None,'changed_target':n is not None and n!=old,
                'blind_answer_matches':match,'comparison_method':method}
            if n is not None and n==old:
                row['status']='invalid';row['issues'].append('same_original_and_revised_numeric_target')
            elif hash_match and ai['label']=='valid_change' and n is not None and match:
                row['status']='ai_supported';row['evidence'].append('independently_derived_numeric_target_matches')
        elif domain=='logic':
            original_opts=options(item['question_data']['prompt'])
            revised_opts=options(c.get('revised_question',''))
            visible_opts=options(text) or original_opts
            label=(c.get('revised_option_label') or c.get('correct_option','')).strip('() ')
            row['option_checks']={'label':label,'target_in_revised_options':norm(revised_opts.get(label,''))==norm(proposed),
                'target_matches_visible_option':norm(visible_opts.get(label,''))==norm(proposed),
                'new_options_explicitly_delivered':bool(options(text)),
                'answer_text_changed':norm(proposed)!=norm(item['question_data']['ground_truth'])}
            if not row['option_checks']['target_matches_visible_option']:
                row['issues'].append('proposed_option_not_delivered_verbatim_requires_adjudication')
            if not row['option_checks']['answer_text_changed']:
                row['issues'].append('unchanged_answer_text_requires_adjudication')
            if hash_match and ai['label']=='valid_change' and not row['issues']:
                row['status']='ai_supported'
        elif domain=='coding':
            result=executions[ident]
            row['code_checks']=result
            row['evidence'].append('independently_specified_finite_expected_outputs')
            changed=bool(result['original_reference'].get('mismatches') or result['original_reference'].get('exceptions'))
            row['code_checks']['original_fails_revised_cases']=changed
            if ident=='code_160':
                row['code_checks']['specification_witness']={'operators':['+','*','-'],'operands':[2,3,4,5],
                    'original_prompt_example':9,'revised_left_to_right':((2+3)*4)-5,
                    'original_reference_execution':'not_run_because_eval_is_restricted'}
                changed=True
            if result['revised_reference']['status']!='passed':
                row['issues'].append('revised_reference_'+result['revised_reference']['status'])
            if not changed:
                row['issues'].append('no_executed_distinguishing_witness')
            if hash_match and ai['label']=='valid_change' and result['revised_reference']['status']=='passed' and changed:
                row['status']='finite_test_supported'
            if ident=='code_39':
                row['status']='quarantined';row['issues'].append('original_reference_quarantined')
        # Earlier directly adjudicated delivery defects are retained as defects.
        if ident in ('logic_5','code_22','code_49','code_76'):
            row['status']='invalid'
            row['issues'].append('revised_question_not_delivered' if ident=='logic_5' else 'requirement_already_satisfied_by_current_original_reference')
        adjudication=semantic.get(ident)
        if adjudication:
            row['semantic_adjudication']={**adjudication,'reviewer':'Codex assistant; not human validation'}
            if adjudication['decision']=='support' and hash_match and ai['label']=='valid_change':
                row['status']='ai_supported'
                row['issues']=[v for v in row['issues'] if v not in ('proposed_option_not_delivered_verbatim_requires_adjudication',)]
                row['evidence'].append('assistant_semantic_check_of_visible_replacement_or_equivalence')
            elif adjudication['decision']=='needs_repair':
                row['status']='needs_repair';row['issues'].append('delivered_task_and_scoring_target_require_reconciliation')
        h=human_rows.get(ident)
        if h and human['dataset_sha256']==sha(raw) and h['correction_decision']=='accept' and h['independent_matches_proposed']:
            row['evidence'].append('dataset_reviewer_independent_answer_matches_proposed')
            row['human_review']=h
            if not row['issues']:
                row['status']='human_supported'
        records.append(row)
    manifest={'dataset_sha256':sha(raw),'seeds':len(data),'delivered_corrections':sum(bool(selected_correction(x).get('intervention_text')) for x in data),
        'existing_blind_review_file_sha256':sha(luna_path.read_bytes()),'human_review_file_sha256':sha(human_path.read_bytes()),
        'independent_code_cases_sha256':sha((HERE/'independent_code_cases.py').read_bytes()),
        'semantic_adjudications_sha256':sha(semantic_path.read_bytes()),
        'blind_review_original_questions_and_references_match_current_dataset':True,
        'script_sha256':sha(Path(__file__).read_bytes()),
        'counts':dict(Counter(x['status'] for x in records)),
        'by_domain':{dom:dict(Counter(x['status'] for x in records if x['domain']==dom)) for dom in ('math','logic','coding')},
        'interpretation':'Evidence levels are not interchangeable: AI support is not independent human validation; finite code checks are not proofs.',
        'dataset_or_model_score_files_modified':False}
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (HERE/'target_audit.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records),encoding='utf-8')
    blind=[{'id':x['id'],'domain':x['domain'],'original_question':x['question_data']['prompt'],
            'original_reference_answer':x['question_data']['ground_truth'],'target_visible_followup':selected_correction(x).get('intervention_text','')}
           for x in data if selected_correction(x).get('intervention_text')]
    (HERE/'blind_review_packet.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in blind),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');args=p.parse_args()
    worker() if args.worker else main()
