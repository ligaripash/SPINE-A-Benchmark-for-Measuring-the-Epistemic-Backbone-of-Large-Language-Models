"""Freeze response labels, blind science extraction inputs, and sample human review.

This performs local file operations only. It never sends model requests.
"""
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import random
import re
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SCORED=ROOT/'model_runs/corrective_branch_v2'
sys.path.insert(0,str(SCORED))
from compute_paper_results import NAMES, PERSONAS, read_rows
sys.path.insert(0,str(ROOT/'model_runs/llama3_8b'))
from score_objective import OPTION,norm


def sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def save_rows(path,rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')


def raw_map(paths):
    found={}
    for path in paths:
        for r in read_rows(path):
            if 'error' in r:continue
            key=(r['id'],r.get('branch'))
            if key in found:
                assert found[key]['messages']==r['messages'] and found[key]['response']==r['response']
            found[key]=r
    return found


def main():
    if (HERE/'manifest.json').exists():
        manifest=json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
        assert digest(HERE/'judge_packet.jsonl')==manifest['judge_packet_sha256']
        assert digest(HERE/'private_response_index.jsonl')==manifest['index_sha256']
        print('Frozen packets already exist; preserved.');return
    data=ROOT/'spine_dataset_corrective_branch_luna_reviewed_v2.json'
    items={r['id']:r for r in json.loads(data.read_text(encoding='utf-8'))}
    packet,index,hashes=[],[],{}
    for name in NAMES:
        original=ROOT/'model_runs'/name
        current=SCORED/name
        base_paths=[original/'base.jsonl']
        attack_paths=[original/'attacks_reused.jsonl',original/'attacks_new.jsonl'] if name.startswith('llama') else [original/'attack.jsonl']
        correction_paths=[current/'correction_reused.jsonl',current/'correction_new.jsonl']
        raws={'base':raw_map(base_paths),'attack':raw_map(attack_paths),'correction':raw_map(correction_paths)}
        raws['base']={(ident,None):r for (ident,_),r in raws['base'].items()}
        score_paths={'base':current/'base_scores_reused.jsonl','attack':current/'attack_scores_reused.jsonl',
                     'correction':current/'correction_scores_final.jsonl'}
        hashes[name]={str(p.relative_to(ROOT)):digest(p) for p in base_paths+attack_paths+correction_paths+list(score_paths.values())}
        for phase,path in score_paths.items():
            for score in read_rows(path):
                ident=score['id'];branch=score.get('branch') if phase=='attack' else (None if phase=='base' else 'correction')
                raw=raws[phase][(ident,branch)]
                item=items[ident]
                assert raw['messages'][0]['content']==item['question_data']['prompt']
                question=item['question_data']['prompt']
                gold=item['question_data']['ground_truth']
                if phase=='correction':
                    correction=item['valid_corrective_intervention']
                    question=correction.get('revised_question',question)
                    gold=correction.get('ground_truth') or correction.get('revised_ground_truth')
                    expected=correction.get('intervention_text',correction.get('correction'))
                    assert raw['messages'][2]['content']==expected
                elif phase=='attack':
                    assert raw['messages'][2]['content']==item['adversarial_attacks'][branch]['attack_text']
                assert gold is not None
                blind=sha(['SPINE-review-20261004',name,ident,branch])[:24]
                public={'task':question,'branch_kind':phase,'original_task':item['question_data']['prompt'],
                    'initial_answer':raw['messages'][1]['content'] if phase!='base' else '',
                    'followup_message':raw['messages'][2]['content'] if phase!='base' else '',
                    'final_response':raw['response']}
                labels,target=[],None
                if item['domain']=='logic':
                    options={k:v.strip(' \t\r\n,.') for k,v in OPTION.findall(question.split('Options:',1)[-1])}
                    labels=list(options)
                    match=[k for k,v in options.items() if norm(v)==norm(str(gold))]
                    if len(match)!=1 and str(gold).strip() in options:match=[str(gold).strip()]
                    assert len(match)==1,(ident,phase,gold,options)
                    target=match[0]
                    packet.append({'blind_id':blind,'judge_input':public,'option_labels':labels,'input_sha256':sha(public)})
                index.append({'blind_id':blind,'run_name':name,'id':ident,'branch':branch,'phase':phase,
                    'domain':item['domain'],'primary_correct':score['correct'],'primary_method':score.get('method'),
                    'target_option':target,'ground_truth':gold,'input_sha256':sha(public),
                    'conversation_sha256':sha(raw['messages']),'response_sha256':sha(raw['response']),
                    'judge_input':public})
    assert len({r['blind_id'] for r in index})==len(index)
    rng=random.Random(20261004)
    rng.shuffle(packet)
    # Probability sample: 24 initial, 36 correction, 60 attack responses.
    # Equal allocation across six models and two non-code domains; attacks also
    # balance personas exactly, using a fixed five-persona allocation per cell.
    human=[]; strata=[]
    for m,name in enumerate(NAMES):
        for d,domain in enumerate(['math','logic']):
            for phase,n in [('base',2),('correction',3)]:
                population=[r for r in index if r['run_name']==name and r['domain']==domain and r['phase']==phase]
                sampled=rng.sample(population,n)
                for r in sampled:
                    human.append({**r,'stratum':f'{name}:{domain}:{phase}', 'inclusion_probability':n/len(population),
                                  'population_size':len(population),'stratum_sample_size':n})
            omitted=PERSONAS[(2*m+d)%6]
            for persona in PERSONAS:
                if persona==omitted:continue
                population=[r for r in index if r['run_name']==name and r['domain']==domain and r['branch']==persona and r['phase']=='attack']
                sampled=rng.choice(population)
                # Ten of twelve model/domain cells are included per persona.
                # Cell allocation is fixed; infer only to the sampled strata.
                human.append({**sampled,'stratum':f'{name}:{domain}:attack:{persona}',
                    'inclusion_probability':1/len(population),'population_size':len(population),'stratum_sample_size':1})
    rng.shuffle(human)
    assert len(human)==120 and len({r['blind_id'] for r in human})==120
    assert Counter(r['branch'] for r in human if r['phase']=='attack')=={p:10 for p in PERSONAS}
    public_human=[{'blind_id':r['blind_id'],'domain':r['domain'],'phase':r['phase'],
        'task':r['judge_input']['task'],'initial_answer':r['judge_input']['initial_answer'],
        'followup_message':r['judge_input']['followup_message'],'response':r['judge_input']['final_response'],
        'reference_answer':r['ground_truth'],'option_labels':next((p['option_labels'] for p in packet if p['blind_id']==r['blind_id']),[])} for r in human]
    save_rows(HERE/'judge_packet.jsonl',packet)
    save_rows(HERE/'private_response_index.jsonl',index)
    save_json(HERE/'human_sample_private.json',human)
    save_json(HERE/'human_sample_blinded.json',public_human)
    # Embed data safely; form never includes model identity or existing labels.
    template=(HERE/'human_review_template.html').read_text(encoding='utf-8')
    embedded=json.dumps(public_human,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    (HERE/'human_response_review.html').write_text(template.replace('__REVIEW_DATA__',embedded),encoding='utf-8')
    manifest={'dataset_sha256':digest(data),'science_judgments_per_judge':len(packet),'response_index_count':len(index),
        'counts':dict(Counter(r['judge_input']['branch_kind'] for r in packet)), 'source_hashes':hashes,
        'judge_packet_sha256':digest(HERE/'judge_packet.jsonl'),'index_sha256':digest(HERE/'private_response_index.jsonl'),
        'human_sample_sha256':digest(HERE/'human_sample_blinded.json'),
        'human_sample_ids':[r['blind_id'] for r in human],'human_sample_n':120,
        'sampling_seed':20261004,'human_scope':'Math/science response correctness, no code questions; 24 base, 60 attack, 36 correction',
        'human_inference':'Descriptive balanced audit, not a dataset-wide prevalence estimate. Persona allocation omits two fixed model/domain cells per persona; unequal strata and cluster dependencies must be respected.',
        'analysis_scope':'All recorded science baseline responses and primary-eligible science attack/correction responses. Mathematics and finite code-test labels fixed. Targets fixed. API judges receive only blinded task and response context, never targets or existing labels.',
        'planned_judges':['gpt-6-luna','gemini-3.1-pro-preview'],
        'ensemble':'Primary hybrid scorer plus two independent API extractors. This is a three-scorer ensemble, not three independent LLM judges.',
        'eligibility_sensitivity':'Fixed-primary-cohort comparison and separate intersection of primary-eligible and scheme-confirmed initial answers; independently promoted anchors are not added.',
        'status':'prepared; external transmission awaiting explicit approval and private credentials; human labels pending'}
    save_json(HERE/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ['science_judgments_per_judge','counts','human_sample_n','status']},indent=2))


if __name__=='__main__':main()
