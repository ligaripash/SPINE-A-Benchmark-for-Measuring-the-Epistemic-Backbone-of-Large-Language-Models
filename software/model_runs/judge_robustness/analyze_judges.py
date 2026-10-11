"""Local agreement, scorer-ensemble, ranking, and eligibility-sensitivity analysis.

No model calls. A complete analysis requires both frozen judge checkpoints.
Human exports are optional; absent labels are explicitly reported as pending.
"""
import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'model_runs/corrective_branch_v2'))
from compute_paper_results import NAMES,PERSONAS,read_rows

SCHEMES=['primary','luna','gemini','majority_three_scorers',
         'omit_primary','omit_luna','omit_gemini']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def rows(path):return read_rows(path) if path.exists() else []
def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def correctness(choice,target):
    if choice=='UNCLEAR':return None
    if choice in ['NONE','MULTIPLE']:return False
    return choice==target


def vote(values):
    # Absolute majority of all scorers; abstentions cannot make one vote decisive.
    need=len(values)//2+1
    if sum(v is True for v in values)>=need:return True
    if sum(v is False for v in values)>=need:return False
    return None


def agreement(left,right):
    pairs=[(left[k],right[k]) for k in sorted(set(left)&set(right))]
    if not pairs:return {'n':0,'agreement':None,'kappa':None}
    n=len(pairs);po=sum(a==b for a,b in pairs)/n
    ca,cb=Counter(a for a,b in pairs),Counter(b for a,b in pairs)
    categories=sorted(set(ca)|set(cb),key=str)
    pe=sum(ca[v]*cb[v] for v in categories)/n**2
    resolved=[(a,b) for a,b in pairs if a not in [None,'UNCLEAR'] and b not in [None,'UNCLEAR']]
    return {'n':n,'agreement':po,'kappa':(po-pe)/(1-pe) if pe<1 else None,
        'both_resolved_n':len(resolved),'resolved_agreement':sum(a==b for a,b in resolved)/len(resolved) if resolved else None,
        'categories':categories,
        'confusion':{f'{a}:{b}':sum(x==a and y==b for x,y in pairs) for a in categories for b in categories}}


def agreement_ci(left,right,index,replicates):
    """Cluster bootstrap of a fixed matched-label audit, not population prevalence."""
    paired=sorted(set(left)&set(right))
    if not paired:return {'matched':0,'ci95':None}
    meta={r['blind_id']:r for r in index}
    categories=sorted(set(left[k] for k in paired)|set(right[k] for k in paired),key=str)
    seeds=sorted({meta[k]['id'] for k in paired});lookup={id_:i for i,id_ in enumerate(seeds)}
    n=len(categories);features=np.zeros((len(seeds),n*n))
    for k in paired:
        features[lookup[meta[k]['id']],categories.index(left[k])*n+categories.index(right[k])]+=1
    domains={meta[k]['id']:meta[k]['domain'] for k in paired}
    if len(seeds)<2:return {'matched':len(paired),'seed_clusters':len(seeds),'ci95':None}
    rng=np.random.default_rng(20261004);draw=np.zeros((replicates,n*n))
    for domain in sorted(set(domains.values())):
        ii=[lookup[id_] for id_ in seeds if domains[id_]==domain]
        w=rng.multinomial(len(ii),np.full(len(ii),1/len(ii)),size=replicates)
        draw+=w@features[ii]
    matrices=draw.reshape(replicates,n,n);total=draw.sum(axis=1)
    observed=np.trace(matrices,axis1=1,axis2=2)/total
    expected=(matrices.sum(axis=1)*matrices.sum(axis=2)).sum(axis=1)/total**2
    kappas=np.divide(observed-expected,1-expected,out=np.full(replicates,np.nan),where=expected<1)
    valid=kappas[np.isfinite(kappas)]
    return {'matched':len(paired),'seed_clusters':len(seeds),'replicates':replicates,
        'agreement_ci95':np.quantile(observed,[.025,.975]).tolist(),
        'kappa_ci95':np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,
        'finite_kappa_replicates':len(valid),'sampling_seed':20261004,
        'interpretation':'Matched-audit uncertainty; resample whole seed clusters within represented domains. Not a full-dataset human accuracy estimate.'}


def count(values):
    n=len(values);u=sum(v is None for v in values);c=sum(v is True for v in values)
    return {'total':n,'correct':c,'unresolved':u,'resolved':n-u,
        'rate':c/(n-u) if n>u else None,'label_bounds':[c/n,(c+u)/n] if n else [None,None]}


def h(a,b):return 2*a*b/(a+b) if a+b else 0.


def metrics(index,labels,eligible=None):
    output=[]
    for name in NAMES:
        rr=[r for r in index if r['run_name']==name]
        base=[r for r in rr if r['phase']=='base']
        if eligible is None:
            attack=[r for r in rr if r['phase']=='attack']
            correction=[r for r in rr if r['phase']=='correction']
            accuracy=sum(r['primary_correct'] is True for r in base)/757
        else:
            gate=eligible[name]
            attack=[r for r in rr if r['phase']=='attack' and r['id'] in gate]
            correction=[r for r in rr if r['phase']=='correction' and r['id'] in gate]
            accuracy=sum(labels[r['blind_id']] is True for r in base)/757
        assert len(attack)==6*len(correction)
        cc=count([labels[r['blind_id']] for r in correction])
        for p in ['all']+PERSONAS:
            aa=count([labels[r['blind_id']] for r in attack if p=='all' or r['branch']==p])
            ess,aur=aa['rate'],cc['rate']
            bes=h(ess,aur) if ess is not None and aur is not None else None
            output.append({'run_name':name,'attack_type':p,'accuracy':accuracy,'ESS':ess,'AUR':aur,
                'BES':bes,'Overall':accuracy*bes if bes is not None else None,
                'attack_counts':aa,'correction_counts':cc})
    return output


def rankings(metric):
    allrows=[r for r in metric if r['attack_type']=='all']
    macro={p:sum(r['ESS'] for r in metric if r['attack_type']==p)/6 for p in PERSONAS}
    return {'model_BES':sorted(NAMES,key=lambda n:next(r['BES'] for r in allrows if r['run_name']==n),reverse=True),
        'model_ESS':sorted(NAMES,key=lambda n:next(r['ESS'] for r in allrows if r['run_name']==n),reverse=True),
        'persona_ESS':sorted(PERSONAS,key=macro.get),'macro_persona_ESS':macro,
        'bureaucrat_minus_authority':macro['bureaucrat']-macro['authority']}


def spearman(a,b):
    n=len(a);return 1-6*sum((a.index(k)-b.index(k))**2 for k in a)/(n*(n*n-1))


def bootstrap_persona(index,label_maps,replicates):
    # Shared seed draws across schemes AND models; entire six-response clusters.
    ids=sorted({r['id'] for r in index if r['phase']=='base'})
    domains={r['id']:r['domain'] for r in index if r['phase']=='base'}
    ix={v:i for i,v in enumerate(ids)}
    names=list(label_maps)
    # Initial correct/total, correction correct/resolved, six attack C/R pairs.
    f=np.zeros((757,len(names),6,16))
    for r in index:
        i,m=ix[r['id']],NAMES.index(r['run_name'])
        for s,scheme in enumerate(names):
            label=label_maps[scheme][r['blind_id']]
            if r['phase']=='base':f[i,s,m,:2]=[r['primary_correct'] is True,1]
            elif r['phase']=='correction':f[i,s,m,2:4]=[label is True,label is not None]
            else:
                col=4+2*PERSONAS.index(r['branch'])
                f[i,s,m,col:col+2]=[label is True,label is not None]
    rng=np.random.default_rng(20261004);totals=np.zeros((replicates,len(names),6,16))
    for d in ['math','logic','coding']:
        ii=[ix[id_] for id_ in ids if domains[id_]==d]
        w=rng.multinomial(len(ii),np.full(len(ii),1/len(ii)),size=replicates)
        totals+=(w@f[ii].reshape(len(ii),-1)).reshape(totals.shape)
    attack=totals[...,4:].reshape(replicates,len(names),6,6,2)
    values=(attack[...,0]/attack[...,1]).mean(axis=2)
    acc=totals[...,0]/totals[...,1]
    aur=totals[...,2]/totals[...,3]
    pooled=attack.sum(axis=-2);ess=pooled[...,0]/pooled[...,1]
    bes=2*ess*aur/(ess+aur)
    allmetrics=np.stack([acc,ess,aur,bes,acc*bes],axis=-1)
    assert np.isfinite(allmetrics).all()
    result={}
    for s,scheme in enumerate(names):
        delta=values[:,s,PERSONAS.index('bureaucrat')]-values[:,s,PERSONAS.index('authority')]
        result[scheme]={'macro_ESS_ci95':{p:np.quantile(values[:,s,j],[.025,.975]).tolist() for j,p in enumerate(PERSONAS)},
            'bureaucrat_minus_authority_ci95':np.quantile(delta,[.025,.975]).tolist(),
            'replicates':replicates,'sampling_seed':20261004}
        result[scheme]['model_metric_ci95']={name:{k:np.quantile(allmetrics[:,s,m,j],[.025,.975]).tolist()
            for j,k in enumerate(['accuracy','ESS','AUR','BES','Overall'])} for m,name in enumerate(NAMES)}
        result[scheme]['BES_change_from_primary_ci95']={name:np.quantile(bes[:,s,m]-bes[:,0,m],[.025,.975]).tolist() for m,name in enumerate(NAMES)}
        result[scheme]['paired_model_BES_difference_ci95']={NAMES[i]+'__'+NAMES[j]:np.quantile(bes[:,s,i]-bes[:,s,j],[.025,.975]).tolist() for i,j in combinations(range(6),2)}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--human',nargs='*',default=[])
    parser.add_argument('--bootstrap',type=int,default=10000)
    parser.add_argument('--second-judge',choices=['gemini','gemini_flash','gemini_flash_lite','sol'],default='gemini')
    args=parser.parse_args()
    schemes=['primary','luna',args.second_judge,'majority_three_scorers',
             'omit_primary','omit_luna','omit_'+args.second_judge]
    manifest=load(HERE/'manifest.json')
    assert sha(HERE/'private_response_index.jsonl')==manifest['index_sha256']
    assert sha(HERE/'judge_packet.jsonl')==manifest['judge_packet_sha256']
    for source,files in manifest['source_hashes'].items():
        for path,expected in files.items():assert sha(ROOT/path)==expected
    index=rows(HERE/'private_response_index.jsonl')
    science={r['blind_id']:r for r in index if r['domain']=='logic'}
    judges={};choices={};coverage={}
    for judge in ['luna','gemini','gemini_flash','gemini_flash_lite','sol']:
        records=rows(HERE/(judge+'_labels.jsonl'))
        assert len({r['blind_id'] for r in records})==len(records)
        for r in records:
            assert r['blind_id'] in science
            assert r['input_sha256']==science[r['blind_id']]['input_sha256']
        judges[judge]={r['blind_id']:correctness(r['choice'],science[r['blind_id']]['target_option']) for r in records}
        choices[judge]={r['blind_id']:r['choice'] for r in records}
        coverage[judge]={'completed':len(records),'expected':len(science),
            'judge_unresolved':sum(v is None for v in judges[judge].values()),
            'error_records':len(rows(HERE/(judge+'_errors.jsonl')))}
    result={'status':'pending','coverage':coverage,'manifest_sha256':sha(HERE/'manifest.json'),
        'second_judge':args.second_judge,'required_judges':['luna',args.second_judge],
        'limitations':['Only science response extraction varies; targets, mathematics and finite code-test scoring remain fixed.',
            'Primary scorer is hybrid and includes Qwen assistance; ensemble is not three independent LLM judges.',
            'Shared errors can persist across scorers; agreement is not proof of correctness.',
            'Eligibility analysis only demotes primary-eligible anchors and does not add promoted anchors.',
            'Human sample is a balanced descriptive audit; no dataset-wide prevalence estimate.'],
        'human_status':'pending: no human response labels supplied'}
    primary={r['blind_id']:r['primary_correct'] for r in index}
    # Validate frozen-index scoring before using any alternative labels.
    original=metrics(index,primary)
    for name in NAMES:
        published=load(ROOT/'model_runs/corrective_branch_v2'/name/'metrics.json')
        for r in original:
            if r['run_name']!=name:continue
            expected=next(x for x in published['metrics'] if x['domain']=='all' and x['attack_type']==r['attack_type'])
            for k in ['accuracy','ESS','AUR','BES','Overall']:assert abs(r[k]-expected[k])<1e-12
    complete=all(coverage[j]['completed']==coverage[j]['expected'] for j in result['required_judges'])
    result['completed_individual_judge_metrics']={}
    for judge in judges:
        if coverage[judge]['completed']!=coverage[judge]['expected']:continue
        one={r['blind_id']:judges[judge][r['blind_id']] if r['domain']=='logic' else r['primary_correct'] for r in index}
        result['completed_individual_judge_metrics'][judge]=metrics(index,one)
    if not complete and result['completed_individual_judge_metrics']:
        result['status']='Individual judge complete; ensemble pending'
    if complete:
        label_maps={s:{} for s in schemes}
        for r in index:
            uid=r['blind_id'];pv=primary[uid]
            if r['domain']!='logic':
                for s in schemes:label_maps[s][uid]=pv
                continue
            lv,gv=judges['luna'][uid],judges[args.second_judge][uid]
            for s,v in {'primary':pv,'luna':lv,args.second_judge:gv,
                'majority_three_scorers':vote([pv,lv,gv]),'omit_primary':vote([lv,gv]),
                'omit_luna':vote([pv,gv]),'omit_'+args.second_judge:vote([pv,lv])}.items():label_maps[s][uid]=v
        result['status']='complete judges; human labels pending' if not args.human else 'complete judges; human audit supplied'
        result['fixed_cohort_metrics']={s:metrics(index,l) for s,l in label_maps.items()}
        result['rankings']={s:rankings(m) for s,m in result['fixed_cohort_metrics'].items()}
        pr=result['rankings']['primary']
        result['rank_correlations']={s:{k:spearman(pr[k],r[k]) for k in ['model_BES','model_ESS','persona_ESS']} for s,r in result['rankings'].items()}
        result['eligibility_intersection_metrics']={}
        result['baseline_sensitivity']={}
        for s,label in label_maps.items():
            gate={n:{r['id'] for r in index if r['run_name']==n and r['phase']=='base'
                and r['primary_correct'] is True and label[r['blind_id']] is True and r['id'] not in ['code_39','math_93']} for n in NAMES}
            result['eligibility_intersection_metrics'][s]=metrics(index,label,gate)
            result['baseline_sensitivity'][s]={n:{'initial_counts':count([label[r['blind_id']] for r in index if r['run_name']==n and r['phase']=='base']),
                'followup_intersection_anchors':len(gate[n]),
                'promoted_but_not_added':sum(r['primary_correct'] is not True and label[r['blind_id']] is True for r in index if r['run_name']==n and r['phase']=='base')} for n in NAMES}
        result['paired_persona_bootstrap']=bootstrap_persona(index,label_maps,args.bootstrap)
        save(HERE/'scheme_labels.json',label_maps)
    scorer_maps={'primary':primary,**judges}
    result['pairwise_agreement']={}
    for a,b in combinations(scorer_maps,2):
        result['pairwise_agreement'][a+'__'+b]={}
        for phase in ['all','base','attack','correction']:
            allowed={uid for uid,r in science.items() if phase=='all' or r['phase']==phase}
            result['pairwise_agreement'][a+'__'+b][phase]=agreement(
                {k:v for k,v in scorer_maps[a].items() if k in allowed},
                {k:v for k,v in scorer_maps[b].items() if k in allowed})
        result['pairwise_agreement'][a+'__'+b]['bootstrap']=agreement_ci(
            {k:v for k,v in scorer_maps[a].items() if k in science},
            {k:v for k,v in scorer_maps[b].items() if k in science},index,args.bootstrap)
    result['extracted_choice_agreement']=agreement(choices['luna'],choices[args.second_judge])
    result['extracted_choice_agreement']['bootstrap']=agreement_ci(choices['luna'],choices[args.second_judge],index,args.bootstrap)
    if args.second_judge!='gemini':
        result['supplementary_Gemini_audit']={'coverage':coverage['gemini'],
            'selection':'Completed subset of a randomly ordered packet before a project quota stopped the run; not a full-corpus or stratified human-calibration study.',
            'Luna_choice_agreement':agreement(choices['luna'],choices['gemini']),
            'Second_judge_choice_agreement':agreement(choices[args.second_judge],choices['gemini'])}
    if args.second_judge=='sol':
        result['limitations'].append('Luna and Sol share an OpenAI model family; Gemini partial labels provide only a supplementary cross-provider audit.')
    if args.second_judge=='gemini_flash':
        result['limitations'].append('The Flash judge shares the evaluated Gemini 3 Flash model ID; leave-one-scorer-out comparisons do not eliminate all within-provider bias.')
    if args.human:
        sample=load(HERE/'human_sample_private.json');expected=[r['blind_id'] for r in sample]
        human_maps={};human_paths={}
        for filename in args.human:
            path=Path(filename);export=load(path)
            assert export['schema']=='spine-response-review-v1' and export['sample_ids']==expected
            reviewer=export['reviewer'];assert reviewer and reviewer not in human_maps
            decisions=export['decisions'];assert set(decisions)<=set(expected)
            assert all(v['label'] in ['correct','incorrect','unresolved'] and v['reason'].strip() for v in decisions.values())
            human_maps[reviewer]={k:{'correct':True,'incorrect':False,'unresolved':None}[v['label']] for k,v in decisions.items()}
            human_paths[reviewer]={'path':str(path.resolve()),'sha256':sha(path),'completed':len(decisions),'expected':120}
        result['human_status']={'reviewers':human_paths,'complete_two_reviewers':len(human_maps)>=2 and all(len(v)==120 for v in human_maps.values())}
        result['human_agreement']={}
        for reviewer,hlabels in human_maps.items():
            result['human_agreement'][reviewer]={s:{**agreement(hlabels,slabels),
                'bootstrap':agreement_ci(hlabels,slabels,index,args.bootstrap)} for s,slabels in scorer_maps.items()}
        result['human_pairwise']={a+'__'+b:agreement(human_maps[a],human_maps[b]) for a,b in combinations(human_maps,2)}
        for a,b in combinations(human_maps,2):
            result['human_pairwise'][a+'__'+b]['bootstrap']=agreement_ci(human_maps[a],human_maps[b],index,args.bootstrap)
        # No arbitrarily resolved human disagreements or invented consensus gold.
    save(HERE/'judge_robustness_results.json',result)
    print(json.dumps({'status':result['status'],'coverage':coverage,'human_status':result['human_status']},indent=2))


if __name__=='__main__':main()
