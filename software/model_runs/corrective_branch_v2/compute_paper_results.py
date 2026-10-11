"""Derive ARR results and matched-anchor metrics from fixed scored responses.

No generation, label changes, or additional model calls are performed.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = ['llama3_8b_revised', 'qwen2_5_7b', 'mistral_nemo',
         'deepseek_r1_1_5b', 'gpt_4o_mini', 'gemini_3_flash']
PERSONAS = ['sophistry', 'authority', 'consensus', 'victim', 'doubter', 'bureaucrat']


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def counts(rows):
    c = Counter('correct' if r['correct'] is True else 'incorrect' if r['correct'] is False else 'unresolved' for r in rows)
    total = len(rows)
    resolved = c['correct'] + c['incorrect']
    return dict(correct=c['correct'], incorrect=c['incorrect'], unresolved=c['unresolved'], total=total,
                resolved_only=c['correct']/resolved if resolved else None,
                bounds=[c['correct']/total, (c['correct']+c['unresolved'])/total] if total else [None, None])


def harmonic(a, b):
    if a is None or b is None:
        return None
    return 2*a*b/(a+b) if a+b else 0.0


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    models, eligible, inputs = [], [], {}
    for name in NAMES:
        directory = HERE/name
        metrics_path = directory/'metrics.json'
        metric = json.loads(metrics_path.read_text(encoding='utf-8'))
        ids = set(json.loads((directory/'correction_eligible_ids.json').read_text(encoding='utf-8')))
        base = read_rows(directory/'base_scores_reused.jsonl')
        expected = {r['id'] for r in base if r['correct'] is True} - {'code_39', 'math_93'}
        assert ids == expected
        corrections = read_rows(directory/'correction_scores_final.jsonl')
        attacks = read_rows(directory/'attack_scores_reused.jsonl')
        assert {r['id'] for r in corrections} == ids
        assert len(attacks) == len(ids)*6
        assert len({(r['id'], r['branch']) for r in attacks}) == len(attacks)
        assert all(r['correct'] is not None for r in corrections)
        eligible.append(ids)
        models.append(metric)
        inputs[name] = {'attacks':attacks, 'corrections':corrections,
                       'hashes':{f:digest(directory/f) for f in ['metrics.json', 'base_scores_reused.jsonl',
                                                                'attack_scores_reused.jsonl', 'correction_scores_final.jsonl']}}
    common = set.intersection(*eligible)
    common_rows = []
    for metric in models:
        name = metric['run_name']
        records = inputs[name]
        corrected = counts([r for r in records['corrections'] if r['id'] in common])
        assert corrected['total'] == len(common)
        for persona in ['all']+PERSONAS:
            attacked = counts([r for r in records['attacks'] if r['id'] in common and (persona=='all' or r['branch']==persona)])
            assert attacked['total'] == len(common)*(6 if persona=='all' else 1)
            common_rows.append({'model':metric['model'], 'run_name':name, 'attack_type':persona,
                'attack_counts':attacked, 'correction_counts':corrected,
                'ESS':attacked['resolved_only'], 'AUR':corrected['resolved_only'],
                'BES':harmonic(attacked['resolved_only'], corrected['resolved_only']),
                'BES_label_bounds':[harmonic(attacked['bounds'][i], corrected['bounds'][i]) for i in [0,1]]})
    common_data = {'definition':'Intersection of confirmed initially correct anchors with available scored corrections across all six models; fixed labels and exact recorded responses.',
                   'anchor_count':len(common), 'ids':sorted(common),
                   'domains':dict(Counter(i.split('_')[0] for i in common)),
                   'source_hashes':{name:inputs[name]['hashes'] for name in NAMES},
                   'metrics':common_rows,
                   'limitations':'Selection on initial correctness; code tests finite; bounds concern unresolved labels, not sampling confidence. No generation or new adjudication.'}
    (HERE/'common_anchor_metrics.json').write_text(json.dumps(common_data,indent=2)+'\n',encoding='utf-8')
    macro = {}
    for domain in ['all', 'math', 'logic', 'coding']:
        rows = [next(r for r in m['metrics'] if r['domain']==domain and r['attack_type']=='all') for m in models]
        macro[domain] = {k:sum(r[k] for r in rows)/len(rows) for k in ['accuracy', 'ESS', 'AUR', 'BES', 'Overall']}
    macro_persona = {p:sum(next(r for r in m['metrics'] if r['domain']=='all' and r['attack_type']==p)['ESS'] for m in models)/len(models) for p in PERSONAS}
    paper_data = {'models':models, 'macro_averages':macro, 'macro_persona_ESS':macro_persona,
                  'common_anchor_file':'common_anchor_metrics.json'}
    (HERE/'paper_results_data.json').write_text(json.dumps(paper_data,indent=2)+'\n',encoding='utf-8')
    print('Common anchors:',len(common),common_data['domains'])
    for r in common_rows:
        if r['attack_type']=='all':
            print(r['model'],{k:round(100*r[k],2) for k in ['ESS','AUR','BES']},
                  'BES bounds',[round(100*x,2) for x in r['BES_label_bounds']],
                  'attack unknown',r['attack_counts']['unresolved'],'AUR counts',r['correction_counts']['correct'],'/',r['correction_counts']['total'])
    print('Domain macro averages:',{d:{k:round(100*v,2) for k,v in row.items()} for d,row in macro.items()})
    print('Common mean persona ESS:',{p:round(100*sum(r['ESS'] for r in common_rows if r['attack_type']==p)/len(models),2) for p in PERSONAS})


if __name__ == '__main__':
    main()
