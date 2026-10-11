"""Coverage-changing diagnostic, not a substitute for independent judge agreement.

Hold primary seed eligibility and validated targets fixed; omit response labels
whose method records Qwen extraction or assistant/manual adjudication. The
remaining subset is not representative, and initial eligibility still depends
on primary scoring. Never modify labels or send model requests.
"""
import hashlib
import json
from pathlib import Path
from compute_paper_results import NAMES, PERSONAS, read_rows

HERE = Path(__file__).resolve().parent


def keep(row):
    return not any(t in row.get('method', '').lower()
                   for t in ['qwen', 'adjudicat', 'assistant', 'manual'])


def stats(rows):
    resolved = [r for r in rows if r['correct'] is not None]
    correct = sum(r['correct'] is True for r in resolved)
    return {'correct':correct, 'resolved':len(resolved), 'retained_rows':len(rows),
            'rate':correct/len(resolved)}


def main():
    records, hashes = [], {}
    for name in NAMES:
        path = HERE/name
        metric = json.loads((path/'metrics.json').read_text(encoding='utf-8'))
        primary = next(r for r in metric['metrics'] if r['domain']=='all' and r['attack_type']=='all')
        attacks = read_rows(path/'attack_scores_reused.jsonl')
        corrections = read_rows(path/'correction_scores_final.jsonl')
        selected = [r for r in attacks if keep(r)]
        a,c = stats(selected), stats([r for r in corrections if keep(r)])
        bes = 2*a['rate']*c['rate']/(a['rate']+c['rate'])
        records.append({'run_name':name, 'model':metric['model'],
            'primary_ESS':primary['ESS'], 'primary_AUR':primary['AUR'], 'primary_BES':primary['BES'],
            'diagnostic_ESS':a['rate'], 'diagnostic_AUR':c['rate'], 'diagnostic_BES':bes,
            'attack_counts':a, 'correction_counts':c,
            'primary_attack_total':len(attacks), 'primary_correction_total':len(corrections),
            'persona_ESS':{p:stats([r for r in selected if r['branch']==p])['rate'] for p in PERSONAS}})
        hashes[name] = {f:hashlib.sha256((path/f).read_bytes()).hexdigest()
                       for f in ['metrics.json','attack_scores_reused.jsonl','correction_scores_final.jsonl']}
    macro = {p:sum(r['persona_ESS'][p] for r in records)/len(records) for p in PERSONAS}
    out = {'analysis':'Omit assisted response labels, resolved-only rates, primary cohorts fixed',
        'exclusion_rule':'Recorded method contains qwen, adjudicat, assistant, or manual (case insensitive)',
        'limitations':['This is not alternative LLM judging, an ensemble, or leave-one-judge-out analysis.',
            'Scoring coverage and task composition change; selection can alter scores even without judge errors.',
            'Primary initial eligibility and validated corrective targets are fixed and are not independently revalidated.',
            'Finite code tests, deterministic extraction limitations and unresolved labels remain.',
            'No model-response human agreement estimate is available from the eight-item dataset-validity review.'],
        'records':records, 'macro_persona_ESS':macro,
        'primary_BES_order':[r['run_name'] for r in sorted(records,key=lambda r:r['primary_BES'],reverse=True)],
        'diagnostic_BES_order':[r['run_name'] for r in sorted(records,key=lambda r:r['diagnostic_BES'],reverse=True)],
        'source_hashes':hashes,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE/'judge_dependence_diagnostic.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:out[k] for k in ['primary_BES_order','diagnostic_BES_order','macro_persona_ESS']},indent=2))


if __name__ == '__main__':
    main()
