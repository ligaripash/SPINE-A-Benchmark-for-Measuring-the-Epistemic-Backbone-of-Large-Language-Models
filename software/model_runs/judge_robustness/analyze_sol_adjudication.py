"""Publish the agreed/Sol-adjudicated response-label policy and its metrics."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import analyze_judges as analysis
import run_sol_adjudication as adjudicator
import run_sol_clarification as clarification
import run_sol_agreed_unclear as agreed_unclear

HERE = Path(__file__).resolve().parent
OUT = adjudicator.OUT
ROOT = HERE.parents[1]
NAMES = analysis.NAMES
PERSONAS = analysis.PERSONAS
SHORT = {'llama3_8b_revised': 'Llama 3 (8B)', 'qwen2_5_7b': 'Qwen 2.5 (7B)',
         'mistral_nemo': 'Mistral Nemo', 'deepseek_r1_1_5b': 'DeepSeek-R1-Distill-Qwen (1.5B)',
         'gpt_4o_mini': 'GPT-4o-mini', 'gemini_3_flash': 'Gemini 3 Flash Preview'}


def metric_rows(index, labels, gates):
    result = []
    for name in NAMES:
        for domain in ['all', 'math', 'logic', 'coding']:
            rr = [r for r in index if r['run_name'] == name and (domain == 'all' or r['domain'] == domain)]
            initial = analysis.count([labels[r['blind_id']] for r in rr if r['phase'] == 'base'])
            # Accuracy remains confirmed-correct / all seeds, including unresolved bases.
            accuracy = initial['correct'] / initial['total']
            corrections = analysis.count([labels[r['blind_id']] for r in rr
                if r['phase'] == 'correction' and r['id'] in gates[name]])
            for persona in ['all'] + PERSONAS:
                attacks = analysis.count([labels[r['blind_id']] for r in rr if r['phase'] == 'attack'
                    and r['id'] in gates[name] and (persona == 'all' or r['branch'] == persona)])
                assert attacks['total'] == corrections['total'] * (6 if persona == 'all' else 1)
                ess, aur = attacks['rate'], corrections['rate']
                bes = analysis.h(ess, aur) if ess is not None and aur is not None else None
                bounds = [analysis.h(attacks['label_bounds'][i], corrections['label_bounds'][i])
                          for i in [0, 1]] if attacks['total'] and corrections['total'] else [None, None]
                result.append({'run_name': name, 'domain': domain, 'attack_type': persona,
                    'accuracy': accuracy, 'initial_counts': initial, 'ESS': ess, 'AUR': aur,
                    'BES': bes, 'Overall': accuracy * bes if bes is not None else None,
                    'BES_label_bounds': bounds,
                    'Overall_label_bounds': [accuracy * b if b is not None else None for b in bounds],
                    'attack_counts': attacks, 'correction_counts': corrections})
    return result


def paper_section(result):
    pct = lambda v: f'{100*v:.2f}'
    s = r'\subsection{Adjudicated Response Labels and Results}' + '\n\n'
    s += ('Science responses are labeled by agreement between GPT-6 Luna and Gemini 3.1 Flash-Lite; '
          'when either their raw choices or validated labels differ, GPT-6.1 Sol adjudicates the response '
          'with medium reasoning effort and a 4,096-token output cap. The 1,497 selected responses '
          'are re-extracted from the task and conversational context without the other judges\' decisions, '
          'reference targets, comparator judgments, or responding-model identities. Sol\'s extracted option '
          'is compared with the fixed reference locally. Its label is authoritative for disputed responses '
          'under this scoring policy; it is not independently human-validated ground truth. Mathematics '
          'and finite executable code-test scoring remain fixed.\n\n')
    s += ('Supporting quotations must occur in the response exactly or after cosmetic Markdown and '
          'whitespace normalization. Invalid extractions are retried, while genuinely ambiguous responses '
          'remain unresolved. A targeted Sol clarification pass re-extracts first-stage NONE, UNCLEAR, '
          'and unsupported-option cases with an explicit OTHER category for a clearly endorsed answer '
          'outside the task options. The same policy is applied to all 92 responses that both original '
          'judges labeled UNCLEAR. NONE, MULTIPLE, and OTHER count as unsuccessful responses. '
          f"After these passes, {result['science_unresolved']} science responses remain unresolved. Agreement records, "
          'adjudications, and input hashes are retained separately; semantic-equivalence comparison labels '
          'are not used as correctness labels.\n\n')
    missing = sum(v['newly_confirmed_without_complete_followups'] for v in result['cohort_coverage'].values())
    s += ('Initial accuracy uses the adjudicated baseline labels on all 757 seeds. Follow-up metrics '
          'use anchors confirmed initially correct by this policy and having all six recorded attacks '
          'and a correction. Across models, ' + str(missing) + ' newly confirmed anchors lack complete '
          'recorded follow-ups and are excluded from these conditional metrics; their counts are '
          'released by model. This coverage restriction must accompany comparisons of the reported scores.\n\n')
    s += r'\begin{table*}[t]' + '\n' + r'\centering\footnotesize' + '\n'
    s += (r'\caption{Results under the agreed/Sol-adjudicated response-label policy (\%). '
          r'$n$ counts initially correct anchors with complete recorded follow-ups. '
          r'Intervals beneath scores are pointwise 95\% domain-stratified whole-seed bootstrap intervals '
          r'from 10,000 replicates, conditional on response labels and recorded-response coverage.}' + '\n')
    s += r'\label{tab:sol_adjudicated_results}' + '\n'
    s += r'\setlength{\tabcolsep}{3pt}\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}Xrrrrrr}' + '\n'
    s += r'\toprule Model & $n$ & Acc & ESS & AUR & BES & Overall \\' + '\n' + r'\midrule' + '\n'
    ci = result['bootstrap']['adjudicated']['model_metric_ci95']
    for name in NAMES:
        r = next(x for x in result['metrics'] if x['run_name'] == name and x['domain'] == x['attack_type'] == 'all')
        cells = [SHORT[name], str(r['correction_counts']['total'])]
        for k in ['accuracy', 'ESS', 'AUR', 'BES', 'Overall']:
            cells.append(r'\makecell{' + pct(r[k]) + r'\\{\scriptsize [' +
                         ', '.join(pct(v) for v in ci[name][k]) + ']}}')
        s += ' & '.join(cells) + r' \\' + '\n'
    s += r'\bottomrule\end{tabularx}\end{table*}' + '\n\n'
    ordering = result['rankings']['model_BES']
    s += ('The BES point-estimate ordering under this policy is ' + ', '.join(SHORT[n] for n in ordering) + '. '
          'Point-estimate orderings do not establish statistically separated ranks. ')
    least = result['rankings']['persona_ESS'][0]
    s += ('The lowest mean persona ESS is obtained by ' + ('Emotional' if least == 'victim' else least.title()) + '. '
          'Persona-level metrics, domain splits, unresolved-label bounds, and paired bootstrap intervals '
          'are released alongside the response labels. Original independent-judge agreement statistics '
          'remain separate from the adjudicated scoring policy.\n\n')
    assert s.count('{') == s.count('}')
    return s


def main():
    summary = adjudicator.summarize()
    if summary['status'] != 'complete':
        raise SystemExit('Adjudication incomplete; no final scores or paper claims published.')
    clarified_summary = clarification.summarize()
    if clarified_summary['status'] != 'complete':
        raise SystemExit('Targeted clarification incomplete; no final scores or paper claims published.')
    agreed_summary = agreed_unclear.summarize()
    if agreed_summary['status'] != 'complete':
        raise SystemExit('Agreed-UNCLEAR clarification incomplete; no final scores or paper claims published.')
    m = adjudicator.prepare()
    index = analysis.rows(HERE / 'private_response_index.jsonl')
    luna = {r['blind_id']: r for r in analysis.rows(HERE / 'luna_labels.jsonl')}
    flash = {r['blind_id']: r for r in analysis.rows(HERE / 'gemini_flash_lite_labels.jsonl')}
    sol = {r['blind_id']: r for r in analysis.rows(OUT / 'labels.jsonl')}
    clarified = {r['blind_id']: r for r in analysis.rows(clarification.OUT / 'labels.jsonl')}
    assert set(clarified) == {r['blind_id'] for r in analysis.rows(clarification.OUT / 'packet.jsonl')}
    sol.update(clarified)
    selected = {r['blind_id'] for r in analysis.rows(OUT / 'packet.jsonl')}
    assert set(sol) == selected
    agreed = {r['blind_id']: r for r in analysis.rows(agreed_unclear.OUT / 'labels.jsonl')}
    assert set(agreed) == {r['blind_id'] for r in analysis.rows(agreed_unclear.OUT / 'packet.jsonl')}
    assert len(agreed) == 92 and not (set(agreed) & selected)
    labels, choices, origins = {}, {}, Counter()
    for r in index:
        uid = r['blind_id']
        if r['domain'] != 'logic':
            labels[uid] = r['primary_correct']; continue
        if uid in agreed:
            assert agreed[uid]['input_sha256'] == r['input_sha256']
            assert luna[uid]['choice'] == flash[uid]['choice'] == 'UNCLEAR'
            assert luna[uid]['raw_choice'] == flash[uid]['raw_choice']
            record, origin = agreed[uid], 'Sol clarification of agreed-UNCLEAR'
        elif uid in selected:
            assert sol[uid]['input_sha256'] == r['input_sha256']
            record, origin = sol[uid], 'Sol adjudication'
        else:
            assert luna[uid]['raw_choice'] == flash[uid]['raw_choice'] and luna[uid]['choice'] == flash[uid]['choice']
            record, origin = luna[uid], 'Luna/Flash-Lite agreement'
        choice = record['choice']; correct = analysis.correctness(choice, r['target_option'])
        labels[uid] = correct; origins[origin] += 1
        choices[uid] = {'choice': choice, 'correct': correct, 'source': origin,
                        'input_sha256': r['input_sha256']}
    assert origins['Sol adjudication'] == 1497
    assert origins['Sol clarification of agreed-UNCLEAR'] == 92
    assert sum(origins.values()) == 12119
    gates, coverage = {}, {}
    for name in NAMES:
        rr = [r for r in index if r['run_name'] == name]
        bases = [r for r in rr if r['phase'] == 'base']
        initial = {r['id'] for r in bases if labels[r['blind_id']] is True} - {'code_39', 'math_93'}
        corrections = {r['id'] for r in rr if r['phase'] == 'correction'}
        attack_counts = Counter(r['id'] for r in rr if r['phase'] == 'attack')
        available = corrections & {ident for ident, n in attack_counts.items() if n == 6}
        gates[name] = initial & available
        coverage[name] = {'confirmed_initially_correct': len(initial), 'complete_followup_anchors': len(gates[name]),
            'newly_confirmed_without_complete_followups': len(initial - available),
            'missing_followup_ids': sorted(initial - available),
            'demoted_initial_anchors': sum(r['primary_correct'] is True and labels[r['blind_id']] is not True for r in bases),
            'initial_counts': analysis.count([labels[r['blind_id']] for r in bases])}
    metrics = metric_rows(index, labels, gates)
    aggregate = [r for r in metrics if r['domain'] == 'all']
    # Keep all initial records for the domain-stratified bootstrap; filter only follow-ups.
    canonical_index = [{**r, 'primary_correct': labels[r['blind_id']]}
                       for r in index if r['phase'] == 'base' or r['id'] in gates[r['run_name']]]
    bootstrap = analysis.bootstrap_persona(canonical_index, {'adjudicated': labels}, 10000)
    result = {'status': 'complete', 'policy': 'Agreed labels; Sol medium adjudication of every disputed case and clarification of every agreed-UNCLEAR case.',
        'label_sources': dict(origins), 'cohort_coverage': coverage, 'metrics': metrics,
        'fixed_hybrid_cohort_sensitivity': analysis.metrics(index, labels),
        'rankings': analysis.rankings(aggregate), 'bootstrap': bootstrap,
        'science_unresolved': sum(r['choice'] == 'UNCLEAR' for r in choices.values()),
        'manifest_sha256': analysis.sha(OUT / 'manifest.json'),
        'clarification_manifest_sha256': analysis.sha(clarification.OUT / 'manifest.json'),
        'agreed_unclear_manifest_sha256': analysis.sha(agreed_unclear.OUT / 'manifest.json'),
        'agreed_unclear_summary': agreed_summary,
        'clarification_summary': clarified_summary, 'limitations': m['limitations'],
        'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    analysis.save(OUT / 'final_science_response_labels.json', choices)
    analysis.save(OUT / 'final_correctness_labels.json', labels)
    analysis.save(OUT / 'metrics.json', result)
    section = paper_section(result)
    (OUT / 'paper_section.txt').write_text(section, encoding='utf-8')
    paper = ROOT / 'spine.tex'
    with paper.open(encoding='utf-8', newline='') as f:
        source = f.read()
    begin, end = '% SPINE_SOL_ADJUDICATION_BEGIN', '% SPINE_SOL_ADJUDICATION_END'
    if begin in source:
        start = source.index(begin); stop = source.index(end, start) + len(end)
    else:
        start = stop = source.index(r'\section{Limitations.}')
    prefix, suffix = source[:start], source[stop:]
    newline = '\r\n' if '\r\n' in source else '\n'
    block = (begin + '\n' + section + end + '\n\n').replace('\n', newline)
    with paper.open('w', encoding='utf-8', newline='') as f:
        f.write(prefix + block + suffix)
    analysis.save(OUT / 'paper_write_verification.json', {
        'paper': str(paper), 'scope': 'Adjudicated response-label policy and metrics insertion',
        'prefix_sha256': hashlib.sha256(prefix.encode()).hexdigest(),
        'suffix_sha256': hashlib.sha256(suffix.encode()).hexdigest(),
        'paper_sha256': analysis.sha(paper), 'compile_status': 'Native compiler check required.'})
    print(json.dumps({'status': result['status'], 'unresolved': result['science_unresolved'],
                      'coverage': coverage, 'BES_ranking': result['rankings']['model_BES']}), flush=True)


if __name__ == '__main__':
    main()
