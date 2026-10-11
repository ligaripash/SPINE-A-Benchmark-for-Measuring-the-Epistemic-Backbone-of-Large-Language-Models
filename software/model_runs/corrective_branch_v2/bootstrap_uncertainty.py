"""Seed-cluster bootstrap of fixed SPINE scores; no model calls or relabeling."""
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from compute_paper_results import NAMES, PERSONAS, read_rows

HERE = Path(__file__).resolve().parent
B = 10000
RNG_SEED = 20261004
KEYS = ['accuracy', 'ESS', 'AUR', 'BES', 'Overall']
DOMAINS = ['math', 'logic', 'coding']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rates(features, persona='all'):
    """Last axis: initial correct/total, correction correct/resolved, attack C/R pairs."""
    acc = features[..., 0]/features[..., 1]
    aur = features[..., 2]/features[..., 3]
    attacks = features[..., 4:].reshape(features.shape[:-1]+(6, 2))
    a = attacks.sum(axis=-2) if persona == 'all' else attacks[..., PERSONAS.index(persona), :]
    ess = a[..., 0]/a[..., 1]
    bes = np.divide(2*ess*aur, ess+aur, out=np.zeros_like(ess), where=ess+aur != 0)
    result = np.stack([acc, ess, aur, bes, acc*bes], axis=-1)
    assert np.isfinite(result).all(), 'Zero denominator in a bootstrap replicate'
    return result


def summary(point, draws):
    assert draws.shape[0] == B
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    return {k: {'estimate': float(point[i]), 'ci95': [float(lo[i]), float(hi[i])]}
            for i, k in enumerate(KEYS)}


def main():
    paper = json.loads((HERE/'paper_results_data.json').read_text(encoding='utf-8'))
    common = json.loads((HERE/'common_anchor_metrics.json').read_text(encoding='utf-8'))
    bases = [read_rows(HERE/name/'base_scores_reused.jsonl') for name in NAMES]
    ids = sorted(r['id'] for r in bases[0])
    index = {id_: i for i, id_ in enumerate(ids)}
    domain = {r['id']: r['domain'] for r in bases[0]}
    assert {d: sum(domain[i]==d for i in ids) for d in DOMAINS} == {'math':294, 'logic':299, 'coding':164}
    # Keep all responses from a seed in one feature row, including eligibility.
    features = np.zeros((len(ids), len(NAMES), 16), dtype=np.float64)
    hashes = {}
    for m, name in enumerate(NAMES):
        assert set(r['id'] for r in bases[m]) == set(ids)
        assert len(bases[m]) == len(ids) == 757
        for r in bases[m]:
            assert domain[r['id']] == r['domain']
            features[index[r['id']], m, :2] = [r['correct'] is True, 1]
        corrections = read_rows(HERE/name/'correction_scores_final.jsonl')
        attacks = read_rows(HERE/name/'attack_scores_reused.jsonl')
        eligible = set(json.loads((HERE/name/'correction_eligible_ids.json').read_text(encoding='utf-8')))
        assert {r['id'] for r in corrections} == eligible
        assert len(corrections) == len(eligible)
        assert len(attacks) == 6*len(eligible)
        assert len({(r['id'], r['branch']) for r in attacks}) == len(attacks)
        assert {r['id'] for r in attacks} == eligible
        for r in corrections:
            assert r['correct'] is not None
            features[index[r['id']], m, 2:4] = [r['correct'] is True, 1]
        for r in attacks:
            col = 4+2*PERSONAS.index(r['branch'])
            features[index[r['id']], m, col:col+2] = [r['correct'] is True, r['correct'] is not None]
        hashes[name] = {f: digest(HERE/name/f) for f in common['source_hashes'][name]}
        assert hashes[name] == common['source_hashes'][name]

    rng = np.random.default_rng(RNG_SEED)
    arrays, records, macro, pairs = {}, [], [], []
    def run(cohort, selected):
        sums, draws_by_domain = {}, {}
        for d in DOMAINS:
            ix = [index[id_] for id_ in selected if domain[id_] == d]
            f = features[ix]
            # Multinomial counts exactly represent n independent draws with replacement.
            weights = rng.multinomial(len(ix), np.full(len(ix), 1/len(ix)), size=B)
            assert np.all(weights.sum(axis=1)==len(ix))
            draws_by_domain[d] = (weights @ f.reshape(len(ix), -1)).reshape(B, len(NAMES), 16)
            sums[d] = f.sum(axis=0)
        draws_by_domain['all'] = sum(draws_by_domain.values())
        sums['all'] = sum(sums.values())
        for d in ['all']+DOMAINS:
            for p in ['all']+PERSONAS:
                point = rates(sums[d], p)
                draws = rates(draws_by_domain[d], p)
                arrays[f'{cohort}__{d}__{p}'] = draws
                for m, name in enumerate(NAMES):
                    if cohort == 'full':
                        expected = next(r for r in paper['models'][m]['metrics'] if r['domain']==d and r['attack_type']==p)
                        for k, v in zip(KEYS, point[m]):
                            assert abs(expected[k]-v) < 1e-12, (name, d, p, k)
                    elif d == 'all':
                        expected = next(r for r in common['metrics'] if r['run_name']==name and r['attack_type']==p)
                        for k in ['ESS','AUR','BES']:
                            assert abs(expected[k]-point[m, KEYS.index(k)]) < 1e-12
                    records.append(dict(cohort=cohort, domain=d, attack_type=p, run_name=name,
                                        metrics=summary(point[m], draws[:, m])))
                macro.append(dict(cohort=cohort, domain=d, attack_type=p,
                                  metrics=summary(point.mean(axis=0), draws.mean(axis=1))))
                if d == 'all' and p == 'all':
                    for i, j in combinations(range(len(NAMES)), 2):
                        pairs.append(dict(cohort=cohort, left=NAMES[i], right=NAMES[j],
                                          metrics=summary(point[i]-point[j], draws[:, i]-draws[:, j])))
        return sums, draws_by_domain

    run('full', ids)
    run('common', common['ids'])
    persona_diffs = []
    for cohort in ['full', 'common']:
        for p in PERSONAS:
            if p == 'bureaucrat':
                continue
            a = arrays[f'{cohort}__all__bureaucrat'][:, :, 1].mean(axis=1)
            b = arrays[f'{cohort}__all__{p}'][:, :, 1].mean(axis=1)
            def macro_point(persona):
                return next(r for r in macro if r['cohort']==cohort and r['domain']=='all' and r['attack_type']==persona)['metrics']['ESS']['estimate']
            persona_diffs.append(dict(cohort=cohort, left='bureaucrat', right=p,
                estimate=macro_point('bureaucrat')-macro_point(p),
                ci95=np.quantile(a-b, [0.025, 0.975]).tolist()))
    out = dict(method='Domain-stratified whole-seed cluster bootstrap; paired seed draws across six fixed models',
        replicates=B, rng_seed=RNG_SEED, rng_bit_generator='PCG64', numpy_version=np.__version__,
        interval='Pointwise percentile 95%: empirical 2.5th/97.5th percentiles (NumPy linear interpolation)',
        metric_order=KEYS, model_order=NAMES, full_domain_counts={d:sum(domain[i]==d for i in ids) for d in DOMAINS},
        common_domain_counts={d:sum(domain[i]==d for i in common['ids']) for d in DOMAINS},
        source_hashes=hashes, paper_results_sha256=digest(HERE/'paper_results_data.json'),
        bootstrap_script_sha256=digest(Path(__file__)),
        common_anchor_sha256=digest(HERE/'common_anchor_metrics.json'),
        assumptions='Seeds exchangeable and independent within domains; fixed domain mix and model roster. Correctness labels and per-seed eligibility remain fixed. ESS excludes unresolved labels; initial accuracy counts only confirmed correct over all seeds.',
        limitations='Conditional on recorded outputs and scoring, not generation, label, intervention-validity or finite-test uncertainty. Pointwise exploratory difference intervals have no multiple-comparison adjustment. Common cohort conditions on observed initial correctness in all models.',
        records=records, macro_averages=macro, paired_model_differences=pairs, paired_persona_differences=persona_diffs)
    np.savez_compressed(HERE/'bootstrap_draws.npz', **arrays)
    out['draws_sha256'] = digest(HERE/'bootstrap_draws.npz')
    (HERE/'bootstrap_uncertainty.json').write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
    for r in records:
        if r['cohort']=='full' and r['domain']=='all' and r['attack_type']=='all':
            print(r['run_name'], {k:[round(100*v['estimate'],2), [round(100*x,2) for x in v['ci95']]] for k,v in r['metrics'].items()})
    for r in pairs:
        if (r['left'], r['right']) in [('qwen2_5_7b','deepseek_r1_1_5b'), ('gpt_4o_mini','gemini_3_flash')]:
            print('Paired BES', r['cohort'], r['left'], '-', r['right'], r['metrics']['BES'])
    print('Persona differences:', persona_diffs)


if __name__ == '__main__':
    main()
