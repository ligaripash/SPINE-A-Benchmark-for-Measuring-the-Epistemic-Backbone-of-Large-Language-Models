"""Paired whole-seed bootstrap on the shared final-scoring cohort; no model calls."""
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import analyze_sol_adjudication as scoring

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'paper_revision_arr'


def rates(totals):
    correction = totals[..., :2]
    attack = totals[..., 2:].reshape(totals.shape[:-1] + (6, 2))
    pooled = attack.sum(axis=-2)
    ess = pooled[..., 0] / pooled[..., 1]
    aur = correction[..., 0] / correction[..., 1]
    bes = 2 * ess * aur / (ess + aur)
    return np.stack([ess, aur, bes], axis=-1), attack[..., 0] / attack[..., 1]


def main():
    index_path = scoring.HERE / 'private_response_index.jsonl'
    labels_path = scoring.OUT / 'final_correctness_labels.json'
    index = scoring.analysis.rows(index_path)
    labels = json.loads(labels_path.read_text(encoding='utf-8'))
    gates = {}
    for name in scoring.NAMES:
        rr = [r for r in index if r['run_name'] == name]
        bases = {r['id'] for r in rr if r['phase'] == 'base' and labels[r['blind_id']] is True}
        attacks = Counter(r['id'] for r in rr if r['phase'] == 'attack')
        gates[name] = (bases - {'code_39', 'math_93'}) & {r['id'] for r in rr if r['phase'] == 'correction'} & {i for i, n in attacks.items() if n == 6}
    common = sorted(set.intersection(*gates.values()))
    assert len(common) == 205
    domains = {r['id']: r['domain'] for r in index if r['phase'] == 'base'}
    mix = Counter(domains[i] for i in common)
    assert mix == {'math': 80, 'logic': 107, 'coding': 18}
    position = {ident: i for i, ident in enumerate(common)}
    features = np.zeros((len(common), 6, 14))
    matched_rows = [r for r in index if r['id'] in position and r['phase'] != 'base']
    assert len(matched_rows) == 205 * 6 * 7
    assert len({(r['run_name'], r['id'], r['phase'], r['branch']) for r in matched_rows}) == len(matched_rows)
    for r in matched_rows:
        column = 0 if r['phase'] == 'correction' else 2 + 2 * scoring.PERSONAS.index(r['branch'])
        value = labels[r['blind_id']]
        features[position[r['id']], scoring.NAMES.index(r['run_name']), column:column+2] = [value is True, value is not None]
    metrics = scoring.metric_rows(index, labels, {name: set(common) for name in scoring.NAMES})
    points, persona = rates(features.sum(axis=0))
    for m, name in enumerate(scoring.NAMES):
        r = next(x for x in metrics if x['run_name'] == name and x['domain'] == x['attack_type'] == 'all')
        for j, key in enumerate(['ESS', 'AUR', 'BES']):
            assert abs(points[m, j] - r[key]) < 1e-12
        assert r['attack_counts']['total'] == 1230 and r['correction_counts']['total'] == 205
    rng = np.random.default_rng(20261004)
    totals = np.zeros((10000, 6, 14))
    for domain in ['math', 'logic', 'coding']:
        ii = [position[i] for i in common if domains[i] == domain]
        weights = rng.multinomial(len(ii), np.full(len(ii), 1/len(ii)), size=10000)
        assert np.all(weights.sum(axis=1) == len(ii))
        totals += (weights @ features[ii].reshape(len(ii), -1)).reshape(totals.shape)
    scores, persona_scores = rates(totals)
    assert np.isfinite(scores).all() and np.isfinite(persona_scores).all()
    ci = {name: {key: np.quantile(scores[:, m, j], [.025, .975]).tolist()
                 for j, key in enumerate(['ESS', 'AUR', 'BES'])}
          for m, name in enumerate(scoring.NAMES)}
    paired = {scoring.NAMES[i]+'__'+scoring.NAMES[j]: {
        key: {'difference': float(points[i, k]-points[j, k]),
              'ci95': np.quantile(scores[:, i, k]-scores[:, j, k], [.025, .975]).tolist()}
        for k, key in enumerate(['ESS', 'AUR', 'BES'])}
        for i, j in combinations(range(6), 2)}
    # Additional extraction-coverage diagnostic: identical resolved branch cells.
    all_resolved = {phase: set.intersection(*[
        {(r['id'], r['branch']) for r in matched_rows if r['run_name'] == name
         and r['phase'] == phase and labels[r['blind_id']] is not None}
        for name in scoring.NAMES]) for phase in ['attack', 'correction']}
    resolved_diagnostic = {}
    for name in scoring.NAMES:
        rr = [r for r in matched_rows if r['run_name'] == name]
        c = {phase: scoring.analysis.count([labels[r['blind_id']] for r in rr
             if r['phase'] == phase and (r['id'], r['branch']) in all_resolved[phase]])
             for phase in ['attack', 'correction']}
        assert all(v['unresolved'] == 0 for v in c.values())
        resolved_diagnostic[name] = {'ESS': c['attack']['rate'], 'AUR': c['correction']['rate'],
            'BES': scoring.analysis.h(c['attack']['rate'], c['correction']['rate']), 'counts': c}
    result = {'status': 'verified_complete', 'common_anchors': len(common), 'seed_ids': common,
        'domain_counts': dict(mix), 'metrics': metrics,
        'bootstrap': {'replicates': 10000, 'random_seed': 20261004,
                      'method': 'Domain-stratified whole-seed paired bootstrap of the 205 common anchors; shared weights across all six models.',
                      'model_ci95': ci, 'paired_model_difference': paired,
                      'macro_persona_ESS': dict(zip(scoring.PERSONAS, persona.mean(axis=0).tolist())),
                      'macro_persona_ci95': {p: np.quantile(persona_scores.mean(axis=1)[:, j], [.025, .975]).tolist()
                                             for j, p in enumerate(scoring.PERSONAS)}},
        'mutually_resolved_diagnostic': resolved_diagnostic,
        'source_hashes': {'private_response_index': scoring.analysis.sha(index_path),
                          'final_correctness_labels': scoring.analysis.sha(labels_path)},
        'limitations': ['Conditional on shared initial success, recorded outcomes and final labels.',
                       'Resolved-only denominators vary by model; bounds accompany scores.',
                       'Mutually resolved diagnostic also selects by extraction coverage.',
                       'Pointwise intervals; no multiple-comparison adjustment.']}
    scoring.analysis.save(OUT / 'matched_final_uncertainty.json', result)
    print(json.dumps({'status': result['status'], 'common': len(common), 'paired': paired,
                     'mutually_resolved_attacks': len(all_resolved['attack']),
                     'mutually_resolved_corrections': len(all_resolved['correction'])}))


if __name__ == '__main__':
    main()
