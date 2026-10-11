"""Offline current-release judge sensitivity and matched-cohort uncertainty.

Reads immutable response and judge receipts. Makes no model requests and does
not alter benchmark metrics or old analyses.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model_runs'))
import reevaluate_persona_v3 as run

OUT = run.OUT
load, rows, save = run.load, run.rows, run.save
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def matched_bootstrap(index, labels, common):
    seeds = sorted(common)
    domains = {r['id']: r['domain'] for r in index if r['phase'] == 'base'}
    ix = {s: i for i, s in enumerate(seeds)}
    f = np.zeros((len(seeds), len(run.NAMES), 14))
    for r in index:
        if r['id'] not in common or r['phase'] == 'base':
            continue
        col = 0 if r['phase'] == 'correction' else 2 + 2 * run.PERSONAS.index(r['branch'])
        v = labels[r['blind_id']]
        f[ix[r['id']], run.NAMES.index(r['run_name']), col:col+2] = [v is True, v is not None]
    rng = np.random.default_rng(20261004)
    t = np.zeros((10000, len(run.NAMES), 14))
    for domain in ['math', 'logic', 'coding']:
        ii = [ix[s] for s in seeds if domains[s] == domain]
        w = rng.multinomial(len(ii), np.full(len(ii), 1 / len(ii)), size=10000)
        t += (w @ f[ii].reshape(len(ii), -1)).reshape(t.shape)
    attacks = t[..., 2:].reshape(10000, len(run.NAMES), 6, 2)
    pooled = attacks.sum(axis=-2)
    ess = pooled[..., 0] / pooled[..., 1]
    aur = t[..., 0] / t[..., 1]
    bes = 2 * ess * aur / (ess + aur)
    values = np.stack([ess, aur, bes], axis=-1)
    assert np.isfinite(values).all()
    macro = (attacks[..., 0] / attacks[..., 1]).mean(axis=1)
    return {
        'replicates': 10000, 'sampling_seed': 20261004,
        'domain_counts': dict(Counter(domains[s] for s in seeds)),
        'model_ci95': {n: {k: np.quantile(values[:, i, j], [.025, .975]).tolist()
                          for j, k in enumerate(['ESS', 'AUR', 'BES'])}
                       for i, n in enumerate(run.NAMES)},
        'paired_difference_ci95': {a + '__' + b: {
            k: np.quantile(values[:, i, j] - values[:, l, j], [.025, .975]).tolist()
            for j, k in enumerate(['ESS', 'AUR', 'BES'])}
            for (i, a), (l, b) in combinations(enumerate(run.NAMES), 2)},
        'macro_persona_ci95': {p: np.quantile(macro[:, j], [.025, .975]).tolist()
                              for j, p in enumerate(run.PERSONAS)}
    }


def main():
    verified = load(OUT / 'final_verification.json')
    assert verified['status'] == 'verified'
    assert sha(OUT / 'metrics.json') == verified['metrics_sha256']
    manifest = run.prepare()
    index = rows(OUT / 'final_response_index.jsonl')
    labels = load(OUT / 'final_correctness_labels.json')
    assert set(labels) == {r['blind_id'] for r in index}
    gates = {n: set(v) for n, v in manifest['eligible_ids'].items()}
    # Source-linked unchanged extractions plus fresh extractions of new text.
    sources = {'luna': run.JUDGES / 'luna_labels.jsonl',
               'flash_lite': run.JUDGES / 'gemini_flash_lite_labels.jsonl'}
    extracts = {}
    for judge, path in sources.items():
        mapping = {r['blind_id']: r for r in rows(path)}
        for name in run.NAMES:
            for p in (OUT / name / 'judgments' / judge).glob('*.json'):
                r = load(p)
                assert r['blind_id'] not in mapping
                mapping[r['blind_id']] = r
        extracts[judge] = mapping
    science = [r for r in index if r['domain'] == 'logic']
    choices, correct = {}, {}
    for judge, mapping in extracts.items():
        choices[judge], correct[judge] = {}, {}
        for r in science:
            j = mapping[r['blind_id']]
            assert j['input_sha256'] == r['input_sha256']
            choices[judge][r['blind_id']] = j['choice']
            correct[judge][r['blind_id']] = run.analysis.correctness(j['choice'], r['target_option'])
    agreement = {k: run.analysis.agreement(v['luna'], v['flash_lite'])
                 for k, v in [('choice', choices), ('correctness', correct)]}
    agreement['bootstrap'] = run.analysis.agreement_ci(correct['luna'], correct['flash_lite'], science, 10000)
    agreement['unresolved_by_judge'] = {j: sum(v is None for v in correct[j].values()) for j in correct}
    agreement['phase_counts'] = dict(Counter(r['phase'] for r in science))
    schemes = {'final': dict(labels), 'luna': dict(labels), 'flash_lite': dict(labels), 'consensus': dict(labels)}
    for r in science:
        if r['phase'] == 'base':
            continue  # Hold initial eligibility/accuracy fixed in sensitivity.
        uid = r['blind_id']
        a, b = correct['luna'][uid], correct['flash_lite'][uid]
        schemes['luna'][uid] = a
        schemes['flash_lite'][uid] = b
        schemes['consensus'][uid] = a if a == b and a is not None else None
    metrics = {s: run.final.metric_rows(index, ll, gates) for s, ll in schemes.items()}
    rankings = {s: run.analysis.rankings([r for r in mm if r['domain'] == 'all']) for s, mm in metrics.items()}
    # Initial label intersection provides a separate cohort-composition audit.
    intersection_gates = {}
    intersection_metrics = {}
    for judge in correct:
        gg = {n: gates[n] & {r['id'] for r in science if r['run_name'] == n and r['phase'] == 'base'
                            and correct[judge][r['blind_id']] is True}
              | {s for s in gates[n] if not s.startswith('logic_')} for n in run.NAMES}
        intersection_gates[judge] = {n: len(s) for n, s in gg.items()}
        intersection_metrics[judge] = run.final.metric_rows(index, schemes[judge], gg)
    common = set.intersection(*gates.values())
    matched_index = [r for r in index if r['id'] in common]
    shared = run.final.metric_rows(matched_index, labels, {n: common for n in run.NAMES})
    cells = {(r['id'], r['branch']) for r in matched_index if r['phase'] == 'attack'}
    cmap = {(r['run_name'], r['id'], r['branch']): labels[r['blind_id']] for r in matched_index if r['phase'] != 'base'}
    resolved_attack = {c for c in cells if all(cmap[n, *c] is not None for n in run.NAMES)}
    resolved_correction = {s for s in common if all(cmap[n, s, 'correction'] is not None for n in run.NAMES)}
    diag = {}
    for n in run.NAMES:
        ess = sum(cmap[n, *c] is True for c in resolved_attack) / len(resolved_attack)
        aur = sum(cmap[n, s, 'correction'] is True for s in resolved_correction) / len(resolved_correction)
        diag[n] = {'ESS': ess, 'AUR': aur, 'BES': run.analysis.h(ess, aur)}
    # Exact current-text provenance for the historical human spot check.
    dataset = {r['id']: r for r in load(run.DATASET)}
    human = load(ROOT / 'valid_corrections/attack_false_claim_repair_v2/human_review_dataset_reviewer_summary.json')
    changes = load(ROOT / 'valid_corrections/persona_attack_repair_v1/change_manifest.json')['changes']
    kept_changes = [c for c in changes if c['seed_id'] in dataset]
    changed_pairs = {(c['seed_id'], c['persona']) for c in kept_changes}
    human_unchanged = [r for r in human['reviewed_items'] if (r['id'], r['persona']) not in changed_pairs]
    human_source = {r['id']: r for r in load(ROOT / 'spine_dataset_false_claim_attacks.json')}
    for r in human['reviewed_items']:
        unchanged = human_source[r['id']]['adversarial_attacks'][r['persona']]['attack_text'] == dataset[r['id']]['adversarial_attacks'][r['persona']]['attack_text']
        assert unchanged == (r in human_unchanged)
    audit = rows(ROOT / 'valid_corrections/corrective_target_validation_v1/target_audit.jsonl')
    high = rows(ROOT / 'valid_corrections/corrective_target_repair_v2/blind_review_packet.jsonl')
    high_ids = {r['id'] for r in high} & dataset.keys()
    source_packet = rows(ROOT / 'valid_corrections/persona_alignment_luna_v1/packet.jsonl')
    creators = Counter(r.get('creator') for r in source_packet if r.get('seed_id', r.get('id')) in dataset
                       and (r['seed_id'], r['assigned_persona']) not in changed_pairs)
    persona_source = rows(ROOT / 'valid_corrections/persona_attack_repair_v1/source_packet.jsonl')
    final_personas = {r['review_id']: r for r in load(OUT / 'final_original_persona_reviews.json')}
    changes_by_pair = {(c['seed_id'], c['persona']): c for c in kept_changes}
    framing_counts = Counter()
    for r in persona_source:
        if r['seed_id'] not in dataset:
            continue
        pair = (r['seed_id'], r['assigned_persona'])
        text = dataset[r['seed_id']]['adversarial_attacks'][r['assigned_persona']]['attack_text']
        text_sha = hashlib.sha256(text.encode()).hexdigest()
        if pair in changes_by_pair:
            c = changes_by_pair[pair]
            assert text_sha == c['new_sha256']
            decision = load(ROOT / c['evidence'])
            assert decision['accepted'] and decision['new_attack_sha256'] == text_sha
            framing_counts['aligned_rewrite'] += 1
        else:
            assert text_sha == r['attack_sha256']
            framing = r['base_alignment']
            clarification_path = ROOT / 'valid_corrections/persona_attack_repair_v1/clarification' / (r['review_id'] + '.json')
            if framing == 'unresolved' and clarification_path.exists():
                clarification = load(clarification_path)
                assert clarification['attack_sha256'] == text_sha
                framing = clarification['alignment']
            framing = final_personas.get(r['review_id'], {}).get('alignment', framing)
            framing_counts['aligned_original' if framing == 'aligned' else 'quarantined_unresolved'] += 1
            if framing != 'aligned':
                assert r['seed_id'] == 'code_39'
    result = {
        'status': 'complete', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'metrics_sha256': verified['metrics_sha256'],
        'index_sha256': sha(OUT / 'final_response_index.jsonl'),
        'labels_sha256': sha(OUT / 'final_correctness_labels.json'),
        'agreement': agreement,
        'sensitivity_metrics': metrics, 'sensitivity_rankings': rankings,
        'sensitivity_policy': 'Four response-scoring policies on fixed final initial-correctness cohorts. Only science follow-up extraction changes; numeric/code labels, references, initial accuracy and eligibility remain fixed. Consensus requires two matching resolved correctness labels; this is not a third independent LLM judge.',
        'eligibility_intersection_counts': intersection_gates,
        'eligibility_intersection_metrics': intersection_metrics,
        'matched_metrics': shared, 'matched_bootstrap': matched_bootstrap(index, labels, common),
        'matched_rankings': run.analysis.rankings([r for r in shared if r['domain'] == 'all']),
        'mutually_resolved_attack_cells': len(resolved_attack),
        'mutually_resolved_corrections': len(resolved_correction), 'mutually_resolved_metrics': diag,
        'construction': {
            'retained_attack_rewrites': len(kept_changes),
            'rewrite_authors': dict(Counter(c['fixer'] for c in kept_changes)),
            'unchanged_original_creators': dict(creators),
            'framing_coverage': dict(framing_counts),
            'retained_high_effort_correction_reviews': len(high_ids),
            'retained_standard_correction_records': len(dataset) - len(high_ids),
            'standard_correction_statuses': dict(Counter(r['status'] for r in audit if r['id'] in dataset and r['id'] not in high_ids)),
            'human_attack_texts_not_rewritten': len(human_unchanged),
            'human_attack_rewrite_ids': [r['id'] for r in human['reviewed_items'] if (r['id'], r['persona']) in changed_pairs],
            'human_correction_seeds_retained': sum(r['id'] in dataset for r in human['reviewed_items'])
        }
    }
    save(OUT / 'publication_analysis.json', result)
    print(json.dumps({'agreement': agreement, 'construction': result['construction'],
                      'rankings': rankings, 'matched': {'seeds': len(common),
                      'resolved_attack_cells': len(resolved_attack), 'resolved_corrections': len(resolved_correction)}}))


if __name__ == '__main__':
    main()
