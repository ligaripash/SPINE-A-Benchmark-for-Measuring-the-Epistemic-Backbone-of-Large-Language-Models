"""Offline wording diagnostic and executable-claim sensitivity, without new calls.

Normative cues are observable lexical flags, not exhaustive semantic labels.
The factual-content diagnostic selects exact-text false-output witnesses only;
counterexamples to a proposed replacement implementation are not sufficient.
No benchmark source, response, or score is changed.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'model_runs/persona_revision_v3'
OUT = RUN / 'attack_wording_audit'
PERSONAS = ['sophistry', 'authority', 'consensus', 'victim', 'doubter', 'bureaucrat']
NAMES = ['llama3_8b_revised', 'qwen2_5_7b', 'mistral_nemo',
         'deepseek_r1_1_5b', 'gpt_4o_mini', 'gemini_3_flash']
CUES = {
    'policy_or_procedure': r'\b(?:polic(?:y|ies)|protocols?|rubrics?|complianc\w*|compliant|non[- ]compliant|procedurally|mandatory|mandated|standards?)\b',
    'requirement_or_directive': r'\b(?:new requirements?|required (?:answer|result|output)|must (?:be|use|return|select|choose)|shall (?:be|return)|should (?:be|return))\b',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def counts(records, labels):
    vals = [labels[r['blind_id']] for r in records]
    correct = sum(v is True for v in vals)
    resolved = sum(v is not None for v in vals)
    return {'total': len(vals), 'correct': correct, 'resolved': resolved,
            'unresolved': len(vals) - resolved,
            'ESS': correct / resolved if resolved else None}


def main():
    verified = read(RUN / 'final_verification.json')
    assert sha(RUN / 'metrics.json') == verified['metrics_sha256']
    dataset_path = ROOT / 'spine_dataset_persona_validated_v3.json'
    assert sha(dataset_path) == verified['dataset_sha256']
    dataset = read(dataset_path)
    tasks = {t['id']: t for t in dataset}
    index = rows(RUN / 'final_response_index.jsonl')
    labels = read(RUN / 'final_correctness_labels.json')
    assert set(labels) == {r['blind_id'] for r in index}
    witnesses = {}
    old_path = ROOT / 'valid_corrections/attack_false_claim_repair_v2/code_attack_machine_audit.json'
    for r in read(old_path)['records']:
        key = r['id'], r['persona']
        if r['id'] not in tasks or r['id'] == 'code_39':
            continue
        text = tasks[r['id']]['adversarial_attacks'][r['persona']]['attack_text']
        if text_sha(text) == r['attack_sha256'] and r['check'].get('claim_is_false'):
            witnesses[key] = {'source': str(old_path.relative_to(ROOT)),
                              'source_sha256': sha(old_path), 'check': r['check']}
    manifest_path = ROOT / 'valid_corrections/persona_attack_repair_v1/change_manifest.json'
    for c in read(manifest_path)['changes']:
        sid, persona = c['seed_id'], c['persona']
        if sid not in tasks or not sid.startswith('code_') or sid == 'code_39':
            continue
        text = tasks[sid]['adversarial_attacks'][persona]['attack_text']
        if text_sha(text) != c['new_sha256']:
            continue
        path = ROOT / c['evidence']
        receipt = read(path)
        check = receipt.get('code_validation', {})
        if receipt.get('accepted') and check.get('accepted') and check.get('claim_is_false') and check.get('visible_test_independently_matches'):
            witnesses[sid, persona] = {'source': str(path.relative_to(ROOT)),
                                      'source_sha256': sha(path), 'check': check}
    tags = {}
    output = []
    for t in dataset:
        for persona in PERSONAS:
            text = t['adversarial_attacks'][persona]['attack_text']
            cue_hits = [{'rule': name, 'match': m.group(), 'span': list(m.span())}
                        for name, pattern in CUES.items()
                        for m in re.finditer(pattern, text, re.I)]
            key = t['id'], persona
            rec = {'id': t['id'], 'domain': t['domain'], 'persona': persona,
                   'attack_sha256': text_sha(text), 'normative_cue': bool(cue_hits),
                   'cue_hits': cue_hits, 'executable_false_output_claim': key in witnesses,
                   'quarantined': t['id'] == 'code_39',
                   'witness': witnesses.get(key)}
            tags[key] = rec
            output.append(rec)
    all_counts = Counter(('cue' if r['normative_cue'] else 'no_cue',
                          'witness' if r['executable_false_output_claim'] else 'no_selected_witness')
                         for r in output)
    attacks = [r for r in index if r['phase'] == 'attack']
    for r in attacks:
        t = tags[r['id'], r['branch']]
        assert not t['quarantined']
        assert text_sha(r['judge_input']['followup_message']) == t['attack_sha256']
    model_stats = {}
    for name in NAMES:
        rr = [r for r in attacks if r['run_name'] == name]
        model_stats[name] = {
            'cue': counts([r for r in rr if tags[r['id'], r['branch']]['normative_cue']], labels),
            'no_cue': counts([r for r in rr if not tags[r['id'], r['branch']]['normative_cue']], labels),
            'executable_claim': counts([r for r in rr if (r['id'], r['branch']) in witnesses], labels),
            'executable_claim_personas': {p: counts([r for r in rr if r['branch'] == p and (r['id'], p) in witnesses], labels) for p in PERSONAS}}
    coding_seeds = sorted(t['id'] for t in dataset if t['domain'] == 'coding')
    seed_ix = {s: i for i, s in enumerate(coding_seeds)}
    f = np.zeros((len(coding_seeds), len(NAMES), len(PERSONAS), 2))
    for r in attacks:
        if (r['id'], r['branch']) not in witnesses:
            continue
        v = labels[r['blind_id']]
        f[seed_ix[r['id']], NAMES.index(r['run_name']), PERSONAS.index(r['branch'])] += [v is True, v is not None]
    rng = np.random.default_rng(20261007)
    w = rng.multinomial(len(coding_seeds), np.full(len(coding_seeds), 1 / len(coding_seeds)), size=10000)
    totals = (w @ f.reshape(len(coding_seeds), -1)).reshape(10000, len(NAMES), len(PERSONAS), 2)
    rates = np.divide(totals[..., 0], totals[..., 1], out=np.full(totals.shape[:-1], np.nan), where=totals[..., 1] > 0)
    macro = rates.mean(axis=1)
    diff = macro[:, PERSONAS.index('bureaucrat')] - macro[:, PERSONAS.index('authority')]
    assert np.isfinite(diff).all()
    persona_macro = {p: float(np.mean([model_stats[n]['executable_claim_personas'][p]['ESS'] for n in NAMES])) for p in PERSONAS}
    kind_counts = Counter()
    correction_rows = []
    for t in dataset:
        if t['domain'] != 'logic':
            continue
        kind = t['valid_corrective_intervention'].get('kind', '').lower().replace('_', ' ')
        group = ('complete revised question' if kind == 'complete revised question' else
                 'measurement/question target, including mixed edits' if any(k in kind for k in ['measurement', 'question specification', 'investigation goal']) else
                 'referent, including mixed edits' if 'referent' in kind else
                 'premise/observation/other')
        kind_counts[group] += 1
        correction_rows.append({'id': t['id'], 'kind': kind, 'group': group})
    summary = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
               'dataset_sha256': sha(dataset_path), 'metrics_sha256': sha(RUN / 'metrics.json'),
               'index_sha256': sha(RUN / 'final_response_index.jsonl'), 'labels_sha256': sha(RUN / 'final_correctness_labels.json'),
               'cue_rules': CUES, 'released_attacks': len(output),
               'cross_counts': {a + '__' + b: v for (a, b), v in all_counts.items()},
               'normative_cue_count': sum(r['normative_cue'] for r in output),
               'no_cue_count': sum(not r['normative_cue'] for r in output),
               'executable_claim_count': len(witnesses),
               'executable_claim_counts_by_persona': dict(Counter(p for s, p in witnesses)),
               'model_stats': model_stats, 'executable_claim_persona_macro_ESS': persona_macro,
               'code_claim_bureaucrat_minus_authority': {'point': persona_macro['bureaucrat'] - persona_macro['authority'], 'ci95': np.quantile(diff, [.025, .975]).tolist()},
               'bootstrap': {'replicates': 10000, 'seed': 20261007, 'sampling_unit': 'whole code seed', 'code_seed_count': len(coding_seeds), 'model_roster_outputs_labels_fixed': True},
               'science_correction_groups_from_construction_metadata': dict(kind_counts),
               'scope': ['Cue flags are keyword diagnostics, not an exhaustive factual/normative semantic annotation.',
                         'No-cue does not establish factuality; cue and false-output witness can coexist.',
                         'Executable subset includes only exact-text false-output claims with saved execution evidence; wrong proposed implementations alone are excluded.',
                         'Executable subset covers code, differs in persona/task coverage, and does not establish cross-domain or causal framing conclusions.',
                         'No model responses, benchmark labels, or main metrics were changed.']}
    OUT.mkdir(exist_ok=True)
    (OUT / 'attack_tags.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in output), encoding='utf-8')
    (OUT / 'science_correction_groups.json').write_text(json.dumps(correction_rows, indent=2) + '\n', encoding='utf-8')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ['released_attacks','normative_cue_count','no_cue_count','executable_claim_count','cross_counts','executable_claim_persona_macro_ESS','code_claim_bureaucrat_minus_authority','science_correction_groups_from_construction_metadata']}, indent=2))


if __name__ == '__main__':
    main()
