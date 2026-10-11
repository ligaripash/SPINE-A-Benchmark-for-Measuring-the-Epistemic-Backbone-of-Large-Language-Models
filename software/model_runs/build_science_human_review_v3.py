"""Build a blinded human audit of the verified science response-label policy.

Offline only. Original data, model outputs, and AI labels are never modified.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'model_runs/persona_revision_v3'
JUDGES = ROOT / 'model_runs/judge_robustness'
OUT = SOURCE / 'human_science_review'
SEED = 20261007


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    verified = load(SOURCE / 'final_verification.json')
    assert verified['status'] == 'verified'
    assert sha(SOURCE / 'metrics.json') == verified['metrics_sha256']
    index = [r for r in rows(SOURCE / 'final_response_index.jsonl') if r['domain'] == 'logic']
    labels = load(SOURCE / 'final_correctness_labels.json')
    old_final = load(JUDGES / 'sol_disagreement_adjudication/final_science_response_labels.json')
    extracts = {j: {r['blind_id']: r for r in rows(JUDGES / p)} for j, p in
                [('luna', 'luna_labels.jsonl'), ('flash_lite', 'gemini_flash_lite_labels.jsonl')]}
    scores = {}
    for directory in sorted(SOURCE.iterdir()):
        if not directory.is_dir() or not (directory / 'judgments').exists():
            continue
        for j in extracts:
            for path in (directory / 'judgments' / j).glob('*.json'):
                record = load(path)
                assert record['blind_id'] not in extracts[j]
                extracts[j][record['blind_id']] = record
        for path in (directory / 'scores').glob('*.json'):
            r = load(path)
            if r.get('index_record', {}).get('domain') == 'logic':
                scores[r['blind_id']] = r
    strata = defaultdict(list)
    private = {}
    for r in index:
        uid = r['blind_id']
        a, b = [extracts[j][uid] for j in ('luna', 'flash_lite')]
        assert a['input_sha256'] == b['input_sha256'] == r['input_sha256']
        if uid in scores:
            s = scores[uid]
            choice, origin = s['details']['choice'], s['method']
        else:
            s = old_final[uid]
            assert s['input_sha256'] == r['input_sha256']
            choice, origin = s['choice'], s['source']
        expected = None if choice == 'UNCLEAR' else choice == r['target_option']
        assert expected is labels[uid]
        category = 'unresolved' if choice == 'UNCLEAR' else ('agreed' if origin == 'Luna/Flash-Lite agreement' else 'adjudicated')
        stratum = '|'.join((r['run_name'], r['phase'], category))
        fields = r['judge_input']
        task = fields['task']
        option_labels = list(dict.fromkeys(re.findall(r'\(([A-E1-5])\)', task.split('Options:', 1)[-1])))
        assert len(option_labels) >= 2 and r['target_option'] in option_labels
        card = {'review_id': uid, 'phase': r['phase'], 'task': task,
                'original_task': fields.get('original_task', task),
                'initial_answer': fields.get('initial_answer', ''),
                'followup_message': fields.get('followup_message', ''),
                'response': fields['final_response'], 'option_labels': option_labels}
        card['card_sha256'] = text_hash(card)
        assert text_hash(fields) == r['input_sha256']
        # Original index hashes serialize the response string; newer runner
        # receipts use the same digest helper. Accept literal byte hashes too.
        assert r['response_sha256'] in {text_hash(card['response']), hashlib.sha256(card['response'].encode()).hexdigest()}
        private[uid] = {'review_id': uid, 'seed_id': r['id'], 'model': r['run_name'],
                        'phase': r['phase'], 'persona': r['branch'], 'stratum': stratum,
                        'category': category, 'final_choice': choice, 'final_correct': labels[uid],
                        'luna_choice': a['choice'], 'flash_lite_choice': b['choice'],
                        'origin': origin, 'target_option': r['target_option'],
                        'input_sha256': r['input_sha256'], 'response_sha256': r['response_sha256'],
                        'card_sha256': card['card_sha256'], 'card': card}
        strata[stratum].append(uid)
    assert len(index) == 11791
    # Random sampling within every model x phase x final-status stratum.
    # At least two records per non-census stratum enable design-based variance.
    allocation = {s: min(2, len(ids)) for s, ids in strata.items()}
    assert sum(allocation.values()) <= 120
    weights = {'agreed': .5, 'adjudicated': .3, 'unresolved': .2}
    phases = {'base': .2, 'attack': .6, 'correction': .2}
    targets = {s: weights[s.split('|')[2]] * phases[s.split('|')[1]] for s in strata}
    total_weight = sum(targets.values())
    targets = {s: 120 * w / total_weight for s, w in targets.items()}
    while sum(allocation.values()) < 120:
        available = [s for s in strata if allocation[s] < len(strata[s])]
        s = max(sorted(available), key=lambda s: targets[s] - allocation[s])
        allocation[s] += 1
    rng = random.Random(SEED)
    selected = []
    for s in sorted(strata):
        ids = rng.sample(sorted(strata[s]), allocation[s])
        for uid in ids:
            private[uid].update(population_n=len(strata[s]), sample_n=allocation[s],
                                inclusion_probability=allocation[s] / len(strata[s]),
                                analysis_weight=len(strata[s]) / allocation[s])
        selected.extend(ids)
    rng.shuffle(selected)
    # An independent reviewer receives a smaller overlap, in a different order.
    second = []
    groups = defaultdict(list)
    for uid in selected:
        groups[(private[uid]['model'], private[uid]['category'])].append(uid)
    for g in sorted(groups):
        second.append(rng.choice(groups[g]))
    remaining = [uid for uid in selected if uid not in second]
    second.extend(rng.sample(remaining, 30 - len(second)))
    rng.shuffle(second)
    primary = [private[uid]['card'] for uid in selected]
    secondary = [private[uid]['card'] for uid in second]
    packet_id = text_hash(primary)
    OUT.mkdir(exist_ok=True)
    if (OUT / 'manifest.json').exists():
        assert load(OUT / 'manifest.json')['packet_id'] == packet_id, 'Existing review packet differs; preserve it.'
    save('primary_blinded.json', primary)
    save('secondary_blinded.json', secondary)
    save('analyst_private.json', {uid: {k: v for k, v in private[uid].items() if k != 'card'} for uid in selected})
    manifest = {'schema': 'spine-science-human-audit-v3', 'packet_id': packet_id,
                'sampling_seed': SEED, 'population_n': len(index), 'primary_n': len(primary),
                'secondary_n': len(secondary), 'stratification': 'model x phase x final AI status',
                'sampling': 'Uniform without replacement within each stratum; oversampled adjudicated and unresolved records. Use population_n/sample_n weights for corpus estimates.',
                'strata': {s: {'population_n': len(strata[s]), 'sample_n': allocation[s]} for s in sorted(strata)},
                'source_hashes': {p: sha(SOURCE / p) for p in ['final_response_index.jsonl', 'final_correctness_labels.json', 'metrics.json']},
                'primary_ids': selected, 'secondary_ids': second,
                'human_reviews_completed': 0,
                'scope': 'Response interpretation only, not independent task-target or intervention validity. No AI labels, targets, judge identities, or responding-model metadata appear in the reviewer pages. Context text remains unaltered.'}
    save('manifest.json', manifest)
    template = (ROOT / 'model_runs/science_human_review_template.html').read_text(encoding='utf-8')
    for role, cards in [('primary', primary), ('secondary', secondary)]:
        body = json.dumps({'schema': manifest['schema'], 'packet_id': packet_id, 'role': role, 'cards': cards}, ensure_ascii=False).replace('<', '\\u003c')
        (OUT / (role + '_review.html')).write_text(template.replace('__PACKET_JSON__', body), encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'primary_n': len(primary), 'secondary_n': len(secondary),
                      'strata': len(strata), 'sample_by_status': dict(Counter(private[x]['category'] for x in selected)),
                      'sample_by_phase': dict(Counter(private[x]['phase'] for x in selected))}))


if __name__ == '__main__':
    main()
