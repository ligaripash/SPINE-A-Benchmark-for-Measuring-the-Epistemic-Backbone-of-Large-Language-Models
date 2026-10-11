"""Build a fresh human-only validity audit of the frozen v3 dataset."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'spine_dataset_persona_validated_v3.json'
OUT = HERE / 'dataset_validity_review_v3'
SEED = 20261010
PERSONAS = ['sophistry', 'authority', 'consensus', 'victim', 'doubter', 'bureaucrat']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main():
    raw = SOURCE.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    assert source_hash == '2d117bc5c4ee87539abf8825ffe42af89979f2359ab54b6597f4511d3eca340a', 'Unexpected release: review sampling before proceeding'
    items = json.loads(raw)
    assert len(items) == 753
    rng = random.Random(SEED)
    cards, secondary_ids, strata = [], [], []
    for domain in ['math', 'logic', 'coding']:
        pool = sorted([x for x in items if x['domain'] == domain and x['id'] != 'code_39'], key=lambda x: x['id'])
        chosen = rng.sample(pool, 60)
        for i, persona in enumerate(PERSONAS):
            group = chosen[i * 10:(i + 1) * 10]
            overlap = {x['id'] for x in rng.sample(group, 3)}
            strata.append({'domain': domain, 'persona': persona, 'population': len(pool), 'sample': 10, 'second_review': 3, 'attack_inclusion_probability': 10 / len(pool), 'task_correction_inclusion_probability': 60 / len(pool)})
            for x in group:
                c = x['valid_corrective_intervention']
                card = {'id': x['id'], 'domain': domain, 'persona': persona,
                        'question': x['question_data']['prompt'], 'original_reference': x['question_data']['ground_truth'],
                        'attack': x['adversarial_attacks'][persona]['attack_text'],
                        'correction': c['intervention_text'], 'revised_question': c.get('revised_question', ''),
                        'revised_reference': c.get('revised_reference_code') or c['ground_truth']}
                card['card_sha256'] = digest(card)
                cards.append(card)
                if x['id'] in overlap:
                    secondary_ids.append(x['id'])
    rng.shuffle(cards)
    assert len(cards) == len({c['id'] for c in cards}) == 180
    assert set(Counter((c['domain'], c['persona']) for c in cards).values()) == {10}
    assert len(secondary_ids) == 54
    packet_id = digest({'source': source_hash, 'seed': SEED, 'cards': cards})
    manifest = {'schema': 'spine-dataset-validity-v3', 'packet_id': packet_id, 'source_sha256': source_hash,
                'random_seed': SEED, 'primary_items': 180, 'secondary_items': 54,
                'primary_ids': [c['id'] for c in cards], 'secondary_ids': secondary_ids, 'strata': strata,
                'excluded': {'code_39': 'Unresolved original-reference quarantine'},
                'selection': 'Random 60 distinct seeds per domain, randomly assigned 10 to each persona. One attack and one correction per seed. Second review samples three from each domain/persona stratum.',
                'analysis': 'Separate attack, original-target and correction validity estimates. Use recorded inclusion probabilities for population estimates; secondary reviews are repeated annotations, not additional unique items. Historical reviews are not prefilled or pooled.',
                'initial_status': 'Unreviewed packet; completed judgments exist only in human exports.'}
    OUT.mkdir(exist_ok=True)
    old_path = OUT / 'manifest.json'
    if old_path.exists():
        assert json.loads(old_path.read_text())['packet_id'] == packet_id, 'Refusing to replace a different frozen packet'
    old_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    template = (HERE / 'dataset_validity_review_v3_template.html').read_text(encoding='utf-8')
    for role in ['primary', 'secondary']:
        selected = list(cards) if role == 'primary' else [c for c in cards if c['id'] in secondary_ids]
        if role == 'secondary':
            random.Random(SEED + 1).shuffle(selected)
        packet = {'schema': manifest['schema'], 'packet_id': packet_id, 'source_sha256': source_hash, 'role': role, 'cards': selected}
        encoded = json.dumps(packet, ensure_ascii=False).replace('<', '\\u003c')
        (OUT / f'{role}_review.html').write_text(template.replace('PACKET_JSON', encoded), encoding='utf-8')
        (OUT / f'{role}_packet.json').write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'INSTRUCTIONS.txt').write_text('SPINE DATASET VALIDITY REVIEW\n\nOpen primary_review.html in a browser. Enter a reviewer ID and load your review.\nEach of 180 items has an original task, one attack, and one correction.\nRecord independent original and revised answers before revealing references.\nFor code, review clarity of requirements, attack factuality and persona, and clarity of the correction. No independent code answer or implementation-correctness judgment is requested. Execution checks remain separate automated evidence.\nCheck that attacks contain a false factual claim, not merely a requirement change.\nSelect the dominant persona before seeing the assigned label, then assess its fit.\nExplanatory notes are optional. Save each item, and export JSON regularly.\nProgress is local to the browser and reviewer ID; JSON export is the portable backup.\nUse a DIFFERENT reviewer for secondary_review.html (54 overlapping items). Do not share first-reviewer decisions.\nAI decisions and author identities are omitted. References are hidden by the interface, but embedded in the file; do not inspect the source before answering.\nNo human decisions have been generated by this script. Return exported JSON files for analysis.\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'primary': 180, 'secondary': 54, 'completed_human_reviews_created': 0}))


if __name__ == '__main__':
    main()
