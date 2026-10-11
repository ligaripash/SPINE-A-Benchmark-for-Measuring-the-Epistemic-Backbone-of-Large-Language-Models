"""Build gold-blind human review materials for all 764 SPINE seeds.

All original tasks and all six attacks per task receive a primary review row.
A stratified 20% seed sample receives an independent second review; every
authored valid correction receives two independent review rows.
"""
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / 'spine_dataset.json'
OUT = HERE / 'full_dataset_review'
FAMILIES = ('sophistry', 'authority', 'consensus', 'victim', 'doubter', 'bureaucrat')
SAMPLE_FRACTION = 0.20
RANDOM_SEED = 20260927


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build():
    source_bytes = SOURCE.read_bytes()
    source = json.loads(source_bytes)
    enriched = json.loads((ROOT / 'spine_dataset_with_valid_corrections.json').read_text(encoding='utf-8'))
    assert len(source) == len(enriched) == 764
    assert len({item['id'] for item in source}) == 764
    corrections = {item['id']: item['valid_corrective_intervention'] for item in enriched
                   if item['valid_corrective_intervention'] is not None}
    assert len(corrections) == 60
    by_domain = defaultdict(list)
    cards = []
    for original, augmented in zip(source, enriched):
        assert original['id'] == augmented['id']
        assert {k: augmented[k] for k in original} == original
        assert set(original['adversarial_attacks']) == set(FAMILIES)
        seed = original['id']
        by_domain[original['domain']].append(seed)
        card = {
            'id': seed,
            'domain': original['domain'],
            'original_question': original['question_data']['prompt'],
            'attacks': {family: original['adversarial_attacks'][family]['attack_text'] for family in FAMILIES},
        }
        if seed in corrections:
            card['valid_correction_followup'] = corrections[seed]['intervention_text']
        cards.append(card)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(cards)
    second = set()
    for domain, ids in sorted(by_domain.items()):
        ids = list(ids)
        rng.shuffle(ids)
        second.update(ids[:round(len(ids) * SAMPLE_FRACTION)])
    OUT.mkdir(exist_ok=True)
    with (OUT / 'blind_tasks.jsonl').open('w', encoding='utf-8') as stream:
        for card in cards:
            stream.write(json.dumps({k:card[k] for k in ('id','domain','original_question')}, ensure_ascii=False)+'\n')
    with (OUT / 'blind_attacks.jsonl').open('w', encoding='utf-8') as stream:
        for card in cards:
            for family in FAMILIES:
                attack_card = {'id':card['id'], 'domain':card['domain'],
                               'original_question':card['original_question'],
                               'persona':family, 'attack_text':card['attacks'][family]}
                stream.write(json.dumps(attack_card, ensure_ascii=False)+'\n')
    with (OUT / 'blind_corrections.jsonl').open('w', encoding='utf-8') as stream:
        for card in cards:
            if card['id'] in corrections:
                correction_card = {'id':card['id'], 'domain':card['domain'],
                                   'original_question':card['original_question'],
                                   'followup_user_message':card['valid_correction_followup']}
                stream.write(json.dumps(correction_card, ensure_ascii=False)+'\n')

    task_fields = [
        'id','domain','reviewer_id','independent_original_answer','question_clear_and_self_contained',
        'one_defensible_answer_or_test_oracle','key_matches_independent_answer_after_reveal',
        'source_answer_provenance_or_reference','decision','issue_codes','explanation',
    ]
    attack_fields = [
        'id','domain','persona','reviewer_id','challenge_status',
        'introduces_legitimate_new_information','asserted_alternative_if_any',
        'asserted_alternative_correct_under_original','pressure_family_fit',
        'wording_unambiguous','decision','issue_codes','explanation',
    ]
    correction_fields = [
        'id','domain','reviewer_id','independent_revised_answer','explicit_change',
        'original_answer_invalid_after_change','revised_answer_unique_and_checkable',
        'internally_consistent','minimal_change','rewrites_options_or_question',
        'kind','key_matches_independent_answer_after_reveal',
        'decision','issue_codes','explanation',
    ]
    for label, subset in (('primary', {c['id'] for c in cards}), ('second_stratified', second)):
        task_rows = []
        attack_rows = []
        for card in cards:
            if card['id'] not in subset:
                continue
            task_rows.append({'id':card['id'], 'domain':card['domain'], 'reviewer_id':label})
            for family in FAMILIES:
                attack_rows.append({'id':card['id'], 'domain':card['domain'],
                                    'persona':family, 'reviewer_id':label})
        write_csv(OUT / f'{label}_tasks.csv', task_fields, task_rows)
        write_csv(OUT / f'{label}_attacks.csv', attack_fields, attack_rows)
    correction_rows = [{'id':c['id'], 'domain':c['domain'], 'reviewer_id':'primary'}
                       for c in cards if c['id'] in corrections]
    write_csv(OUT / 'primary_corrections.csv', correction_fields, correction_rows)
    for row in correction_rows:
        row['reviewer_id'] = 'second'
    write_csv(OUT / 'second_corrections.csv', correction_fields, correction_rows)

    manifest = {
        'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'random_seed': RANDOM_SEED,
        'source_seed_count': len(source),
        'domain_counts': dict(Counter(c['domain'] for c in cards)),
        'primary_task_count':len(cards),
        'primary_attack_count':len(cards)*len(FAMILIES),
        'primary_correction_count':len(corrections),
        'second_stratified_seed_count':len(second),
        'second_stratified_domain_counts':dict(Counter(c['domain'] for c in cards if c['id'] in second)),
        'second_stratified_attack_count':len(second)*len(FAMILIES),
        'second_correction_count':len(corrections),
        'second_review_selection':'20% of seeds per domain, sampled before annotation; all six attacks for each selected seed; all 60 authored corrections also double-reviewed. Add every flagged or disputed primary item to second review.',
        'cards_include_gold_answers':False,
        'status':'unreviewed',
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    build()
