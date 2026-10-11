"""Draw a reproducible, domain-stratified SPINE validity sample.

All 60 authored correction seeds are certainty strata. Draw 20 seeds from each
domain's remaining seeds without replacement. Keep each selected seed and its
six persona challenges together. This is a probability sample of the fixed
764-item source population with unequal, recorded inclusion probabilities.
"""
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'sampled_dataset_review'
SOURCE = ROOT / 'spine_dataset.json'
SEED = 20260927
FAMILIES = ('sophistry','authority','consensus','victim','doubter','bureaucrat')
DOMAINS = ('math','logic','coding')


def write_csv(path, fields, rows):
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def build():
    raw=SOURCE.read_bytes()
    source=json.loads(raw)
    augmented=json.loads((ROOT/'spine_dataset_with_valid_corrections.json').read_text(encoding='utf-8'))
    correction_ids={x['id'] for x in augmented if x['valid_corrective_intervention']}
    assert len(source)==764 and len(correction_ids)==60
    rng=random.Random(SEED)
    strata={}
    primary=set(correction_ids)
    second=set()
    for domain in DOMAINS:
        all_ids=[x['id'] for x in source if x['domain']==domain]
        pilot=[x for x in all_ids if x in correction_ids]
        remaining=[x for x in all_ids if x not in correction_ids]
        assert len(pilot)==20
        selected=rng.sample(remaining,20)
        primary.update(selected)
        second.update(rng.sample(pilot,5))
        second.update(rng.sample(selected,5))
        strata[domain]={
            'population_total':len(all_ids),
            'correction_seed_census':len(pilot),
            'remaining_population':len(remaining),
            'remaining_random_sample':len(selected),
            'remaining_inclusion_probability':20/len(remaining),
            'remaining_design_weight':len(remaining)/20,
        }
    assert len(primary)==120 and len(second)==30 and second<=primary
    OUT.mkdir(exist_ok=True)
    cards=[]
    for old,new in zip(source,augmented):
        assert old['id']==new['id']
        if old['id'] not in primary:
            continue
        seed=old['id']
        correction=new['valid_corrective_intervention']
        cards.append({
            'id':seed,'domain':old['domain'],
            'selection_stratum':'correction_census' if seed in correction_ids else 'remaining_random',
            'inclusion_probability':1.0 if seed in correction_ids else strata[old['domain']]['remaining_inclusion_probability'],
            'design_weight':1.0 if seed in correction_ids else strata[old['domain']]['remaining_design_weight'],
            'original_question':old['question_data']['prompt'],
            'attacks':{p:old['adversarial_attacks'][p]['attack_text'] for p in FAMILIES},
            **({'valid_correction_followup':correction['intervention_text']} if correction else {}),
        })
    rng.shuffle(cards)
    with (OUT/'blind_tasks.jsonl').open('w',encoding='utf-8') as f:
        for c in cards:
            f.write(json.dumps({k:c[k] for k in ('id','domain','original_question')},ensure_ascii=False)+'\n')
    with (OUT/'blind_attacks.jsonl').open('w',encoding='utf-8') as f:
        for c in cards:
            for p in FAMILIES:
                f.write(json.dumps({'id':c['id'],'domain':c['domain'],'persona':p,
                                    'original_question':c['original_question'],
                                    'attack_text':c['attacks'][p]},ensure_ascii=False)+'\n')
    with (OUT/'blind_corrections.jsonl').open('w',encoding='utf-8') as f:
        for c in cards:
            if 'valid_correction_followup' in c:
                f.write(json.dumps({'id':c['id'],'domain':c['domain'],
                                    'original_question':c['original_question'],
                                    'followup_user_message':c['valid_correction_followup']},ensure_ascii=False)+'\n')

    task_fields=['id','domain','reviewer_id','independent_original_answer','question_clear_and_self_contained',
                 'one_defensible_answer_or_test_oracle','key_matches_independent_answer_after_reveal',
                 'source_answer_provenance_or_reference','decision','issue_codes','explanation']
    attack_fields=['id','domain','persona','reviewer_id','challenge_status','introduces_legitimate_new_information',
                   'asserted_alternative_if_any','asserted_alternative_correct_under_original','pressure_family_fit',
                   'wording_unambiguous','decision','issue_codes','explanation']
    correction_fields=['id','domain','reviewer_id','independent_revised_answer','explicit_change',
                       'original_answer_invalid_after_change','revised_answer_unique_and_checkable',
                       'internally_consistent','minimal_change','rewrites_options_or_question','kind',
                       'key_matches_independent_answer_after_reveal','decision','issue_codes','explanation']
    for name,subset in (('primary',primary),('second_stratified',second)):
        reviewer='primary' if name=='primary' else 'second'
        tasks=[{'id':c['id'],'domain':c['domain'],'reviewer_id':reviewer} for c in cards if c['id'] in subset]
        attacks=[{'id':c['id'],'domain':c['domain'],'persona':p,'reviewer_id':reviewer}
                 for c in cards if c['id'] in subset for p in FAMILIES]
        write_csv(OUT/f'{name}_tasks.csv',task_fields,tasks)
        write_csv(OUT/f'{name}_attacks.csv',attack_fields,attacks)
    for name in ('primary','second'):
        corrections=[{'id':c['id'],'domain':c['domain'],'reviewer_id':name}
                     for c in cards if 'valid_correction_followup' in c]
        write_csv(OUT/f'{name}_corrections.csv',correction_fields,corrections)
    manifest={
        'source_sha256':hashlib.sha256(raw).hexdigest(),
        'random_seed':SEED,
        'sample_seed_count':len(primary),
        'sample_seed_ids':[c['id'] for c in cards],
        'second_seed_count':len(second),
        'second_seed_ids':[c['id'] for c in cards if c['id'] in second],
        'primary_task_count':len(primary),
        'primary_attack_count':len(primary)*6,
        'primary_correction_count':len(correction_ids),
        'primary_total_ratings':len(primary)*7+len(correction_ids),
        'second_task_count':len(second),
        'second_attack_count':len(second)*6,
        'second_correction_count':len(correction_ids),
        'second_total_ratings':len(second)*7+len(correction_ids),
        'strata':strata,
        'selection':'All 20 authored-correction seeds per domain plus a simple random sample of 20 of the remaining seeds per domain; each selected seed carries all six attacks. Second review: 5 correction and 5 remaining seeds per domain, plus every correction. Drawn before human labels.',
        'analysis':'Use inverse-probability design weights for estimates about the fixed 764-seed original dataset. Report correction validity for the 60 authored branches only. Do not merge targeted issue audits into the probability sample estimate.',
        'status':'unreviewed',
    }
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if not k.endswith('_ids')},indent=2))


if __name__=='__main__':
    build()
