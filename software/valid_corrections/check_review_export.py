"""Summarize a human's exported review JSON without creating missing ratings.

Usage: python valid_corrections/check_review_export.py path/to/export.json
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAMILIES = ('sophistry','authority','consensus','victim','doubter','bureaucrat')
ALLOWED = {'accept','revise','exclude','unsure'}


def check(path):
    source = json.loads((ROOT/'spine_dataset.json').read_text(encoding='utf-8'))
    augmented = json.loads((ROOT/'spine_dataset_with_valid_corrections.json').read_text(encoding='utf-8'))
    expected_hash = hashlib.sha256((ROOT/'spine_dataset.json').read_bytes()).hexdigest()
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('schema') != 'spine-human-review-v1' or payload.get('source_sha256') != expected_hash:
        raise ValueError('Export schema or source dataset hash does not match')
    ratings=payload.get('ratings',{})
    scope=payload.get('scope','full')
    review_mode=payload.get('review_mode','all')
    if scope not in {'full','sampled','rapid'}:
        raise ValueError('Unknown review scope')
    if review_mode not in {'all','evidence'}:
        raise ValueError('Unknown review mode')
    corrected_hash=hashlib.sha256((ROOT/'spine_dataset_ground_truth_corrected.json').read_bytes()).hexdigest()
    repaired_path=ROOT/'valid_corrections/attack_false_claim_repair_v2/spine_dataset_false_claim_attacks.json'
    repaired_hash=hashlib.sha256(repaired_path.read_bytes()).hexdigest() if repaired_path.exists() else None
    repaired=payload.get('corrected_sha256')==repaired_hash
    if repaired:
        corrected_hash=repaired_hash
    elif payload.get('corrected_sha256') not in (None, corrected_hash):
        raise ValueError('Unknown corrected dataset hash')
    corrected_dataset=repaired_path if repaired else ROOT/'spine_dataset_ground_truth_corrected.json'
    if review_mode=='evidence':
        if payload.get('corrected_sha256') != corrected_hash:
            raise ValueError('Export corrected dataset hash does not match')
    rapid_attacks={}
    if scope=='rapid':
        rapid_manifest=(ROOT/'valid_corrections/attack_false_claim_repair_v2/rapid_review_manifest.json'
                        if repaired else ROOT/'valid_corrections/rapid_evidence_review/manifest.json')
        rapid=json.loads(rapid_manifest.read_text(encoding='utf-8'))
        if rapid['source_sha256'] != corrected_hash:
            raise ValueError('Rapid sample and corrected dataset hashes do not match')
        rapid_attacks={row['id']:row['persona'] for row in rapid['selected']}
        sample_ids=set(rapid_attacks)
    elif scope=='sampled':
        if review_mode=='evidence' or payload.get('sample_seed_count')==119:
            sample_manifest=(ROOT/'valid_corrections/attack_false_claim_repair_v2/sampled_review_manifest.json'
                             if repaired else ROOT/'valid_corrections/sampled_evidence_review/manifest.json')
            sample=json.loads(sample_manifest.read_text(encoding='utf-8'))
            if sample['population_sha256'] != corrected_hash:
                raise ValueError('Sample and corrected dataset hashes do not match')
        else:
            sample=json.loads((ROOT/'valid_corrections/sampled_dataset_review/manifest.json').read_text(encoding='utf-8'))
            if sample['source_sha256'] != expected_hash:
                raise ValueError('Sample and source hashes do not match')
        sample_ids=set(sample['sample_seed_ids'])
    else:
        sample_ids={item['id'] for item in source}
    expected=[]
    retained={item['id'] for item in json.loads(corrected_dataset.read_text(encoding='utf-8'))} if review_mode=='evidence' or scope=='rapid' or (scope=='sampled' and payload.get('sample_seed_count')==119) else None
    for item in source:
        seed=item['id']
        if seed not in sample_ids or (retained is not None and seed not in retained) or (repaired and item['domain']=='coding'):
            continue
        if review_mode=='all':
            expected.append('task:'+seed)
        expected.extend('attack:'+seed+':'+p for p in ((rapid_attacks[seed],) if scope=='rapid' else FAMILIES))
    if review_mode=='evidence' or scope=='rapid' or (scope=='sampled' and payload.get('sample_seed_count')==119):
        expected.extend('correction:'+item['id'] for item in source if item['id'] in sample_ids and item['id'] in retained and ((not repaired and scope!='rapid') or item['domain']!='coding'))
    else:
        expected.extend('correction:'+item['id'] for item in augmented if item['valid_corrective_intervention'] and (scope!='rapid' or item['id'] in sample_ids) and (not repaired or item['domain']!='coding'))
    status=Counter()
    invalid=[]
    by_unit=Counter()
    for key in expected:
        unit=key.split(':',1)[0]
        r=ratings.get(key,{})
        decision=r.get('decision')
        if decision not in ALLOWED:
            status['unrated']+=1
            by_unit[unit+'_unrated']+=1
            continue
        status[decision]+=1
        by_unit[unit+'_'+decision]+=1
        if key.startswith('attack:'):
            category=r.get('category')
            if not category:
                invalid.append((key,'challenge status missing'))
            eligible={'false_claim'} if repaired else {'false_claim','unsupported_pressure'}
            if decision=='accept' and category not in eligible:
                invalid.append((key,'accepted challenge has ineligible status'))
        elif key.startswith('correction:') and decision=='accept' and not str(r.get('independent','')).strip():
            invalid.append((key,'accepted task/correction lacks independent answer'))
        elif key.startswith('task:') and decision=='accept' and not str(r.get('independent','')).strip():
            invalid.append((key,'accepted task/correction lacks independent answer'))
        if decision in {'revise','exclude','unsure'} and not str(r.get('note','')).strip():
            invalid.append((key,'non-accept decision needs explanation'))
    unexpected=sorted(set(ratings)-set(expected))
    result={
        'reviewer':payload.get('reviewer',''),
        'scope':scope,
        'review_mode':review_mode,
        'source_sha256':expected_hash,
        'dataset_version':'repaired_false_claim_attacks' if repaired else 'ground_truth_corrected',
        'expected_units':len(expected),
        'status_counts':dict(status),
        'by_unit_and_decision':dict(by_unit),
        'form_issues_count':len(invalid),
        'form_issues_preview':[{'key':k,'issue':v} for k,v in invalid[:30]],
        'ratings_outside_scope':len(unexpected),
        'complete_and_well_formed':status['unrated']==0 and not invalid and bool(payload.get('reviewer')),
    }
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export',type=Path)
    args=parser.parse_args()
    print(json.dumps(check(args.export),indent=2))
