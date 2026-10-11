"""Produce conservative triage for generated correction drafts.

This does not confer gold status. Scientific and functional validity still
require independent checks and human review.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from generate_gpt4o_candidates import ROOT, SOURCE, options, screen
from candidate_store import load_candidates

IN = ROOT / 'valid_corrections/reasoned_candidate_drafts.jsonl'
OUT = ROOT / 'valid_corrections/generated_candidate_triage.csv'
REPORT = ROOT / 'valid_corrections/generated_candidate_audit.json'
ADJUDICATION = ROOT / 'valid_corrections/original_source_adjudication.csv'


def numeric(value: str) -> Decimal | None:
    value = value.strip().replace(',', '').replace('$', '')
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def check(item: dict, row: dict) -> list[str]:
    flags = []
    candidate = row.get('candidate') or {}
    if row.get('error') or not candidate:
        return ['generation_failed']
    flags.extend(screen(item, candidate))
    if candidate.get('unable'):
        return sorted(set(flags + ['source_or_revision_requires_adjudication']))
    if item['domain'] == 'math':
        revised = numeric(candidate.get('revised_ground_truth', ''))
        original = numeric(str(item['question_data']['ground_truth']))
        if revised is None:
            flags.append('revised_gold_not_plain_numeric')
        if revised is not None and original is not None and revised == original:
            flags.append('same_numeric_answer')
        derivation = candidate.get('derivation', '')
        # A model can output an answer that conflicts with its own final
        # arithmetic line. This catches obvious cases without claiming to
        # verify the natural-language word problem.
        nums = re.findall(r'(?<![A-Za-z0-9])(-?\$?\d[\d,]*(?:\.\d+)?)(?![\d])', derivation)
        if nums and revised is not None:
            last = numeric(nums[-1])
            if last is not None and last != revised:
                flags.append('derivation_final_number_mismatch')
    elif item['domain'] == 'logic':
        old_opts = options(item['question_data']['prompt'])
        new_opts = options(candidate.get('revised_question', ''))
        if len(new_opts) != 4:
            flags.append('revised_question_missing_four_options')
        if len(old_opts) != 4:
            flags.append('source_option_parse_issue')
        label = candidate.get('revised_option_label', '')
        if label in new_opts and new_opts[label].strip().casefold() == str(item['question_data']['ground_truth']).strip().casefold():
            flags.append('revised_option_still_original_answer')
    else:
        tests = candidate.get('distinguishing_tests') or []
        if len(tests) < 3:
            flags.append('insufficient_code_tests')
        if not candidate.get('revised_reference_code', '').strip():
            flags.append('missing_revised_code')
    return sorted(set(flags))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=IN)
    args = p.parse_args()
    raw = SOURCE.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    source = {x['id']: x for x in json.loads(raw)}
    augmented = json.loads((ROOT / 'spine_dataset_with_valid_corrections.json').read_text(encoding='utf-8'))
    missing = {x['id'] for x in augmented if x['valid_corrective_intervention'] is None}
    results = []
    seen = set()
    rows = (load_candidates().values() if args.input == IN else
            (json.loads(line) for line in args.input.read_text(encoding='utf-8').splitlines() if line.strip()))
    for row in rows:
        item_id = row['id']
        if item_id in seen:
            raise ValueError(f'duplicate candidate: {item_id}')
        seen.add(item_id)
        if item_id not in source or item_id not in missing:
            raise ValueError(f'unexpected candidate: {item_id}')
        if row['source_sha256'] != source_hash:
            raise ValueError(f'source hash mismatch: {item_id}')
        flags = check(source[item_id], row)
        candidate = row.get('candidate') or {}
        results.append({'id': item_id, 'domain': source[item_id]['domain'],
                        'generator': row.get('generator', ''),
                        'state': ('generation_failed' if not candidate else
                                  'unable' if candidate.get('unable') else
                                  'candidate_unvalidated'),
                        'flags': ';'.join(flags),
                        'original_answer': str(source[item_id]['question_data']['ground_truth']),
                        'revised_answer': str(candidate.get('revised_ground_truth', '')),
                        'intervention_text': candidate.get('intervention_text', ''),
                        'reason': candidate.get('reason', '')})
    with OUT.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]) if results else
                                ['id','domain','generator','state','flags','original_answer',
                                 'revised_answer','intervention_text','reason'])
        writer.writeheader()
        writer.writerows(results)
    disputed = [{'id': row['id'], 'domain': row['domain'],
                 'original_question': source[row['id']]['question_data']['prompt'],
                 'stored_answer': source[row['id']]['question_data']['ground_truth'],
                 'audit_concern': row['reason'], 'independent_answer': '',
                 'original_validity_decision': '', 'adjudication_notes': ''}
                for row in results if row['state'] == 'unable']
    extra_source_checks = {
        'code_124': 'The stored valid_date reference returns False for 12-31-1999 although the docstring permits this date. EvalPlus independently reports the same HumanEval/124 reference bug: https://arxiv.org/html/2305.01210v3',
        'logic_81': 'Possible ambiguity: the stem names deep-sea organisms generally, while both attracting prey and escaping predators are documented uses of bioluminescence: https://oceanservice.noaa.gov/facts/biolum.html',
    }
    for item_id, concern in extra_source_checks.items():
        item = source[item_id]
        disputed.append({'id': item_id, 'domain': item['domain'],
                         'original_question': item['question_data']['prompt'],
                         'stored_answer': item['question_data']['ground_truth'],
                         'audit_concern': concern, 'independent_answer': '',
                         'original_validity_decision': '', 'adjudication_notes': ''})
    if ADJUDICATION.exists():
        with ADJUDICATION.open(newline='', encoding='utf-8-sig') as stream:
            prior = list(csv.DictReader(stream))
        if any(any(row.get(key, '').strip() for key in
                   ('independent_answer','original_validity_decision','adjudication_notes'))
               for row in prior):
            raise ValueError('refusing to overwrite human adjudication responses')
    with ADJUDICATION.open('w', newline='', encoding='utf-8-sig') as stream:
        fields = ['id','domain','original_question','stored_answer','audit_concern',
                  'independent_answer','original_validity_decision','adjudication_notes']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(disputed)
    flag_counts = Counter(f for row in results for f in row['flags'].split(';') if f)
    by_domain = defaultdict(Counter)
    for row in results:
        by_domain[row['domain']][row['state']] += 1
    report = {'source_sha256': source_hash, 'total_seeds': len(source),
              'pilot_authored': len(source) - len(missing),
              'missing_before_generation': len(missing),
              'drafts_recorded': len(results),
              'missing_drafts': len(missing) - len(results),
              'candidate_states': dict(Counter(row['state'] for row in results)),
              'source_adjudication_items': len(disputed),
              'states_by_domain': {d: dict(c) for d, c in by_domain.items()},
              'flag_counts': dict(flag_counts),
              'gold_promotions': 0,
              'independent_human_validation': 'pending'}
    REPORT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
