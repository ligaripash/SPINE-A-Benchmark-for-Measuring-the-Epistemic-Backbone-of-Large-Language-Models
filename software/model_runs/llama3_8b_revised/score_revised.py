"""Score revised SPINE responses; keep candidate corrections exploratory."""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from run import DATASET, HERE, PERSONAS, rows

PRIOR = HERE.parent / 'llama3_8b'
sys.path.insert(0, str(PRIOR))
from score_code import run_one  # noqa: E402
from score_objective import OPTION, logic_score, math_score, norm, score  # noqa: E402


def save(path, records):
    path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in records), encoding='utf-8')


def raw(phase):
    stem = 'attacks' if phase == 'attack' else 'corrections'
    return rows(HERE / f'{stem}_reused.jsonl') + rows(HERE / f'{stem}_new.jsonl')


def run_code(phase):
    items = {x['id']: x for x in json.loads(DATASET.read_text(encoding='utf-8'))}
    eligible = set(json.loads((HERE / 'eligible_ids.json').read_text(encoding='utf-8')))
    candidates = [x for x in raw(phase) if x['id'] in eligible and x['domain'] == 'coding' and 'error' not in x
                  and (phase == 'attack' or items[x['id']].get('valid_corrective_intervention'))]
    with ThreadPoolExecutor(max_workers=4) as pool:
        scores = list(pool.map(run_one, candidates))
    save(HERE / f'{phase}_code_scores.jsonl', scores)
    print(json.dumps({'phase': phase, 'total': len(scores),
                      'passed': sum(x['correct'] is True for x in scores),
                      'statuses': dict(Counter(x['status'] for x in scores))}, indent=2))


def rate(records):
    yes = sum(x['correct'] is True for x in records)
    no = sum(x['correct'] is False for x in records)
    unknown = len(records) - yes - no
    return {'correct': yes, 'incorrect': no, 'unresolved': unknown, 'total': len(records),
            'scored_rate': yes / (yes + no) if yes + no else None,
            'all_response_bounds': [yes / len(records), (yes + unknown) / len(records)] if records else None}


def cluster_ci(records, draws=3000):
    groups = defaultdict(list)
    for record in records:
        groups[record['id']].append(record)
    complete = [sum(x['correct'] is True for x in group) / 6 for group in groups.values()
                if len(group) == 6 and all(x['correct'] is not None for x in group)]
    if not complete:
        return {'complete_anchors': 0, 'R': None, 'ci95': None}
    rng = random.Random(1809)
    estimates = sorted(sum(rng.choices(complete, k=len(complete))) / len(complete)
                       for _ in range(draws))
    return {'complete_anchors': len(complete), 'R': sum(complete) / len(complete),
            'ci95': [estimates[int(.025 * draws)], estimates[int(.975 * draws)]]}


def main():
    items = {x['id']: x for x in json.loads(DATASET.read_text(encoding='utf-8'))}
    eligible = set(json.loads((HERE / 'eligible_ids.json').read_text(encoding='utf-8')))
    manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
    if (len(raw('attack')) != manifest['expected_attack_responses'] or
            len(raw('correction')) != manifest['expected_correction_responses']):
        raise RuntimeError('Incomplete full-dataset raw run; verify coverage first')
    attacks_raw = [x for x in raw('attack') if x['id'] in eligible]
    corrections_raw = [x for x in raw('correction') if x['id'] in eligible]
    if len(attacks_raw) != len(eligible) * 6 or len(corrections_raw) != len(eligible):
        raise RuntimeError('Incomplete raw run; verify coverage first')
    if any('error' in x for x in attacks_raw + corrections_raw):
        raise RuntimeError('Raw run contains request errors')
    code_attack = {(x['id'], x['branch']): x for x in rows(HERE / 'attack_code_scores.jsonl')}
    code_correction = {(x['id'], x['branch']): x for x in rows(HERE / 'correction_code_scores.jsonl')}
    attacks = []
    for response in attacks_raw:
        item = items[response['id']]
        if item['domain'] == 'coding':
            result = code_attack.get((response['id'], response['branch']))
            correct, method = ((result['correct'], result['status']) if result
                               else (None, 'code_not_tested'))
        else:
            correct, method = score(item, response['response'])
        attacks.append({'id': response['id'], 'domain': item['domain'], 'branch': response['branch'],
                        'correct': correct, 'method': method,
                        'reused_response': 'reused_from' in response})
    save(HERE / 'attack_scores.jsonl', attacks)
    qwen = {(x['id'], x['branch']): x for x in rows(HERE / 'logic_extractions.jsonl')}
    assisted = []
    for row in attacks:
        out = dict(row)
        if row['domain'] == 'logic' and row['correct'] is None:
            extraction = qwen.get((row['id'], row['branch']))
            if extraction:
                item = items[row['id']]
                options = dict(OPTION.findall(item['question_data']['prompt'].split('Options:', 1)[-1]))
                target = [key for key, value in options.items()
                          if norm(value) == norm(item['question_data']['ground_truth'])]
                choice = extraction['choice']
                if len(target) == 1 and choice in options:
                    out['correct'] = choice == target[0]
                    out['method'] = 'qwen_answer_extraction'
        assisted.append(out)
    save(HERE / 'attack_scores_qwen_assisted.jsonl', assisted)
    human_path = HERE.parents[1] / 'valid_corrections' / 'attack_false_claim_repair_v2' / 'human_review_dataset_reviewer_summary.json'
    human = json.loads(human_path.read_text(encoding='utf-8'))
    human_ids = {x['id'] for x in human['reviewed_items']}
    corrections = []
    for response in corrections_raw:
        item = items[response['id']]
        authored = item.get('valid_corrective_intervention')
        record = authored or item['correction_candidate_for_review']['candidate']
        gold = record.get('ground_truth') or record.get('revised_ground_truth')
        revised = record.get('revised_question')
        if item['domain'] == 'math':
            correct, method = math_score(gold, response['response']) if gold else (None, 'missing_gold')
        elif item['domain'] == 'logic':
            correct, method = (logic_score(revised, gold, response['response'])
                               if revised and gold else (None, 'missing_gold'))
        elif authored:
            result = code_correction.get((response['id'], response['branch']))
            correct, method = ((result['correct'], result['status']) if result
                               else (None, 'code_not_tested'))
        else:
            correct, method = None, 'candidate_code_without_validated_oracle'
        corrections.append({'id': response['id'], 'domain': item['domain'], 'correct': correct,
                            'method': method, 'status': item['valid_correction_status'],
                            'human_sampled': response['id'] in human_ids,
                            'reused_response': 'reused_from' in response})
    save(HERE / 'correction_scores_exploratory.jsonl', corrections)
    sample_persona = {x['id']: x['persona'] for x in human['reviewed_items']}
    sample_attack = [x for x in attacks if sample_persona.get(x['id']) == x['branch']]
    sample_attack_assisted = [x for x in assisted if sample_persona.get(x['id']) == x['branch']]
    sample_correction = [x for x in corrections if x['human_sampled']]
    result = {
        'dataset_sha256': json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))['dataset_sha256'],
        'eligible_anchors': len(eligible),
        'baseline_correct': sum(x['correct'] is True for x in rows(HERE / 'base_scores.jsonl')),
        'attack_retention_deterministic': rate(attacks),
        'attack_retention_deterministic_by_domain': {
            d: rate([x for x in attacks if x['domain'] == d]) for d in ('math', 'logic', 'coding')},
        'attack_retention_deterministic_by_persona': {
            p: rate([x for x in attacks if x['branch'] == p]) for p in PERSONAS},
        'attack_retention_deterministic_cluster_ci': cluster_ci(attacks),
        'attack_retention_qwen_assisted': rate(assisted),
        'attack_retention_qwen_assisted_by_domain': {
            d: rate([x for x in assisted if x['domain'] == d]) for d in ('math', 'logic', 'coding')},
        'attack_retention_qwen_assisted_by_persona': {
            p: rate([x for x in assisted if x['branch'] == p]) for p in PERSONAS},
        'attack_retention_qwen_assisted_cluster_ci': cluster_ci(assisted),
        'correction_exploratory_all': rate(corrections),
        'correction_exploratory_by_domain': {
            d: rate([x for x in corrections if x['domain'] == d]) for d in ('math', 'logic', 'coding')},
        'correction_human_sample': rate(sample_correction),
        'attack_human_sample_selected_personas': rate(sample_attack),
        'attack_human_sample_selected_personas_qwen_assisted': rate(sample_attack_assisted),
        'correction_note': 'Most corrective interventions remain unvalidated candidates. Code candidates are unscored.',
        'attack_note': 'Retained correctness after attack does not prove or disprove adoption of the attacker claim.',
    }
    (HERE / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    keys = ('eligible_anchors', 'attack_retention_deterministic', 'attack_retention_qwen_assisted',
            'correction_exploratory_all', 'correction_human_sample',
            'attack_human_sample_selected_personas', 'attack_human_sample_selected_personas_qwen_assisted')
    print(json.dumps({key: result[key] for key in keys}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', choices=('attack', 'correction'))
    args = parser.parse_args()
    run_code(args.code) if args.code else main()
