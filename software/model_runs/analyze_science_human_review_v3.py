"""Validate real reviewer exports and compare with frozen AI response labels.

This tool does not change benchmark labels, metrics, or the paper. Synthetic
reviewer IDs are rejected. Incomplete exports get coverage counts only.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'model_runs/persona_revision_v3/human_science_review'


def load(p):
    return json.loads(p.read_text(encoding='utf-8'))


def cosmetic(s):
    return re.sub(r'\s+', ' ', re.sub(r'[*_`]', '', unicodedata.normalize('NFKC', s))).strip()


def correct(choice, target):
    return None if choice == 'UNCLEAR' else choice == target


def validate(path, manifest, role):
    value = load(path)
    if value.get('schema') != manifest['schema'] or value.get('packet_id') != manifest['packet_id'] or value.get('role') != role:
        raise ValueError('Wrong packet or reviewer role')
    reviewer = value.get('reviewer', '').strip()
    if not reviewer or any(x in reviewer.lower() for x in ['synthetic', 'ui_test', 'test_only']):
        raise ValueError('A real human reviewer ID is required; synthetic UI checks are not study evidence')
    expected = manifest[role + '_ids']
    if value.get('sample_ids') != expected:
        raise ValueError('Sample IDs or their order do not match the frozen packet')
    cards = {r['review_id']: r for r in load(AUDIT / (role + '_blinded.json'))}
    decisions = value.get('decisions')
    if not isinstance(decisions, dict) or not set(decisions) <= cards.keys():
        raise ValueError('Invalid decisions')
    for uid, d in decisions.items():
        card = cards[uid]
        if d.get('card_sha256') != card['card_sha256']:
            raise ValueError('Source text hash mismatch: ' + uid)
        choice = d.get('choice')
        if choice not in card['option_labels'] + ['NONE', 'MULTIPLE', 'OTHER', 'UNCLEAR']:
            raise ValueError('Unsupported human label: ' + uid)
        if d.get('confidence') not in [None, '', 'high', 'medium', 'low'] or not d.get('saved_at'):
            raise ValueError('Invalid optional confidence or missing timestamp: ' + uid)
        evidence = d.get('evidence', '').strip()
        if evidence and cosmetic(evidence) not in cosmetic(card['response']):
            raise ValueError('Supporting quote not found in response: ' + uid)
        if choice == 'OTHER' and not d.get('answer', '').strip():
            raise ValueError('Missing out-of-option answer text: ' + uid)
    return value


def paired_stats(ids, left, right, weights):
    total = sum(weights[i] for i in ids)
    if not total:
        return {'n': 0, 'agreement': None, 'kappa': None}
    a, b = Counter(), Counter()
    agree = 0
    for i in ids:
        w = weights[i]
        a[str(left[i])] += w
        b[str(right[i])] += w
        agree += w * (left[i] == right[i])
    observed = agree / total
    expected = sum(a[k] * b[k] for k in set(a) | set(b)) / total ** 2
    return {'n': len(ids), 'weighted_n': total, 'agreement': observed,
            'kappa': (observed - expected) / (1 - expected) if expected < 1 else None}


def design_interval(ids, indicator, private, manifest):
    """Approximate design-based interval for the fixed response-corpus mean.

    Human primary review must be complete. Stratum random draws are independent;
    no assumption of independent generated responses is needed for this finite
    population estimate. Annotation uncertainty is not captured.
    """
    grouped = defaultdict(list)
    for uid in ids:
        grouped[private[uid]['stratum']].append(float(indicator[uid]))
    population = manifest['population_n']
    mean, variance = 0., 0.
    for s, v in grouped.items():
        N, n = manifest['strata'][s]['population_n'], len(v)
        assert n == manifest['strata'][s]['sample_n']
        p = sum(v) / n
        mean += N / population * p
        if n < N:
            assert n >= 2
            sample_variance = sum((x - p) ** 2 for x in v) / (n - 1)
            variance += (N / population) ** 2 * (1 - n / N) * sample_variance / n
    se = math.sqrt(variance)
    return {'estimate': mean, 'standard_error': se,
            'approximate_design_ci95': [max(0., mean - 1.96 * se), min(1., mean + 1.96 * se)],
            'caveat': 'Normal-approximation finite-population stratified-sampling interval; may be unreliable with sparse or unanimous stratum labels. Does not capture human annotation error.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', type=Path, required=True, help='Actual primary reviewer JSON export')
    parser.add_argument('--secondary', type=Path, help='Actual independent secondary export')
    args = parser.parse_args()
    manifest = load(AUDIT / 'manifest.json')
    for name, expected in manifest['source_hashes'].items():
        actual = hashlib.sha256((AUDIT.parent / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Frozen scoring source changed: ' + name)
    private = load(AUDIT / 'analyst_private.json')
    primary = validate(args.primary, manifest, 'primary')
    secondary = validate(args.secondary, manifest, 'secondary') if args.secondary else None
    if secondary and secondary['reviewer'].casefold() == primary['reviewer'].casefold():
        raise ValueError('Secondary review must be a different independent human reviewer')
    ids = list(primary['decisions'])
    report = {'schema': 'spine-science-human-audit-results-v3', 'packet_id': manifest['packet_id'],
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'primary_reviewer': primary['reviewer'], 'primary_completed': len(ids),
              'primary_expected': manifest['primary_n'],
              'status': 'complete_primary' if len(ids) == manifest['primary_n'] else 'incomplete',
              'scope': 'Human response-label audit, not human validation of task references or dataset-wide intervention validity. AI and benchmark labels remain unchanged.'}
    human_choice = {uid: d['choice'] for uid, d in primary['decisions'].items()}
    human_correct = {uid: correct(human_choice[uid], private[uid]['target_option']) for uid in ids}
    ai_choice = {uid: private[uid]['final_choice'] for uid in ids}
    ai_correct = {uid: private[uid]['final_correct'] for uid in ids}
    unit = dict.fromkeys(ids, 1.)
    report['sample_unweighted_descriptive'] = {
        'extraction': paired_stats(ids, human_choice, ai_choice, unit),
        'correctness': paired_stats(ids, human_correct, ai_correct, unit),
        'warning': 'Oversampled disputed and unresolved records; raw sample agreement is not corpus agreement.'}
    report['by_status'] = {cat: {
        'extraction': paired_stats([i for i in ids if private[i]['category'] == cat], human_choice, ai_choice, unit),
        'correctness': paired_stats([i for i in ids if private[i]['category'] == cat], human_correct, ai_correct, unit)}
        for cat in ['agreed', 'adjudicated', 'unresolved']}
    report['disagreements'] = [{'review_id': uid, 'model': private[uid]['model'], 'phase': private[uid]['phase'],
                                'seed_id': private[uid]['seed_id'], 'source': private[uid]['origin'],
                                'human_choice': human_choice[uid], 'ai_choice': ai_choice[uid],
                                'human_correct': human_correct[uid], 'ai_correct': ai_correct[uid],
                                'human_evidence': primary['decisions'][uid].get('evidence', ''),
                                'human_reason': primary['decisions'][uid].get('reason', '')}
                               for uid in ids if human_choice[uid] != ai_choice[uid]]
    if report['status'] == 'complete_primary':
        weights = {uid: private[uid]['analysis_weight'] for uid in ids}
        assert abs(sum(weights.values()) - manifest['population_n']) < 1e-6
        report['population_weighted'] = {
            'extraction': paired_stats(ids, human_choice, ai_choice, weights),
            'correctness': paired_stats(ids, human_correct, ai_correct, weights),
            'mutually_resolved_correctness': paired_stats([i for i in ids if human_correct[i] is not None and ai_correct[i] is not None], human_correct, ai_correct, weights),
            'human_unresolved_rate': sum(weights[i] for i in ids if human_correct[i] is None) / sum(weights.values()),
            'correctness_agreement_interval': design_interval(ids, {i: human_correct[i] == ai_correct[i] for i in ids}, private, manifest)}
        report['individual_judges_weighted'] = {}
        report['by_status_weighted'] = {cat: {
            'extraction': paired_stats([i for i in ids if private[i]['category'] == cat], human_choice, ai_choice, weights),
            'correctness': paired_stats([i for i in ids if private[i]['category'] == cat], human_correct, ai_correct, weights)}
            for cat in ['agreed', 'adjudicated', 'unresolved']}
        for judge in ['luna', 'flash_lite']:
            choices = {uid: private[uid][judge + '_choice'] for uid in ids}
            correctness = {uid: correct(choices[uid], private[uid]['target_option']) for uid in ids}
            report['individual_judges_weighted'][judge] = {
                'extraction': paired_stats(ids, human_choice, choices, weights),
                'correctness': paired_stats(ids, human_correct, correctness, weights)}
    else:
        report['population_weighted'] = None
        report['warning'] = 'Primary review incomplete. No corpus agreement estimate is reported.'
    if secondary:
        overlap = [i for i in ids if i in secondary['decisions']]
        second_choice = {i: secondary['decisions'][i]['choice'] for i in overlap}
        second_correct = {i: correct(second_choice[i], private[i]['target_option']) for i in overlap}
        report['human_human'] = {'reviewers': [primary['reviewer'], secondary['reviewer']],
                                'secondary_completed': len(secondary['decisions']), 'expected_overlap': manifest['secondary_n'],
                                'extraction': paired_stats(overlap, human_choice, second_choice, unit),
                                'correctness': paired_stats(overlap, human_correct, second_correct, unit),
                                'disagreement_ids': [i for i in overlap if human_choice[i] != second_choice[i]],
                                'scope': 'Descriptive agreement on the selected overlap; no independent-human consensus or adjudication is inferred.'}
    exports = AUDIT / 'received_exports'
    exports.mkdir(exist_ok=True)
    for role, path in [('primary', args.primary), ('secondary', args.secondary)]:
        if path:
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            (exports / (role + '_' + digest + '.json')).write_bytes(data)
    (AUDIT / 'human_agreement_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['status', 'primary_completed', 'primary_expected', 'population_weighted']}, indent=2))


if __name__ == '__main__':
    main()
