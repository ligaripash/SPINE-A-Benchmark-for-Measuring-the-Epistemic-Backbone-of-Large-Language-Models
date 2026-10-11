"""Freeze a stratified human-review sample for SPINE-authored branches.

All retained authored pilots are included. Draw 20 of the remaining correction
candidates per domain without replacement; keep all six attacks with each seed.
"""
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / 'spine_dataset_ground_truth_corrected.json'
OUTPUT = HERE / 'sampled_evidence_review'
SEED = 20260927
DOMAINS = ('math', 'logic', 'coding')


def build():
    raw = SOURCE.read_bytes()
    rows = json.loads(raw)
    assert len(rows) == 757
    rng = random.Random(SEED)
    sample_ids = []
    second_ids = []
    strata = {}
    for domain in DOMAINS:
        pilots = sorted(x['id'] for x in rows if x['domain'] == domain and x.get('valid_corrective_intervention'))
        candidates = sorted(x['id'] for x in rows if x['domain'] == domain and not x.get('valid_corrective_intervention'))
        drawn = rng.sample(candidates, 20)
        sample_ids.extend(pilots + drawn)
        second_ids.extend(rng.sample(pilots, 5) + rng.sample(drawn, 5))
        strata[domain] = {
            'pilot_population': len(pilots),
            'pilot_sample': len(pilots),
            'pilot_inclusion_probability': 1.0,
            'candidate_population': len(candidates),
            'candidate_sample': len(drawn),
            'candidate_inclusion_probability': 20 / len(candidates),
            'candidate_weight': len(candidates) / 20,
        }
    assert len(sample_ids) == len(set(sample_ids)) == 119
    assert len(second_ids) == len(set(second_ids)) == 30
    OUTPUT.mkdir(exist_ok=True)
    manifest = {
        'population_file': SOURCE.name,
        'population_sha256': hashlib.sha256(raw).hexdigest(),
        'random_seed': SEED,
        'population_seed_count': len(rows),
        'sample_seed_count': len(sample_ids),
        'sample_seed_ids': sample_ids,
        'primary_attack_count': 6 * len(sample_ids),
        'primary_correction_count': len(sample_ids),
        'primary_total_decisions': 7 * len(sample_ids),
        'second_seed_count': len(second_ids),
        'second_seed_ids': second_ids,
        'second_total_decisions': 7 * len(second_ids),
        'strata': strata,
        'selection': 'Census of retained authored pilots; independent simple random sample of 20 correction candidates per domain. All six attacks and one correction record per selected seed. Second review: five pilots and five sampled candidates per domain, plus flagged and disputed items.',
        'analysis': 'Estimate attack and correction validity using domain-by-origin stratum weights; retain six attacks together when estimating uncertainty. The pilot stratum is a census. Report targeted rechecks separately.',
        'status': 'unreviewed',
    }
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith('_ids')}, indent=2))


if __name__ == '__main__':
    build()
