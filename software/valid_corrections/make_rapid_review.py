"""Freeze a two-to-three-hour qualitative SPINE evidence spot check."""
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'spine_dataset_ground_truth_corrected.json'
OUT = HERE / 'rapid_evidence_review'
SEED = 20260927
PERSONAS = {
    'math': ('sophistry', 'authority', 'consensus', 'victim'),
    'logic': ('sophistry', 'authority', 'doubter', 'bureaucrat'),
    'coding': ('consensus', 'victim', 'doubter', 'bureaucrat'),
}


def build():
    raw = SOURCE.read_bytes()
    rows = json.loads(raw)
    rng = random.Random(SEED)
    selected = []
    for domain, personas in PERSONAS.items():
        pilots = sorted(x['id'] for x in rows if x['domain'] == domain and x.get('valid_corrective_intervention'))
        candidates = sorted(x['id'] for x in rows if x['domain'] == domain and not x.get('valid_corrective_intervention'))
        ids = rng.sample(pilots, 2) + rng.sample(candidates, 2)
        order = list(personas)
        rng.shuffle(order)
        selected.extend({'id': item_id, 'domain': domain, 'persona': persona,
                         'origin': 'authored_pilot' if item_id in pilots else 'candidate'}
                        for item_id, persona in zip(ids, order))
    assert len(selected) == 12 and len({x['id'] for x in selected}) == 12
    OUT.mkdir(exist_ok=True)
    manifest = {
        'purpose': 'Qualitative spot check within a two-to-three-hour human budget; not a prevalence estimate or full validation.',
        'source_sha256': hashlib.sha256(raw).hexdigest(),
        'random_seed': SEED,
        'seed_count': len(selected),
        'attack_count': len(selected),
        'human_correction_count': sum(x['domain'] != 'coding' for x in selected),
        'machine_only_code_correction_count': sum(x['domain'] == 'coding' for x in selected),
        'total_human_decisions': len(selected) + sum(x['domain'] != 'coding' for x in selected),
        'selection': 'Two authored pilots and two correction candidates per domain; one prespecified persona attack per selected seed. Each persona appears twice across two domains.',
        'selected': selected,
        'status': 'unreviewed',
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'selected'}, indent=2))


if __name__ == '__main__':
    build()
