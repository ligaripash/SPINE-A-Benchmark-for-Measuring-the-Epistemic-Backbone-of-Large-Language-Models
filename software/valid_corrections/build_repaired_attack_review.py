"""Make a separate blind human-review page for the repaired attack dataset."""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPAIR = HERE / 'attack_false_claim_repair_v2'
BASE_HTML = HERE / 'full_dataset_review/review_spine.html'
DATASET = REPAIR / 'spine_dataset_false_claim_attacks.json'
OUT = REPAIR / 'review_repaired_attacks.html'
SEED = 20260928
PERSONAS = {
    'math': (('sophistry', 'pilot'), ('consensus', 'pilot'),
             ('authority', 'candidate'), ('victim', 'candidate')),
    'logic': (('authority', 'pilot'), ('doubter', 'pilot'),
              ('sophistry', 'candidate'), ('bureaucrat', 'candidate')),
}


def main():
    html = BASE_HTML.read_text(encoding='utf-8')
    match = re.search(r'(<script type="application/json" id="dataset">)(.*?)(</script>)',
                      html, flags=re.S)
    if not match:
        raise SystemExit('Embedded review dataset not found')
    embedded = json.loads(match.group(2))
    raw = DATASET.read_bytes()
    repaired = {item['id']: item for item in json.loads(raw)}
    changed = {(row['id'], row['persona']) for row in json.loads(
        (REPAIR / 'applied_manifest.json').read_text(encoding='utf-8'))['changes']}
    rng = random.Random(SEED)
    selected = []
    used = set()
    for domain, pairs in PERSONAS.items():
        for persona, origin in pairs:
            choices = sorted(item_id for item_id, item in repaired.items()
                             if item['domain'] == domain and (item_id, persona) in changed
                             and bool(item.get('valid_corrective_intervention')) == (origin == 'pilot')
                             and item_id not in used)
            item_id = rng.choice(choices)
            assert item_id in choices
            used.add(item_id)
            selected.append({'id': item_id, 'domain': domain, 'persona': persona,
                             'origin': 'authored_pilot' if origin == 'pilot' else 'candidate'})
    assert len(selected) == len(used) == 8
    rapid_attacks = {row['id']: row['persona'] for row in selected}
    count = 0
    for item in embedded['items']:
        item['inRapid'] = item['id'] in rapid_attacks
        item['rapidAttack'] = rapid_attacks.get(item['id'], '')
        if item['id'] not in repaired:
            continue
        for persona, attack in repaired[item['id']]['adversarial_attacks'].items():
            item['attacks'][persona] = attack['attack_text']
            if (item['id'], persona) in changed:
                item['attackFlags'][persona] = []
                count += 1
    if count != 1692:
        raise SystemExit(f'Expected 1692 replacements in page, got {count}')
    embedded['corrected_sha256'] = hashlib.sha256(raw).hexdigest()
    embedded['attack_repair_count'] = count
    embedded['rapid_seed_count'] = len(selected)
    new_data = json.dumps(embedded, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    html = html[:match.start(2)] + new_data + html[match.end(2):]
    html = html.replace('<title>SPINE dataset review</title>',
                        '<title>SPINE repaired attack review</title>')
    html = html.replace('<h1>SPINE dataset review</h1>',
                        '<h1>SPINE repaired attack review</h1>')
    html = html.replace('Saved locally in this browser. Export a backup.',
                        'Repaired-attack version. Saved locally in this browser. Export a backup.')
    html = html.replace("const storeKey = 'spine-review-' + dataset.source_sha256;",
                        "const storeKey = 'spine-review-repaired-' + dataset.corrected_sha256;")
    html = html.replace("(payload.review_mode==='evidence'&&payload.corrected_sha256!==dataset.corrected_sha256)",
                        "(payload.corrected_sha256!==dataset.corrected_sha256)")
    html = html.replace('<option value="coding">Code</option>', '')
    html = html.replace('Quick spot check (12 seeds)', 'Quick spot check (8 non-code seeds)')
    html = html.replace('Extended sample (119 seeds)', 'Extended sample (non-code seeds)')
    html = html.replace('All retained seeds (757)', 'All retained non-code seeds')
    html = html.replace("return !!item.correction&&!(item.domain==='coding'&&$('review-scope').value==='rapid');",
                        "return !!item.correction&&item.domain!=='coding';")
    html = html.replace("return (s==='rapid'?x.inRapid:s==='sampled'?x.inSample:true)&&",
                        "return x.domain!=='coding'&&(s==='rapid'?x.inRapid:s==='sampled'?x.inSample:true)&&")
    html = html.replace('Accept only false claims or unsupported pressure against the original correct answer.',
                        'Accept only a false claim against the original correct answer.')
    html = html.replace('This is a 12-seed qualitative spot check. Four code corrections have machine checks only; the 20 human decisions are 12 attacks and 8 math/reasoning corrections.',
                        'This is an 8-seed qualitative spot check with 16 human decisions: 8 attacks and 8 corrections. Code is excluded from every human-review scope.')
    html = html.replace('The extended 119-seed sample is selected.',
                        'The extended non-code sample is selected.')
    html = html.replace('ratings:state.ratings};const blob=',
                        "ratings:Object.fromEntries(Object.entries(state.ratings).filter(([key])=>dataById[key.split(':')[1]]?.domain!=='coding'))};const blob=")
    OUT.write_text(html, encoding='utf-8')
    rapid = json.loads((HERE / 'rapid_evidence_review/manifest.json').read_text(encoding='utf-8'))
    rapid['source_sha256'] = embedded['corrected_sha256']
    rapid['random_seed'] = SEED
    rapid['selected'] = selected
    rapid['seed_count'] = len(selected)
    rapid['attack_count'] = len(selected)
    rapid['human_correction_count'] = len(selected)
    rapid['machine_only_code_correction_count'] = 0
    rapid['total_human_decisions'] = 2 * len(selected)
    rapid['selection'] = ('Eight repaired attacks from math and logic only: four seeds per domain, '
                          'two authored corrections and two candidates per domain. '
                          'No code item requires human review.')
    rapid['status'] = 'unreviewed_repaired_attacks'
    rapid['note'] = 'New prespecified qualitative sample of repaired attacks; not a prevalence estimate.'
    (REPAIR / 'rapid_review_manifest.json').write_text(
        json.dumps(rapid, indent=2) + '\n', encoding='utf-8')
    sample = json.loads((HERE / 'sampled_evidence_review/manifest.json').read_text(encoding='utf-8'))
    sample['population_sha256'] = embedded['corrected_sha256']
    sample['note'] = ('Same preselected seed sample; the repaired human interface '
                      'shows only non-code seeds. Code evidence is machine-checked separately.')
    (REPAIR / 'sampled_review_manifest.json').write_text(
        json.dumps(sample, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {OUT}; {count} repaired attacks, dataset sha256 {embedded["corrected_sha256"]}')


if __name__ == '__main__':
    main()
