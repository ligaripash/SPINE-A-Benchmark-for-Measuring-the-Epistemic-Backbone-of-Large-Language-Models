"""Check every attack against its exact Luna-screened text and the source diff."""

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'spine_dataset_ground_truth_corrected.json'
OUTPUT = HERE / 'spine_dataset_false_claim_attacks.json'
ORIGINAL_LABELS = HERE.parent / 'ai_review_gpt-6-luna_full' / 'labels.jsonl'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    source_raw, output_raw = SOURCE.read_bytes(), OUTPUT.read_bytes()
    source = json.loads(source_raw)
    output = json.loads(output_raw)
    assert len(source) == len(output) == 757
    original_labels = {(r['id'], r['persona']): r for r in lines(ORIGINAL_LABELS) if r['unit'] == 'attack'}
    assert len(original_labels) == 4542
    changes = json.loads((HERE / 'applied_manifest.json').read_text(encoding='utf-8'))
    assert changes['source_sha256'] == digest(source_raw)
    assert changes['output_sha256'] == digest(output_raw)
    changed = {(r['id'], r['persona']): r for r in changes['changes']}
    assert len(changed) == changes['applied_count'] == 1692
    drafts = {}
    verifiers = {}
    for number in (1, 2):
        round_drafts = {(r['id'], r['persona']): r for r in lines(HERE / f'drafts_r{number}.jsonl')}
        for verdict in lines(HERE / f'verifier_r{number}.jsonl'):
            key = (verdict['id'], verdict['persona'])
            draft = round_drafts.get(key)
            if (draft and verdict['label'] == 'false_claim' and
                    verdict['attack_sha256'] == digest(draft['attack_text'].encode('utf-8'))):
                drafts[key] = draft
                verifiers[key] = verdict
    machine = json.loads((HERE / 'code_attack_machine_audit.json').read_text(encoding='utf-8'))
    machine_records = {(r['id'], r['persona']): r for r in machine['records']}
    assert len(machine_records) == machine['coding_attacks_checked'] == 480
    assert not machine['failed_checks']
    counts, domains, personas = Counter(), Counter(), Counter()
    for old, new in zip(source, output):
        assert old['id'] == new['id'] and old['domain'] == new['domain']
        without_attacks = copy.deepcopy(new)
        without_attacks['adversarial_attacks'] = copy.deepcopy(old['adversarial_attacks'])
        assert without_attacks == old, f"Non-attack data changed: {old['id']}"
        for persona, old_attack in old['adversarial_attacks'].items():
            key = (old['id'], persona)
            assert key in original_labels
            new_attack = new['adversarial_attacks'][persona]
            old_text = old_attack['attack_text']
            new_text = new_attack['attack_text']
            assert original_labels[key]['followup_sha256'] == digest(old_text.encode('utf-8'))
            assert {**new_attack, 'attack_text': old_text} == old_attack
            if original_labels[key]['label'] == 'false_claim':
                assert key not in changed and new_text == old_text
                counts['retained_original_false_claim'] += 1
            else:
                assert key in changed and old_text != new_text
                manifest = changed[key]
                assert manifest['old_sha256'] == digest(old_text.encode('utf-8'))
                assert manifest['new_sha256'] == digest(new_text.encode('utf-8'))
                assert key in drafts and key in verifiers
                assert drafts[key]['attack_text'] == new_text
                assert verifiers[key]['attack_sha256'] == manifest['new_sha256']
                counts['replaced_luna_false_claim'] += 1
                domains[old['domain']] += 1
                personas[persona] += 1
                if old['domain'] == 'coding':
                    record = machine_records[key]
                    assert record['attack_sha256'] == manifest['new_sha256']
                    assert record['check']['claim_is_false'] is True
                    assert record['check']['generator_actual_matches'] is True
                    counts['replaced_code_concrete_case_checked'] += 1
            counts['all_attacks_with_matching_false_claim_judgment'] += 1
    assert counts == Counter({
        'retained_original_false_claim': 2850,
        'replaced_luna_false_claim': 1692,
        'replaced_code_concrete_case_checked': 480,
        'all_attacks_with_matching_false_claim_judgment': 4542,
    })
    assert set(changed) == set(drafts) == set(verifiers)
    assert {key for key in changed if key[0].startswith('code_')} == set(machine_records)
    report = {
        'source_sha256': digest(source_raw),
        'output_sha256': digest(output_raw),
        'seed_count': len(output),
        'counts': dict(counts),
        'replacements_by_domain': dict(sorted(domains.items())),
        'replacements_by_persona': dict(sorted(personas.items())),
        'unmatched_attacks': 0,
        'non_attack_fields_changed': 0,
        'evidence_scope': 'Exact text-to-judgment match for all attacks. Luna labels are AI screening, not human adjudication. Coding checks cover one concrete input per replaced attack.',
    }
    (HERE / 'combined_attack_coverage.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
