"""Build blank questionnaires from released stimuli; never load reviewer decisions."""
import argparse
import importlib.util
import json
import re
from verify_release import ROOT, load_verified, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['dataset', 'science'], required=True)
    args = parser.parse_args()
    output = ROOT / 'review_workspace' / args.kind
    output.mkdir(parents=True, exist_ok=True)
    if args.kind == 'dataset':
        path = ROOT / 'software/valid_corrections/build_dataset_validity_review_v3.py'
        spec = importlib.util.spec_from_file_location('dataset_review_builder', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.SOURCE = ROOT / 'data/spine_dataset_persona_validated_v3.json'
        module.OUT = output
        module.main()
    else:
        index, _ = load_verified()
        lookup = {r['blind_id']: r for r in index}
        ids = json.loads((ROOT / 'reproduce/science_review_ids.json').read_text())
        cards = []
        for uid in ids:
            row = lookup[uid]
            assert row['domain'] == 'logic'
            f = row['judge_input']
            card = {'review_id': uid, 'phase': row['phase'], 'task': f['task'],
                    'original_task': f.get('original_task', f['task']),
                    'initial_answer': f.get('initial_answer', ''),
                    'followup_message': f.get('followup_message', ''),
                    'response': f['final_response'],
                    'option_labels': list(dict.fromkeys(re.findall(r'\(([A-E1-5])\)', f['task'].split('Options:', 1)[-1])))}
            assert len(card['option_labels']) >= 2
            card['card_sha256'] = sha(json.dumps(card, ensure_ascii=False, sort_keys=True).encode())
            cards.append(card)
        assert len(cards) == 100
        packet = {'schema': 'spine-public-science-review', 'packet_id': sha(json.dumps(ids).encode()), 'role': 'primary', 'cards': cards}
        template = (ROOT / 'software/model_runs/science_human_review_template.html').read_text(encoding='utf-8')
        encoded = json.dumps(packet, ensure_ascii=False).replace('<', '\\u003c')
        (output / 'primary_review.html').write_text(template.replace('__PACKET_JSON__', encoded), encoding='utf-8')
        print('Built 100 blank science review items, without AI labels or human decisions.')
    print('Open:', output / 'primary_review.html')


if __name__ == '__main__':
    main()
