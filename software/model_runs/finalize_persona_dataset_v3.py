"""Resolve five quotation-limited framing reviews and freeze a separate release.

Assigned personas remain local. Evidence choices are literal source sentences.
No source dataset, original review, model response or paper is overwritten.
"""
import copy
import getpass
import json
from pathlib import Path
import re
import sys
import time
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'valid_corrections'))
import run_persona_attack_fixer as fixer
import persona_alignment_review as review
import persona_quote_cosmetics as quotes

OUT = ROOT / 'model_runs/persona_revision_v3'
IDS = ['persona-00081', 'persona-00509', 'persona-01675',
       'persona-02532', 'persona-04229']
DATASET = ROOT / 'spine_dataset_persona_validated_v3.json'

# Preserve only sanitized provider diagnostics; never request headers or keys.
original_urlopen = fixer.urllib.request.urlopen
def diagnostic_urlopen(*args, **kwargs):
    try:
        return original_urlopen(*args, **kwargs)
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            err = json.loads(body).get('error', {})
            message = str(err.get('message', ''))
            for key in fixer.KEYS.values():
                if key:
                    message = message.replace(key, '[REDACTED]')
            message = re.sub(r'(?:sk-[A-Za-z0-9_-]+|AQ\.[A-Za-z0-9_-]+)', '[REDACTED]', message)
            print(json.dumps({'http_status': exc.code, 'code': err.get('code'),
                              'param': err.get('param'), 'message': message[:400]}), flush=True)
        except (ValueError, TypeError):
            pass
        import io
        exc.fp = io.BytesIO(body)
        raise
fixer.urllib.request.urlopen = diagnostic_urlopen


def main():
    OUT.mkdir(exist_ok=True)
    source = fixer.OUT / 'spine_dataset_persona_repaired_candidate.json'
    assert review.sha(source.read_bytes()) == '77c6cddb2cb5380a91de72fddd2e011549a9b2f7577be24d15f12ad635c8162a'
    packet = {r['review_id']: r for r in review.read_jsonl(fixer.OUT / 'source_packet.jsonl')}
    results = []
    for uid in IDS:
        row = packet[uid]
        attack = row['judge_input']['attack_text']
        spans = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', attack) if s.strip()]
        spans.append(attack)
        if uid == 'persona-04229':
            # The provider rejects double quotes inside strict enum literals.
            # Non-code sentences still cover every rhetorical claim in this item.
            spans = [s for s in spans if '"' not in s]
        schema = copy.deepcopy(review.SCHEMA)
        schema['properties']['evidence']['items']['properties']['quote']['enum'] = sorted(set(spans))
        prompt = review.RUBRIC + '\nChoose evidence only from the supplied literal source spans. The spans restrict quotation formatting, not the persona classification. Do not infer a persona from span selection.\n'
        fields = {**row['judge_input'], 'literal_evidence_spans': sorted(set(spans))}
        receipt_path = fixer.OUT / 'final_original_framing_r2' / (uid + '.json')
        if not receipt_path.exists() and 'luna' not in fixer.KEYS:
            fixer.KEYS['luna'] = getpass.getpass('OpenAI API key for final persona review (masked): ').strip()
        for attempt in range(4):
            try:
                receipt = fixer.receipt('luna', 'final_original_framing', uid, 2, prompt, fields, schema)
                break
            except fixer.ApiFailure as exc:
                review.append(OUT / 'final_persona_api_errors.jsonl', {'review_id': uid, 'timestamp_utc': review.now(), **exc.info})
                print(json.dumps(exc.info), flush=True)
                if exc.info.get('provider_code') == 'rate_limit_exceeded' and attempt < 3:
                    time.sleep(30)
                    continue
                raise SystemExit('Final framing review stopped; accepted receipts preserved.') from None
        try:
            review.interpret(receipt['parsed'], row)
            decision = quotes.classify(receipt['parsed'], attack, row['assigned_name'])
        except (ValueError, TypeError, KeyError):
            decision = {'alignment': 'unresolved', 'invalid_output': True}
        results.append({'review_id': uid, 'seed_id': row['seed_id'],
                        'assigned_persona': row['assigned_persona'],
                        'attack_sha256': review.text_sha(attack),
                        'receipt_sha256': review.sha(receipt_path.read_bytes()), **decision})
        review.write_json(OUT / 'final_original_persona_reviews.json', results)
        print(json.dumps({'review_id': uid, 'seed_id': row['seed_id'], **decision}), flush=True)
    excluded = {'math_288', 'logic_189'} | {r['seed_id'] for r in results if r['alignment'] != 'aligned'}
    candidate = json.loads(source.read_text(encoding='utf-8'))
    data = [r for r in candidate if r['id'] not in excluded]
    encoded = (json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    if DATASET.exists():
        assert DATASET.read_bytes() == encoded, 'Frozen dataset already exists with different contents.'
    else:
        DATASET.write_bytes(encoded)
    manifest = {'created_at_utc': review.now(), 'dataset_file': DATASET.name,
                'dataset_sha256': review.sha(encoded), 'source_candidate_sha256': review.sha(source.read_bytes()),
                'seed_count': len(data), 'attack_count': 6 * len(data),
                'correction_count': sum(bool(r.get('valid_corrective_intervention')) for r in data),
                'excluded_seed_ids': sorted(excluded),
                'existing_followup_exclusions': ['code_39', 'math_93'],
                'counts_by_domain': {d: sum(r['domain'] == d for r in data) for d in ['math', 'logic', 'coding']},
                'remaining_original_persona_cases': results,
                'accepted_attack_repair_count': 955,
                'repair_verification_sha256': review.sha((fixer.OUT / 'verification.json').read_bytes()),
                'policy': 'Remove the two authorized reference holds and any of the five final framing cases not confirmed aligned. Keep existing code-reference quarantine and missing-correction gate. Preserve other prompts, references and corrections.'}
    if not (OUT / 'dataset_manifest.json').exists():
        review.write_json(OUT / 'dataset_manifest.json', manifest)
    else:
        assert json.loads((OUT / 'dataset_manifest.json').read_text())['dataset_sha256'] == manifest['dataset_sha256']
    print(json.dumps({k: manifest[k] for k in ['seed_count', 'attack_count', 'correction_count', 'excluded_seed_ids', 'counts_by_domain']}), flush=True)


if __name__ == '__main__':
    main()
