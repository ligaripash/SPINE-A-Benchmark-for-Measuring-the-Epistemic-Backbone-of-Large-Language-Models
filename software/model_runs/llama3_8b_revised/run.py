"""Checkpointed Llama 3 (8B) run on the revised false-claim SPINE dataset.

Reuse is permitted only when model digest, complete conversation, and reference
dataset text match exactly. All correction candidates are generated, but their
status is retained for separate, explicitly exploratory scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATASET = ROOT / 'spine_dataset_false_claim_attacks.json'
PRIOR = ROOT / 'model_runs' / 'llama3_8b'
MODEL = 'llama3:latest'
API = 'http://127.0.0.1:11434'
PERSONAS = ('sophistry', 'authority', 'consensus', 'victim', 'doubter', 'bureaucrat')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def append(path, value):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + '\n')
        handle.flush()


def post(path, payload, timeout=600):
    request = urllib.request.Request(API + path, data=json.dumps(payload).encode('utf-8'),
                                     headers={'Content-Type': 'application/json'}, method='POST')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))


def model_digest():
    with urllib.request.urlopen(API + '/api/tags', timeout=10) as response:
        tags = json.load(response)
    matches = [item['digest'] for item in tags['models'] if item['name'] == MODEL]
    if len(matches) != 1:
        raise RuntimeError('Expected exactly one local llama3:latest model')
    return matches[0]


def correction(item):
    if item.get('valid_corrective_intervention'):
        return item['valid_corrective_intervention']['intervention_text']
    return (item.get('correction_candidate_for_review') or {}).get('candidate', {}).get('intervention_text')


def messages(item, anchor, branch):
    followup = (item['adversarial_attacks'][branch]['attack_text'] if branch in PERSONAS
                else correction(item))
    if not followup:
        raise ValueError(f'No {branch} text for {item["id"]}')
    return [
        {'role': 'user', 'content': item['question_data']['prompt']},
        {'role': 'assistant', 'content': anchor['response']},
        {'role': 'user', 'content': followup},
    ]


def prepare():
    dataset_raw = DATASET.read_bytes()
    items = json.loads(dataset_raw)
    prior_manifest = json.loads((PRIOR / 'manifest.json').read_text(encoding='utf-8'))
    current_digest = model_digest()
    if current_digest != prior_manifest['model_digest']:
        raise RuntimeError('Model digest changed; baseline cannot be reused')
    base = {r['id']: r for r in rows(PRIOR / 'base.jsonl') if 'error' not in r}
    base_scores = {r['id']: r for r in rows(PRIOR / 'base_scores.jsonl')}
    if len(base) != 757 or len(base_scores) != 757:
        raise RuntimeError('Baseline coverage incomplete')
    for item in items:
        if base[item['id']]['messages'] != [{'role': 'user', 'content': item['question_data']['prompt']}]:
            raise RuntimeError(f'Baseline prompt changed: {item["id"]}')
    eligible = [item['id'] for item in items if base_scores[item['id']]['correct'] is True
                and item['id'] != 'code_39']
    followup_ids = [item['id'] for item in items]
    old_attacks = {}
    for path in sorted(PRIOR.glob('attacks_*.jsonl')):
        for row in rows(path):
            if 'error' not in row:
                old_attacks[(row['id'], row['branch'])] = (row, path.name)
    old_corrections = {r['id']: r for r in rows(PRIOR / 'corrections.jsonl') if 'error' not in r}
    attack_reused, correction_reused = [], []
    for item in items:
        ident = item['id']
        for persona in PERSONAS:
            pair = old_attacks.get((ident, persona))
            if pair and pair[0]['messages'] == messages(item, base[ident], persona):
                copy = dict(pair[0])
                copy['reused_from'] = str(PRIOR / pair[1])
                attack_reused.append(copy)
        candidate = old_corrections.get(ident)
        if candidate and candidate['messages'] == messages(item, base[ident], 'correction'):
            copy = dict(candidate)
            copy['reused_from'] = str(PRIOR / 'corrections.jsonl')
            copy['correction_status'] = item['valid_correction_status']
            correction_reused.append(copy)
    manifest = {
        'dataset': str(DATASET),
        'dataset_sha256': sha(dataset_raw),
        'model': MODEL,
        'model_digest': current_digest,
        'prior_run_manifest_sha256': sha((PRIOR / 'manifest.json').read_bytes()),
        'prior_baseline_sha256': sha((PRIOR / 'base.jsonl').read_bytes()),
        'generation': {'temperature': 0, 'stream': False,
                       'num_predict_math_logic': 256, 'num_predict_coding': 512},
        'eligible_policy': 'Accuracy metrics require a correct baseline; raw follow-ups cover all retained seeds.',
        'eligible_count': len(eligible),
        'followup_seed_count': len(followup_ids),
        'expected_attack_responses': len(followup_ids) * 6,
        'expected_correction_responses': sum(bool(correction(item)) for item in items),
        'reused_attack_responses': len(attack_reused),
        'reused_correction_responses': len(correction_reused),
        'correction_warning': 'Generated correction candidate responses are exploratory; most interventions lack independent validation.',
        'prepared_utc': datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = HERE / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        comparable = {k: v for k, v in manifest.items() if k != 'prepared_utc'}
        stable = ('dataset_sha256', 'model_digest', 'prior_run_manifest_sha256',
                  'prior_baseline_sha256', 'reused_attack_responses', 'reused_correction_responses')
        if any(previous.get(key) != manifest[key] for key in stable):
            raise RuntimeError('Existing run manifest does not match this dataset/model')
        manifest['prepared_utc'] = previous['prepared_utc']
        manifest['expanded_to_all_seeds_utc'] = previous.get('expanded_to_all_seeds_utc', datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    outputs = {
        'base.jsonl': PRIOR / 'base.jsonl',
        'base_scores.jsonl': PRIOR / 'base_scores.jsonl',
    }
    for name, source in outputs.items():
        destination = HERE / name
        if destination.exists() and sha(destination.read_bytes()) != sha(source.read_bytes()):
            raise RuntimeError(f'Existing {name} differs from prior run')
        if not destination.exists():
            shutil.copyfile(source, destination)
    (HERE / 'eligible_ids.json').write_text(json.dumps(eligible, indent=2) + '\n', encoding='utf-8')
    (HERE / 'followup_ids.json').write_text(json.dumps(followup_ids, indent=2) + '\n', encoding='utf-8')
    for name, records in (('attacks_reused.jsonl', attack_reused),
                          ('corrections_reused.jsonl', correction_reused)):
        path = HERE / name
        content = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records)
        if path.exists() and path.read_text(encoding='utf-8') != content:
            raise RuntimeError(f'Existing {name} does not match exact conversation reuse')
        if not path.exists():
            path.write_text(content, encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('dataset_sha256', 'model_digest', 'eligible_count',
            'expected_attack_responses', 'expected_correction_responses',
            'reused_attack_responses', 'reused_correction_responses')}, indent=2), flush=True)


def generate(phase, workers, limit):
    manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
    if sha(DATASET.read_bytes()) != manifest['dataset_sha256'] or model_digest() != manifest['model_digest']:
        raise RuntimeError('Dataset or local model changed since preparation')
    items = json.loads(DATASET.read_text(encoding='utf-8'))
    followup_ids = set(json.loads((HERE / 'followup_ids.json').read_text(encoding='utf-8')))
    base = {r['id']: r for r in rows(HERE / 'base.jsonl')}
    prefix = 'attacks' if phase == 'attack' else 'corrections'
    output = HERE / f'{prefix}_new.jsonl'
    reused = {(r['id'], r['branch']) for r in rows(HERE / f'{prefix}_reused.jsonl')}
    done = {(r['id'], r['branch']) for r in rows(output) if 'error' not in r}
    pending = []
    for item in items:
        if item['id'] not in followup_ids:
            continue
        for branch in (PERSONAS if phase == 'attack' else ('correction',)):
            key = (item['id'], branch)
            if key in reused or key in done or (phase == 'correction' and not correction(item)):
                continue
            pending.append((item, branch))
    if limit:
        pending = pending[:limit]
    print(f'{phase}: {len(pending)} requests pending for this invocation', flush=True)

    def one(item, branch):
        start = time.monotonic()
        conversation = messages(item, base[item['id']], branch)
        options = {'temperature': 0, 'num_predict': 512 if item['domain'] == 'coding' else 256}
        response = post('/api/chat', {'model': MODEL, 'messages': conversation,
                                      'stream': False, 'options': options})
        row = {'id': item['id'], 'domain': item['domain'], 'branch': branch,
               'messages': conversation, 'response': response['message']['content'],
               'done_reason': response.get('done_reason'), 'eval_count': response.get('eval_count'),
               'eval_duration_ns': response.get('eval_duration'),
               'wall_seconds': round(time.monotonic() - start, 3),
               'timestamp_utc': datetime.now(timezone.utc).isoformat(),
               'dataset_sha256': manifest['dataset_sha256']}
        if phase == 'correction':
            row['correction_status'] = item['valid_correction_status']
        return row

    failures = Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, item, branch): (item['id'], branch) for item, branch in pending}
        for index, future in enumerate(as_completed(futures), 1):
            ident, branch = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                failures[type(exc).__name__] += 1
                row = {'id': ident, 'branch': branch, 'error': repr(exc),
                       'timestamp_utc': datetime.now(timezone.utc).isoformat()}
            append(output, row)
            if index % 20 == 0 or index == len(pending):
                print(f'{phase}: {index}/{len(pending)}; latest={ident}/{branch}; errors={sum(failures.values())}', flush=True)
    if failures:
        raise RuntimeError(f'{phase}: {dict(failures)} failures; rerun to retry checkpointed errors')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('prepare', 'attack', 'correction'), required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        raise SystemExit('Workers must be 1–4')
    if args.phase == 'prepare':
        prepare()
    else:
        generate(args.phase, args.workers, args.limit)
