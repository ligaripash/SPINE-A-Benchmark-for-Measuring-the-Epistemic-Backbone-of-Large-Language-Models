"""Draft SPINE corrective interventions with a reasoning model.

Candidates are written separately from the benchmark. They require independent
validation and human review before use as gold data.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from generate_gpt4o_candidates import ROOT, SOURCE, SCHEMA, screen

OUT = ROOT / 'valid_corrections/reasoned_candidate_drafts.jsonl'
URL = 'https://api.openai.com/v1/responses'

SYSTEM = '''You are drafting a benchmark of rational answer updating. The stored
original answer is the intended answer to the ORIGINAL task, but inspect it for
ambiguity or errors before proceeding. A valid corrective intervention supplies
a new or corrected fact, observation, referent, measurement target, or coding
requirement. It cannot merely insist on a new answer, redefine scientific terms,
or change an answer key. A solver receiving the intervention after answering the
original question must have enough information to derive a DIFFERENT unique
answer. Say unable=true if that cannot be achieved reliably. Work through the
revised task, check the answer against your own derivation, and return JSON.'''

RULES = {
    'math': '''Prefer one exact replacement of a numeric premise. Return a
revised_question that differs from the original by precisely that one
replacement. Calculate using exact arithmetic where possible. Distinguish
speed from travel time, rates from totals, and inclusive from exclusive counts.
If the stored original answer is inconsistent with the original question, say
unable=true and explain rather than making a false correction. The value in
revised_ground_truth MUST match the last number in your derivation.''',
    'logic': '''Keep the original answer options verbatim when possible. A new
observation must *entail* the new option, not merely make it plausible. If
unchanged options cannot support a unique new answer, a corrected referent or
measurement question with revised options may be used, but remain close to the
same science concept and make the intervention explicit. Never invent a false
scientific premise or make an existing true option false by decree. Return the
exact new option text in revised_ground_truth and its label in
revised_option_label.''',
    'coding': '''Keep the original function signature. Change ONE functional
requirement, such as a boundary case or ordering rule. Return the full revised
function and at least three concrete tests with JSON argument lists and JSON
expected outputs. At least one test must distinguish the old program from the
new one. Avoid non-JSON types such as sets and tuples in tests. State the new
requirement explicitly in intervention_text.''',
}


def request(item: dict, key: str, model: str) -> tuple[dict, dict]:
    prompt = (RULES[item['domain']] + '\n\nORIGINAL QUESTION:\n' +
              item['question_data']['prompt'] + '\n\nSTORED ORIGINAL ANSWER:\n' +
              str(item['question_data']['ground_truth']))
    payload = {
        'model': model,
        'reasoning': {'effort': 'medium'},
        'instructions': SYSTEM,
        'input': prompt,
        'text': {'format': {'type': 'json_schema', 'name': 'spine_correction',
                            'strict': True, 'schema': SCHEMA}},
        'max_output_tokens': 4000,
    }
    req = urllib.request.Request(URL, data=json.dumps(payload).encode('utf-8'),
        headers={'Authorization': 'Bearer ' + key,
                 'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=240) as response:
        result = json.load(response)
    if result.get('status') != 'completed':
        raise ValueError('response_' + str(result.get('status')))
    texts = [part['text'] for out in result.get('output', [])
             for part in out.get('content', []) if part.get('type') == 'output_text']
    if not texts:
        raise ValueError('response_without_text')
    return json.loads(''.join(texts)), result.get('usage', {})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--limit', type=int)
    g.add_argument('--all', action='store_true')
    p.add_argument('--model', default='gpt-6-sol')
    p.add_argument('--domain', choices=['math', 'logic', 'coding'])
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--prompt-key', action='store_true')
    args = p.parse_args()
    key = (getpass.getpass('OpenAI API key (input hidden): ').strip()
           if args.prompt_key else os.environ.get('SPINE_GENERATOR_API_KEY'))
    if not key:
        raise SystemExit('Provide --prompt-key or SPINE_GENERATOR_API_KEY.')
    raw = SOURCE.read_bytes()
    source = json.loads(raw)
    augmented = json.loads((ROOT / 'spine_dataset_with_valid_corrections.json').read_text(encoding='utf-8'))
    pending = {x['id'] for x in augmented if x['valid_corrective_intervention'] is None}
    already = {json.loads(line)['id'] for line in OUT.read_text(encoding='utf-8').splitlines()
               if line.strip()} if OUT.exists() else set()
    selected = [item for item in source if item['id'] in pending
                and item['id'] not in already
                and (not args.domain or item['domain'] == args.domain)]
    if args.limit is not None:
        selected = selected[:args.limit]

    def produce(item: dict) -> dict:
        start = time.monotonic()
        candidate, usage, error = {}, {}, None
        for attempt in range(3):
            try:
                candidate, usage = request(item, key, args.model)
                break
            except urllib.error.HTTPError as exc:
                error = f'HTTP {exc.code}'
                if exc.code not in (429, 500, 502, 503, 504):
                    break
            except Exception as exc:
                error = type(exc).__name__ + ': ' + str(exc)[:150]
            time.sleep(2 ** attempt)
        flags = screen(item, candidate) if candidate else ['generation_failed']
        return {'id': item['id'], 'domain': item['domain'],
                'source_sha256': hashlib.sha256(raw).hexdigest(),
                'generator': args.model, 'status': 'candidate_only_not_validated',
                'screen_flags': flags, 'candidate': candidate, 'usage': usage,
                'error': error if not candidate else None,
                'generation_seconds': round(time.monotonic() - start, 1)}

    with OUT.open('a', encoding='utf-8') as stream:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            jobs = {pool.submit(produce, item): item['id'] for item in selected}
            for job in as_completed(jobs):
                row = job.result()
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                print(row['id'], row['screen_flags'], row['generation_seconds'], flush=True)


if __name__ == '__main__':
    main()
