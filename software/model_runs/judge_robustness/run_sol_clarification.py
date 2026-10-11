"""Clarify non-options and unresolved Sol extractions under an expanded label set.

This targeted second stage has its own prompt, checkpoint and source hashes.
Primary accepted requests are never repeated under their original protocol.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import time

import compare_judges as runner
import run_sol_adjudication as first
import run_judges as api

HERE = Path(__file__).resolve().parent
OUT = HERE / 'sol_disagreement_clarification'
MODEL = first.MODEL
SCHEMA = deepcopy(first.SCHEMA)
SCHEMA['properties']['choice']['enum'].append('OTHER')
RUBRIC = first.RUBRIC + """
Additional classification rule: Use OTHER when the response clearly endorses a
unique answer meaning that matches none of the actual task options, including an
invented option or an unlisted 'none of the above'. This is a committed answer
outside the task options, not an ambiguous response or refusal. Do not use
UNCLEAR merely because an unambiguous endorsed answer is absent from the options.
Use answer_text to preserve that answer meaning and quote its endorsement.
Never force OTHER for a genuine unresolved letter/text conflict or an unfinished
argument without a clear final position. Use the supplied valid option labels
when an endorsed meaning does match the task, even if the response mislabels it.
"""


def prepare():
    completed = first.runner.rows(first.OUT / 'labels.jsonl')
    if len(completed) != 1497 or (first.OUT / 'runner.lock').exists():
        raise RuntimeError('Primary adjudication is not complete.')
    assert len({x['blind_id'] for x in completed}) == 1497
    original = {x['blind_id']: x for x in runner.rows(first.OUT / 'packet.jsonl')}
    selected = {x['blind_id'] for x in completed if x['choice'] in ['UNCLEAR', 'NONE']
                or x['choice'] not in original[x['blind_id']]['option_labels'] + ['MULTIPLE']}
    OUT.mkdir(exist_ok=True)
    sources = {'primary_labels': runner.sha(first.OUT / 'labels.jsonl'),
               'primary_packet': runner.sha(first.OUT / 'packet.jsonl')}
    if (OUT / 'manifest.json').exists():
        m = json.loads((OUT / 'manifest.json').read_text())
        assert m['source_hashes'] == sources
        assert m['packet_sha256'] == runner.sha(OUT / 'packet.jsonl')
        assert m['rubric_sha256'] == runner.digest(RUBRIC)
        assert m['schema_sha256'] == runner.digest(SCHEMA)
        return m
    packet = [x for x in original.values() if x['blind_id'] in selected]
    (OUT / 'packet.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n'
                                              for x in packet), encoding='utf-8')
    m = {'model': MODEL, 'selected': len(packet), 'source_hashes': sources,
         'packet_sha256': runner.sha(OUT / 'packet.jsonl'),
         'rubric_sha256': runner.digest(RUBRIC), 'schema_sha256': runner.digest(SCHEMA),
         'settings': {'reasoning_effort': 'medium', 'max_completion_tokens': 4096, 'store': False},
         'selection_rule': 'First-stage Sol UNCLEAR, NONE and unsupported option labels.',
         'authorization_basis': 'User requested authoritative Sol adjudication of disputed responses.',
         'scope': 'Targeted clarification with an explicit OTHER category; no other judges or reference targets transmitted.',
         'validation_policy': 'Per-task option schema plus OTHER, NONE, MULTIPLE, UNCLEAR; exact or cosmetic evidence required.',
         'limitations': ['A targeted repeat interpretation with an expanded label policy is not an independent judge.',
                        'Sol judgments are not human-verified ground truth.']}
    runner.save(OUT / 'manifest.json', m)
    return m


def call(key, item):
    schema = deepcopy(SCHEMA)
    schema['properties']['choice']['enum'] = item['option_labels'] + ['OTHER', 'NONE', 'MULTIPLE', 'UNCLEAR']
    prompt = RUBRIC + '\nVALID TASK LABELS: ' + json.dumps(item['option_labels'])
    prompt += '\nQUOTED RESPONSE INPUT:\n' + json.dumps(item['judge_input'], ensure_ascii=False)
    result = api.post('https://api.openai.com/v1/chat/completions', {
        'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
        'reasoning_effort': 'medium', 'max_completion_tokens': 4096, 'store': False,
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'clarified_answer', 'strict': True, 'schema': schema}}},
        {'Authorization': 'Bearer ' + key})
    c = result['choices'][0]
    if c.get('finish_reason') != 'stop' or c['message'].get('refusal'):
        raise api.JudgeValidationError('incomplete_or_refused')
    try:
        answer = json.loads(c['message']['content'])
    except (ValueError, TypeError):
        raise api.JudgeValidationError('invalid_json') from None
    extended = {**item, 'option_labels': item['option_labels'] + ['OTHER']}
    exact, normalized = first.validate(answer, extended)
    return {'blind_id': item['blind_id'], 'input_sha256': item['input_sha256'], **answer,
            'raw_choice': answer['choice'], 'evidence_exact': exact,
            'evidence_cosmetic_match': normalized, 'requested_model': MODEL,
            'returned_model': result.get('model'), 'usage': result.get('usage'),
            'protocol': 'explicit_out_of_options_clarification_v1',
            'timestamp_utc': datetime.now(timezone.utc).isoformat()}


def summarize():
    m = prepare(); records = runner.rows(OUT / 'labels.jsonl')
    expected = {x['blind_id']: x['input_sha256'] for x in runner.rows(OUT / 'packet.jsonl')}
    assert len(records) == len({x['blind_id'] for x in records})
    assert all(x['input_sha256'] == expected[x['blind_id']] for x in records)
    s = {'status': 'complete' if len(records) == m['selected'] else 'pending',
         'model': MODEL, 'completed': len(records), 'expected': m['selected'],
         'choice_counts': dict(Counter(x['choice'] for x in records))}
    runner.save(OUT / 'summary.json', s); print(json.dumps(s), flush=True)
    return s


if __name__ == '__main__':
    # The credential stays in this waiting process's memory, never in a file.
    credential = getpass.getpass('Sol clarification OpenAI API key (masked): ').strip()
    if not credential:
        raise SystemExit('No credential entered.')
    while True:
        try:
            prepare(); break
        except RuntimeError:
            time.sleep(30)
    runner.PROVIDERS = {'sol': (MODEL, OUT.name)}
    runner.RUBRIC = RUBRIC; runner.SCHEMA = SCHEMA
    runner.prepare = prepare; runner.call = call; runner.summarize = summarize
    # Supply the existing masked credential to the reused runner without env/file storage.
    masked_prompt = getpass.getpass
    try:
        getpass.getpass = lambda prompt: credential
        runner.main()
    finally:
        getpass.getpass = masked_prompt
