"""Apply the explicit OTHER policy to every agreed-UNCLEAR science extraction."""
from collections import Counter
import json
from pathlib import Path

import compare_judges as runner
import run_sol_adjudication as first
import run_sol_clarification as clarification

HERE = Path(__file__).resolve().parent
OUT = HERE / 'sol_agreed_unclear_clarification'
MODEL = clarification.MODEL
RUBRIC = clarification.RUBRIC
SCHEMA = clarification.SCHEMA
call = clarification.call


def prepare():
    sources = {name: runner.sha(HERE / name) for name in
               ['judge_packet.jsonl', 'luna_labels.jsonl', 'gemini_flash_lite_labels.jsonl']}
    sources['disputed_packet'] = runner.sha(first.OUT / 'packet.jsonl')
    luna = {r['blind_id']: r for r in runner.rows(HERE / 'luna_labels.jsonl')}
    flash = {r['blind_id']: r for r in runner.rows(HERE / 'gemini_flash_lite_labels.jsonl')}
    disputed = {r['blind_id'] for r in runner.rows(first.OUT / 'packet.jsonl')}
    packet = [r for r in runner.rows(HERE / 'judge_packet.jsonl')
              if r['blind_id'] not in disputed
              and luna[r['blind_id']]['choice'] == flash[r['blind_id']]['choice'] == 'UNCLEAR'
              and luna[r['blind_id']]['raw_choice'] == flash[r['blind_id']]['raw_choice']]
    assert len(packet) == 92 and not ({r['blind_id'] for r in packet} & disputed)
    OUT.mkdir(exist_ok=True)
    if (OUT / 'manifest.json').exists():
        m = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
        assert m['source_hashes'] == sources
        assert m['packet_sha256'] == runner.sha(OUT / 'packet.jsonl')
        assert m['rubric_sha256'] == runner.digest(RUBRIC)
        assert m['schema_sha256'] == runner.digest(SCHEMA)
        assert {r['blind_id'] for r in packet} == {r['blind_id'] for r in runner.rows(OUT / 'packet.jsonl')}
        return m
    (OUT / 'packet.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n'
                                             for r in packet), encoding='utf-8')
    m = {'model': MODEL, 'selected': len(packet), 'source_hashes': sources,
         'packet_sha256': runner.sha(OUT / 'packet.jsonl'),
         'rubric_sha256': runner.digest(RUBRIC), 'schema_sha256': runner.digest(SCHEMA),
         'settings': {'reasoning_effort': 'medium', 'max_completion_tokens': 4096, 'store': False},
         'selection_rule': 'All agreed-UNCLEAR records outside the 1,497 disputed records.',
         'authorization_basis': 'User explicitly approved applying Sol clarification to the remaining 92 agreed-UNCLEAR responses.',
         'scope': 'Same blinded OTHER clarification policy; no judge decisions, reference answers or model identities transmitted.',
         'limitations': ['Model adjudication is not independent human validation.',
                        'Genuinely ambiguous responses remain unresolved.']}
    runner.save(OUT / 'manifest.json', m)
    return m


def summarize():
    m = prepare()
    records = runner.rows(OUT / 'labels.jsonl')
    expected = {r['blind_id']: r['input_sha256'] for r in runner.rows(OUT / 'packet.jsonl')}
    assert len(records) == len({r['blind_id'] for r in records})
    assert all(r['input_sha256'] == expected[r['blind_id']] for r in records)
    s = {'status': 'complete' if len(records) == m['selected'] else 'pending',
         'model': MODEL, 'completed': len(records), 'expected': m['selected'],
         'choice_counts': dict(Counter(r['choice'] for r in records))}
    runner.save(OUT / 'summary.json', s)
    print(json.dumps(s), flush=True)
    return s


if __name__ == '__main__':
    runner.PROVIDERS = {'sol': (MODEL, OUT.name)}
    runner.RUBRIC = RUBRIC
    runner.SCHEMA = SCHEMA
    runner.prepare = prepare
    runner.call = call
    runner.summarize = summarize
    runner.main()
