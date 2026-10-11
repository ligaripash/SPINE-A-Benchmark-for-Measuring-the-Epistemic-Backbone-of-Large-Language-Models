"""Apply transparent, response-level correction adjudications without changing raw scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    directory = HERE / args.name
    decisions = json.loads((directory / 'correction_adjudications.json').read_text(encoding='utf-8'))
    original = [json.loads(s) for s in (directory / 'correction_scores_assisted.jsonl').read_text(
        encoding='utf-8').splitlines() if s.strip()]
    observed = {x['id'] for x in original if x['correct'] is None}
    if set(decisions) != observed:
        raise RuntimeError(f'Adjudications do not cover exact unresolved set: {observed ^ set(decisions)}')
    final = []
    for row in original:
        updated = dict(row)
        if row['id'] in decisions:
            updated['correct'] = decisions[row['id']]['correct']
            updated['method'] = 'manual_answer_adjudication'
            updated['adjudication_reason'] = decisions[row['id']]['reason']
        final.append(updated)
    (directory / 'correction_scores_final.jsonl').write_text(
        ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in final), encoding='utf-8')
    print(f'{args.name}: {len(final)} correction scores, {len(decisions)} adjudicated')


if __name__ == '__main__':
    main()
