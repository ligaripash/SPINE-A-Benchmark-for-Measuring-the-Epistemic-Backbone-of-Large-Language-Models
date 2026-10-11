"""Wait for a checkpointed raw run, then perform the full scoring pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = ('qwen2_5_7b', 'mistral_nemo', 'deepseek_r1_8b',
         'deepseek_r1_1_5b', 'gpt_4o_mini', 'gemini_3_flash')


def coverage(path):
    if not path.exists():
        return 0
    try:
        with path.open(encoding='utf-8') as stream:
            return len({(row['id'], row['branch']) for line in stream if line.strip()
                        if 'error' not in (row := json.loads(line))})
    except (json.JSONDecodeError, OSError):
        return 0


def run(script, name, phase=None):
    command = [sys.executable, '-u', str(HERE / script), '--name', name]
    if phase:
        command += ['--phase', phase]
    print('RUN', ' '.join(command), flush=True)
    subprocess.run(command, cwd=HERE.parent, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, choices=NAMES)
    args = parser.parse_args()
    directory = HERE / args.name
    while not (directory / 'manifest.json').exists():
        time.sleep(30)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    expected = {'base': manifest['expected_base'], 'attack': manifest['expected_attacks'],
                'correction': manifest['expected_corrections']}
    previous = None
    while True:
        current = {phase: coverage(directory / f'{phase}.jsonl') for phase in expected}
        if current != previous:
            print(f'{args.name}: raw coverage {current} / {expected}', flush=True)
            previous = current
        if current == expected:
            break
        time.sleep(60)
    for phase in ('base', 'attack', 'correction'):
        run('score_four_models.py', args.name, phase)
        run('extract_comparison_logic.py', args.name, phase)
    run('compute_four_model_metrics.py', args.name)
    print('RUN', str(HERE / 'summarize_four_model_metrics.py'), flush=True)
    subprocess.run([sys.executable, '-u', str(HERE / 'summarize_four_model_metrics.py')],
                   cwd=HERE.parent, check=True)
    print(f'{args.name}: DONE', flush=True)


if __name__ == '__main__':
    main()
