"""Wait for the isolated adjudication checkpoint, then score and update the paper."""
import os
from pathlib import Path
import subprocess
import sys
import time

import run_sol_adjudication as adjudication
import run_sol_clarification as clarification
import run_sol_agreed_unclear as agreed_unclear


def main():
    m = adjudication.prepare(); out = adjudication.OUT
    while True:
        try:
            labels = adjudication.runner.rows(out / 'labels.jsonl')
            ready = len(labels) == len({r['blind_id'] for r in labels}) == m['selected']
        except (ValueError, OSError):
            ready = False
        if ready and not (out / 'runner.lock').exists():
            required = clarification.prepare()['selected']
            clarified = clarification.runner.rows(clarification.OUT / 'labels.jsonl')
            if len(clarified) == len({r['blind_id'] for r in clarified}) == required and not (clarification.OUT / 'runner.lock').exists():
                required_agreed = agreed_unclear.prepare()['selected']
                agreed = agreed_unclear.runner.rows(agreed_unclear.OUT / 'labels.jsonl')
                if len(agreed) == len({r['blind_id'] for r in agreed}) == required_agreed and not (agreed_unclear.OUT / 'runner.lock').exists():
                    break
        time.sleep(30)
    lock = out / 'completion.lock'
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return
    os.close(handle)
    try:
        with (out / 'completion.log').open('w', encoding='utf-8') as f:
            subprocess.run([sys.executable, '-X', 'utf8',
                            str(Path(__file__).with_name('analyze_sol_adjudication.py'))],
                           stdout=f, stderr=subprocess.STDOUT, check=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
