"""One-shot completion helper for the third-judge comparison."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time

import compare_judges as comparison


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--provider',choices=comparison.PROVIDERS,default='sol');args=parser.parse_args()
    comparison.configure(args.provider);OUT=comparison.OUT;m=comparison.prepare()
    while True:
        try:
            labels=comparison.rows(OUT/'labels.jsonl')
            complete=len(labels)==len({r['blind_id'] for r in labels})==m['selected']
        except (ValueError,OSError):complete=False
        if complete and not (OUT/'runner.lock').exists():break
        time.sleep(30)
    lock=OUT/'completion.lock'
    try:handle=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:return
    os.close(handle)
    try:
        with (OUT/'completion.log').open('w',encoding='utf-8') as f:
            subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).with_name('write_comparison_section.py')),'--provider',args.provider],stdout=f,stderr=subprocess.STDOUT,check=True)
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
