"""One-shot completion worker for active judge jobs, not a recurring automation."""
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def complete(path,n):
    if not path.exists():return False
    try:
        rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
        return len(rows)==n and len({r['blind_id'] for r in rows})==n
    except (json.JSONDecodeError,OSError):return False


def main():
    expected=json.loads((HERE/'manifest.json').read_text())['science_judgments_per_judge']
    # Wait without scoring partly written files or duplicating active scoring.
    while True:
        manifest=json.loads((HERE/'manifest.json').read_text())
        required=manifest.get('analysis_judges',['luna','gemini'])
        if all(complete(HERE/(j+'_labels.jsonl'),expected) and not (HERE/(j+'.lock')).exists() for j in required):break
        time.sleep(30)
    import os
    lock=HERE/'analysis.lock'
    try:handle=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:return
    os.close(handle)
    try:
        command=[sys.executable,'-X','utf8',str(HERE/'analyze_judges.py'),'--second-judge',required[1]]
        exports=sorted((HERE/'human_exports').glob('*.json')) if (HERE/'human_exports').exists() else []
        if exports:command+=['--human']+[str(p) for p in exports]
        with (HERE/'completion_analysis.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            subprocess.run([sys.executable,'-X','utf8',str(HERE/'write_paper_analysis.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
