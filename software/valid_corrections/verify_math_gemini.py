"""Independently solve drafted math revisions with Gemini code execution.

Agreement is a triage signal, not proof of the question's validity.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / 'valid_corrections/reasoned_candidate_drafts.jsonl'
OUT = ROOT / 'valid_corrections/gemini_math_checks.jsonl'
MODEL = 'gemini-3.8-flash'
URL = f'https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent'


def number(value):
    try:
        return Decimal(str(value).replace(',', '').replace('$', '').strip())
    except InvalidOperation:
        return None


def request(row, key):
    q = row['candidate']['revised_question']
    prompt = ('Solve this revised math word problem independently. Do not use any '
              'stored answer or candidate derivation. Write and execute Python '
              'for the arithmetic, then return only a JSON object with keys '
              '"answer" (a plain numeric string), "derivation" (short string), '
              'and "well_posed" (boolean). If ambiguous or inconsistent, set '
              'well_posed=false and explain.\n\nQUESTION:\n' + q)
    payload = {'contents': [{'parts': [{'text': prompt}]}],
               'tools': [{'code_execution': {}}],
               'generationConfig': {'responseMimeType': 'application/json'}}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode('utf-8'),
        headers={'x-goog-api-key': key, 'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=180) as response:
        result = json.load(response)
    candidate = result['candidates'][0]
    parts = candidate['content']['parts']
    texts = [p['text'] for p in parts if 'text' in p]
    text = '\n'.join(texts).strip()
    match = re.search(r'\{.*\}', text, re.S)
    if not match:
        raise ValueError('no_json_object_in_response')
    answer = json.loads(match.group())
    traces = [{'code': p.get('executableCode',{}).get('code',''),
               'outcome': p.get('codeExecutionResult',{}).get('outcome',''),
               'output': p.get('codeExecutionResult',{}).get('output','')[:1000]}
              for p in parts if 'executableCode' in p or 'codeExecutionResult' in p]
    return answer, traces, result.get('usageMetadata',{})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--limit', type=int)
    g.add_argument('--all', action='store_true')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--prompt-key', action='store_true')
    args = p.parse_args()
    key = (getpass.getpass('Gemini API key (input hidden): ').strip()
           if args.prompt_key else os.environ.get('SPINE_GEMINI_API_KEY'))
    if not key:
        raise SystemExit('Provide --prompt-key or SPINE_GEMINI_API_KEY.')
    rows = [json.loads(line) for line in IN.read_text(encoding='utf-8').splitlines() if line.strip()]
    already = {json.loads(line)['id'] for line in OUT.read_text(encoding='utf-8').splitlines()
               if line.strip()} if OUT.exists() else set()
    selected = [row for row in rows if row['domain']=='math' and row['id'] not in already
                and row.get('candidate') and not row['candidate'].get('unable')]
    if args.limit is not None:
        selected = selected[:args.limit]

    def produce(row):
        start = time.monotonic()
        answer, traces, usage, error = {}, [], {}, None
        for attempt in range(3):
            try:
                answer, traces, usage = request(row, key)
                break
            except urllib.error.HTTPError as exc:
                error = f'HTTP {exc.code}'
                if exc.code not in (429,500,502,503,504):
                    break
            except Exception as exc:
                error = type(exc).__name__+': '+str(exc)[:150]
            time.sleep(2**attempt)
        proposed = row['candidate']['revised_ground_truth']
        solved = answer.get('answer','')
        return {'id':row['id'], 'verifier':MODEL, 'candidate_generator':row['generator'],
                'status':'independent_model_check_only', 'proposed_answer':proposed,
                'solver':answer, 'agrees_numeric':(number(proposed)==number(solved)
                  if number(proposed) is not None and number(solved) is not None else None),
                'code_execution_seen':any(x['code'] or x['output'] for x in traces),
                'execution_traces':traces, 'usage':usage, 'error':error,
                'seconds':round(time.monotonic()-start,1)}

    with OUT.open('a',encoding='utf-8') as stream:
        with ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool:
            jobs={pool.submit(produce,row):row['id'] for row in selected}
            for job in as_completed(jobs):
                row=job.result()
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                print(row['id'],row['agrees_numeric'],row['error'],row['seconds'],flush=True)


if __name__=='__main__':
    main()
