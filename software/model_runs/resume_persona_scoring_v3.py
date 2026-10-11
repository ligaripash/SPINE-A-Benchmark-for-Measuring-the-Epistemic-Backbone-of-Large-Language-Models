"""Resume one stopped scoring lane while other target runners continue.

This entry point cannot make target-model requests. It requires complete raw
outputs, a documented stopped lane, and an exclusive per-model scoring lock.
Successful judge receipts are reused exactly; credentials stay in memory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import getpass
import json
import os
import time
import reevaluate_persona_v3 as run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=run.NAMES, required=True)
    args = parser.parse_args()
    name = args.model
    manifest = run.prepare()
    directory = run.OUT / name
    packet = run.rows(directory / 'request_packet.jsonl')
    state = run.load(directory / 'progress.json')
    if state['status'] not in ['blocked', 'failed', 'scoring_resume_stopped']:
        raise SystemExit('Scoring lane is not documented as stopped; do not run concurrently.')
    assert all((directory / 'raw' / (r['id'] + '__' + r['branch'] + '.json')).exists() for r in packet)
    if list(directory.rglob('*.inflight.json')):
        raise SystemExit('Unconfirmed prior requests require diagnosis; no automatic resume.')
    lock = directory / 'scoring_resume.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump({'pid': os.getpid(), 'started_at_utc': run.now(), 'model': name,
                   'scope': 'Scoring only; target request generation disabled.'}, stream)
    try:
        run.KEYS['openai'] = getpass.getpass('OpenAI API key for resumed blinded scoring (masked): ').strip()
        run.KEYS['gemini'] = getpass.getpass('Gemini API key for resumed blinded scoring (masked): ').strip()
        if not all(run.KEYS.values()):
            raise SystemExit('Both credentials required for the frozen scoring policy.')
        old = {r['blind_id']: r for r in run.rows(run.OUT / 'source_response_index.jsonl')}
        items = {r['id']: r for r in run.load(run.DATASET)}
        pending = [r for r in packet if not (directory / 'scores' / (r['id'] + '__' + r['branch'] + '.json')).exists()]
        original_post = run.post
        def transient_post(provider, url, payload):
            # A returned HTTP 5xx is a failed service response, distinct from a
            # transport timeout whose completion is unknown. Retain diagnostics.
            for attempt in range(3):
                try:
                    return original_post(provider, url, payload)
                except run.ApiFailure as exc:
                    if exc.info.get('http_status') not in (500, 502, 503, 504) or attempt == 2:
                        raise
                    run.save(directory / 'resume_service_errors' / (str(time.time_ns()) + '.json'),
                             {'timestamp_utc': run.now(), **exc.info})
                    time.sleep(30)
        run.post = transient_post
        # Serial resume avoids three simultaneous requests during service recovery.
        run.model_status(name, packet, 'scoring_resumed', prior_failure=state.get('error'),
                         resume_policy='Only missing score records; successful target/judge receipts retained.')
        for count, request in enumerate(pending, 1):
            raw = run.load(directory / 'raw' / (request['id'] + '__' + request['branch'] + '.json'))
            assert raw['request_sha256'] == request['request_sha256']
            run.score_response(request, raw, old[request['old_blind_id']], items)
            if count % 5 == 0 or count == len(pending):
                print(json.dumps(run.model_status(name, packet, 'scoring_resumed')), flush=True)
        print(json.dumps(run.model_status(name, packet, 'complete')), flush=True)
    except run.ApiFailure as exc:
        print(json.dumps(run.model_status(name, packet, 'scoring_resume_stopped', error=exc.info)), flush=True)
    except Exception as exc:
        run.model_status(name, packet, 'scoring_resume_stopped', error={'type': type(exc).__name__})
        raise
    finally:
        lock.unlink(missing_ok=True)
        run.KEYS.clear()


if __name__ == '__main__':
    main()
