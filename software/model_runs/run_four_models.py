"""Checkpointed SPINE responses for four requested comparison models.

All models receive the same original prompt and independent baseline->followup
branches as the Llama run. The OpenAI key is read from the environment or a
masked terminal prompt, never written to artifacts or logs.
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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / 'spine_dataset_false_claim_attacks.json'
PERSONAS = ('sophistry', 'authority', 'consensus', 'victim', 'doubter', 'bureaucrat')
LOCAL_API = os.environ.get('SPINE_OLLAMA_API', 'http://127.0.0.1:11434')
CONFIGS = {
    'qwen2_5_7b': {'provider': 'ollama', 'model': 'qwen2.5:latest'},
    'mistral_nemo': {'provider': 'ollama', 'model': 'mistral-nemo:latest'},
    'deepseek_r1_8b': {'provider': 'ollama', 'model': 'deepseek-r1:latest',
                       'math_logic_cap': 1024, 'coding_cap': 2048},
    'deepseek_r1_1_5b': {'provider': 'ollama', 'model': 'deepseek-r1:1.5b',
                         'math_logic_cap': 2048, 'coding_cap': 2048,
                         'num_ctx': 4096},
    'gpt_4o_mini': {'provider': 'openai', 'model': 'gpt-4o-mini-2024-07-18'},
    'gemini_3_flash': {'provider': 'gemini', 'model': 'gemini-3-flash-preview',
                       'math_logic_cap': 2048, 'coding_cap': 4096,
                       'thinking_level': 'low'},
}
LOCAL_CONTEXT_TOKENS = 4096


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def append(path: Path, row: dict) -> None:
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def request(url: str, payload: dict, *, key: str | None = None,
            key_header: str = 'Authorization', timeout: int = 600) -> dict:
    headers = {'Content-Type': 'application/json'}
    if key:
        headers[key_header] = key if key_header == 'x-goog-api-key' else f'Bearer {key}'
    body = json.dumps(payload).encode('utf-8')
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code not in (408, 409, 429, 500, 502, 503, 504) or attempt == 4:
                # Do not include response headers or request body; they may
                # contain account-specific information.
                raise RuntimeError(f'HTTP {exc.code}') from None
        except (urllib.error.URLError, TimeoutError):
            if attempt == 4:
                raise
        time.sleep(min(30, 3 * 2**attempt))
    raise RuntimeError('Request retries exhausted')


def model_identity(config: dict) -> str:
    if config['provider'] in ('openai', 'gemini'):
        return config['model']
    with urllib.request.urlopen(LOCAL_API + '/api/tags', timeout=10) as response:
        matches = [x['digest'] for x in json.load(response)['models']
                   if x['name'] == config['model']]
    if len(matches) != 1:
        raise RuntimeError(f'Local model not installed: {config["model"]}')
    return matches[0]


def correction(item: dict) -> str | None:
    if item.get('valid_corrective_intervention'):
        return item['valid_corrective_intervention']['intervention_text']
    return (item.get('correction_candidate_for_review') or {}).get('candidate', {}).get('intervention_text')


def messages(item: dict, phase: str, baseline: dict | None, branch: str | None) -> list[dict]:
    conversation = [{'role': 'user', 'content': item['question_data']['prompt']}]
    if phase == 'base':
        return conversation
    if baseline is None:
        raise RuntimeError(f'Baseline missing for {item["id"]}')
    followup = (item['adversarial_attacks'][branch]['attack_text'] if phase == 'attack'
                else correction(item))
    if not followup:
        raise RuntimeError(f'Followup missing for {item["id"]}/{branch}')
    return conversation + [{'role': 'assistant', 'content': baseline['response']},
                           {'role': 'user', 'content': followup}]


def init_run(name: str, config: dict) -> tuple[Path, list[dict], dict]:
    here = ROOT / 'model_runs' / name
    here.mkdir(parents=True, exist_ok=True)
    raw = DATASET.read_bytes()
    items = json.loads(raw)
    identity = model_identity(config)
    manifest = {'dataset': str(DATASET), 'dataset_sha256': sha(raw),
                'provider': config['provider'], 'model': config['model'],
                'model_identity': identity,
                'temperature': 1.0 if config['provider'] == 'gemini' else 0,
                'max_tokens_math_logic': config.get('math_logic_cap', 256),
                'max_tokens_coding': config.get('coding_cap', 512),
                'thinking_level': config.get('thinking_level'),
                'local_num_ctx': config.get('num_ctx', LOCAL_CONTEXT_TOKENS)
                if config['provider'] == 'ollama' else None,
                'system_message': None, 'followup_policy': 'all_retained_seeds',
                'expected_base': len(items), 'expected_attacks': len(items) * 6,
                'expected_corrections': sum(bool(correction(x)) for x in items)}
    path = here / 'manifest.json'
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if {k: previous.get(k) for k in manifest} != manifest:
            raise RuntimeError(f'Existing manifest differs for {name}')
    else:
        path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return here, items, manifest


def call_model(config: dict, conversation: list[dict], max_tokens: int,
               api_key: str | None, baseline: dict | None = None) -> tuple[str, dict]:
    if config['provider'] == 'ollama':
        response = request(LOCAL_API + '/api/chat',
                           {'model': config['model'], 'messages': conversation,
                            'stream': False, 'keep_alive': '30m',
                            'options': {'temperature': 0, 'num_predict': max_tokens,
                                        'num_ctx': config.get('num_ctx', LOCAL_CONTEXT_TOKENS)}},
                           timeout=900)
        message = response['message']
        return message.get('content', ''), {
            'done_reason': response.get('done_reason'),
            'eval_count': response.get('eval_count'),
            'eval_duration_ns': response.get('eval_duration'),
            'thinking_chars': len(message.get('thinking', ''))}
    if config['provider'] == 'gemini':
        contents = [{'role': 'user', 'parts': [{'text': conversation[0]['content']}]}]
        if len(conversation) > 1:
            if not baseline or not baseline.get('gemini_model_content'):
                raise RuntimeError('Gemini baseline model content missing')
            contents += [baseline['gemini_model_content'],
                         {'role': 'user', 'parts': [{'text': conversation[2]['content']}]}]
        response = request(
            'https://generativelanguage.googleapis.com/v1beta/models/' +
            config['model'] + ':generateContent',
            {'contents': contents,
             'generationConfig': {'maxOutputTokens': max_tokens,
                                  'thinkingConfig': {'thinkingLevel': config['thinking_level']}}},
            key=api_key, key_header='x-goog-api-key', timeout=300)
        candidates = response.get('candidates') or []
        if not candidates:
            raise RuntimeError('Gemini returned no candidate: ' +
                               str(response.get('promptFeedback', {}).get('blockReason', 'unknown')))
        candidate = candidates[0]
        content = candidate.get('content') or {}
        parts = content.get('parts') or []
        answer = ''.join(part.get('text', '') for part in parts if not part.get('thought'))
        return answer, {'done_reason': candidate.get('finishReason'),
                        'returned_model': response.get('modelVersion'),
                        'usage': response.get('usageMetadata'),
                        'gemini_model_content': content}
    response = request('https://api.openai.com/v1/chat/completions',
                       {'model': config['model'], 'messages': conversation,
                        'temperature': 0, 'max_tokens': max_tokens},
                       key=api_key, timeout=180)
    choice = response['choices'][0]
    return choice['message'].get('content') or '', {
        'done_reason': choice.get('finish_reason'), 'returned_model': response.get('model'),
        'usage': response.get('usage'), 'request_id': response.get('id')}


def run_phase(name: str, phase: str, workers: int, api_key: str | None,
              only_ids: set[str] | None = None) -> None:
    config = CONFIGS[name]
    here, items, manifest = init_run(name, config)
    if only_ids is not None:
        items = [item for item in items if item['id'] in only_ids]
    if phase != 'base' and not {item['id'] for item in items}.issubset(
            {x['id'] for x in jsonl(here / 'base.jsonl') if 'error' not in x}):
        raise RuntimeError(f'{name}: complete baseline required before {phase}')
    base = {x['id']: x for x in jsonl(here / 'base.jsonl') if 'error' not in x}
    path = here / f'{phase}.jsonl'
    done = {(x['id'], x['branch']) for x in jsonl(path) if 'error' not in x}
    pending = []
    for item in items:
        branches = ((None,) if phase == 'base' else PERSONAS if phase == 'attack'
                    else ('correction',) if correction(item) else ())
        for branch in branches:
            if (item['id'], branch) not in done:
                pending.append((item, branch))
    print(f'{name}/{phase}: pending={len(pending)}', flush=True)
    if not pending:
        return

    def one(item: dict, branch: str | None) -> dict:
        start = time.monotonic()
        conversation = messages(item, phase, base.get(item['id']), branch)
        cap = (config.get('coding_cap', 512) if item['domain'] == 'coding' else
               config.get('math_logic_cap', 256))
        content, meta = call_model(config, conversation, cap, api_key, base.get(item['id']))
        return {'id': item['id'], 'domain': item['domain'], 'branch': branch,
                'messages': conversation, 'response': content, **meta,
                'wall_seconds': round(time.monotonic() - start, 3),
                'timestamp_utc': now(), 'dataset_sha256': manifest['dataset_sha256']}

    errors = Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, item, branch): (item['id'], branch)
                   for item, branch in pending}
        for index, future in enumerate(as_completed(futures), 1):
            ident, branch = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                errors[type(exc).__name__] += 1
                row = {'id': ident, 'branch': branch, 'error': type(exc).__name__ + ': ' + str(exc),
                       'timestamp_utc': now()}
            append(path, row)
            if index % 20 == 0 or index == len(pending):
                print(f'{name}/{phase}: {index}/{len(pending)}, errors={sum(errors.values())}', flush=True)
    if errors:
        raise RuntimeError(f'{name}/{phase}: {dict(errors)}; rerun to retry checkpointed failures')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, choices=CONFIGS)
    parser.add_argument('--phase', choices=('base', 'attack', 'correction', 'all'), default='all')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--prompt-key', action='store_true')
    parser.add_argument('--pilot-then-all', action='store_true',
                        help='For Gemini, checkpoint one seed per domain before the full run')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit('workers must be 1..8')
    config = CONFIGS[args.name]
    api_key = None
    if config['provider'] == 'openai':
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key and args.prompt_key:
            api_key = getpass.getpass('Enter a fresh OpenAI API key (hidden): ')
        if not api_key:
            raise SystemExit('OpenAI API key unavailable; use --prompt-key in an interactive terminal')
    elif config['provider'] == 'gemini':
        api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        if not api_key and args.prompt_key:
            api_key = getpass.getpass('Enter Gemini API key (hidden): ')
        if not api_key:
            raise SystemExit('Gemini API key unavailable; use --prompt-key in an interactive terminal')
    phases = ('base', 'attack', 'correction') if args.phase == 'all' else (args.phase,)
    if args.pilot_then_all:
        if args.name != 'gemini_3_flash' or args.phase != 'all':
            raise SystemExit('--pilot-then-all requires Gemini and --phase all')
        pilot_ids = {'math_284', 'logic_124', 'code_132'}
        for phase in phases:
            run_phase(args.name, phase, args.workers, api_key, pilot_ids)
    for phase in phases:
        run_phase(args.name, phase, args.workers, api_key)


if __name__ == '__main__':
    main()
