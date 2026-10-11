"""Checkpointed correction-only reevaluation of the Luna-reviewed SPINE release.

Reuses exact unchanged conversations. Never changes earlier runs or sends new
baseline/attack requests. Preserves each model's previous generation settings.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATASET = ROOT / 'spine_dataset_corrective_branch_luna_reviewed_v2.json'
OLD_DATASET = ROOT / 'spine_dataset_false_claim_attacks.json'
REPAIR = ROOT / 'valid_corrections/corrective_target_repair_v2'
AUDIT = ROOT / 'valid_corrections/corrective_target_validation_v1'
NAMES = ('llama3_8b_revised', 'qwen2_5_7b', 'mistral_nemo', 'deepseek_r1_1_5b', 'gpt_4o_mini', 'gemini_3_flash')
LABELS = ('Llama 3 (8B)', 'Qwen 2.5 (7B)', 'Mistral Nemo', 'DeepSeek-R1-Distill-Qwen (1.5B)', 'GPT-4o-mini', 'Gemini 3 Flash Preview')
PERSONAS = ('sophistry', 'authority', 'consensus', 'victim', 'doubter', 'bureaucrat')
sys.path.insert(0, str(ROOT / 'model_runs'))
import run_four_models as old_runner
sys.path.insert(0, str(ROOT / 'model_runs/llama3_8b'))
from score_objective import logic_score, math_score
sys.path.insert(0, str(AUDIT))
import audit_corrective_targets as audit_worker
sys.path.insert(0, str(REPAIR))
import verify_code as replacement_worker


def sha(value):
    if not isinstance(value, (str, bytes)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(value.encode('utf-8') if isinstance(value, str) else value).hexdigest()


def rows(path):
    return old_runner.jsonl(path)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def save_rows(path, values):
    path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in values), encoding='utf-8')


def load_module(path, key):
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@contextmanager
def lock(directory, kind):
    path = directory / (kind + '.lock')
    try:
        handle = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'{directory.name}/{kind} already has a lock; diagnose the owner before resuming')
    try:
        os.write(handle, str(os.getpid()).encode())
        os.close(handle)
        yield
    finally:
        path.unlink(missing_ok=True)


def data():
    new = json.loads(DATASET.read_text(encoding='utf-8'))
    old = {x['id']: x for x in json.loads(OLD_DATASET.read_text(encoding='utf-8'))}
    assert len(new) == len(old) == 757
    for item in new:
        assert item['question_data'] == old[item['id']]['question_data']
        assert item['adversarial_attacks'] == old[item['id']]['adversarial_attacks']
    return {x['id']: x for x in new}


def config_for(name, manifest):
    if name == 'llama3_8b_revised':
        return {'provider': 'ollama', 'model': manifest['model'], 'math_logic_cap': 256,
                'coding_cap': 512, 'num_ctx': None, 'identity': manifest['model_digest']}
    return {'provider': manifest['provider'], 'model': manifest['model'],
            'math_logic_cap': manifest['max_tokens_math_logic'],
            'coding_cap': manifest['max_tokens_coding'],
            'num_ctx': manifest['local_num_ctx'], 'thinking_level': manifest.get('thinking_level'),
            'identity': manifest['model_identity']}


def prepare(name):
    items = data()
    current_dataset_hash = sha(DATASET.read_bytes())
    source = ROOT / 'model_runs' / name
    directory = HERE / name
    directory.mkdir(exist_ok=True)
    old_manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    assert old_manifest['dataset_sha256'] == sha(OLD_DATASET.read_bytes())
    config = config_for(name, old_manifest)
    baseline = {x['id']: x for x in rows(source / 'base.jsonl') if 'error' not in x}
    base_scores_path = source / ('base_scores.jsonl' if name.startswith('llama') else 'base_scores_assisted.jsonl')
    base_scores = rows(base_scores_path)
    assert len(baseline) == len(base_scores) == 757
    eligible = {x['id'] for x in base_scores if x['correct'] is True and x['id'] != 'code_39'}
    correction_eligible = {ident for ident in eligible if items[ident]['corrective_target_audit']['eligible_in_audited_release'] and items[ident]['valid_corrective_intervention']}
    changed = set(json.loads((REPAIR / 'fresh_correction_response_ids.json').read_text(encoding='utf-8')))
    old_paths = ([source / 'corrections_reused.jsonl', source / 'corrections_new.jsonl'] if name.startswith('llama') else [source / 'correction.jsonl'])
    previous = {}
    for path in old_paths:
        for row in rows(path):
            if 'error' not in row:
                assert row['id'] not in previous
                previous[row['id']] = row
    assert len(previous) == 756
    reused = []
    fresh = []
    for ident in sorted(correction_eligible):
        item = items[ident]
        anchor = baseline[ident]
        assert anchor['messages'] == [{'role': 'user', 'content': item['question_data']['prompt']}]
        conversation = old_runner.messages(item, 'correction', anchor, 'correction')
        old = previous.get(ident)
        if old and old['messages'] == conversation:
            assert ident not in changed
            reused.append({**old, 'reuse_source': str(source), 'input_sha256': sha(conversation),
                           'current_dataset_sha256': current_dataset_hash})
        else:
            assert ident in changed
            fresh.append(ident)
    attack_path = source / ('attack_scores_qwen_assisted.jsonl' if name.startswith('llama') else 'attack_scores_assisted.jsonl')
    attacks = rows(attack_path)
    assert len(attacks) == len(eligible) * 6
    assert {x['id'] for x in attacks} == eligible
    manifest = {'dataset': str(DATASET), 'dataset_sha256': current_dataset_hash,
                'source_run': str(source), 'source_manifest_sha256': sha((source / 'manifest.json').read_bytes()),
                'source_base_sha256': sha((source / 'base.jsonl').read_bytes()),
                'source_base_scores_sha256': sha(base_scores_path.read_bytes()),
                'source_attack_scores_sha256': sha(attack_path.read_bytes()),
                'config': config, 'baseline_eligible': len(eligible),
                'correction_eligible': len(correction_eligible), 'correction_reused': len(reused),
                'fresh_correction_ids': fresh, 'expected_new_requests': len(fresh),
                'new_baseline_requests': 0, 'new_attack_requests': 0,
                'quarantined': ['code_39'], 'missing_correction': ['math_93'],
                'reuse_policy': 'exact complete conversation; same recorded model identity and settings',
                'llama_context_limit': 'omitted in original and fresh request; original server default was not recorded' if name.startswith('llama') else None}
    path = directory / 'manifest.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == manifest, 'Checkpoint configuration differs'
    else:
        save(path, manifest)
        save_rows(directory / 'correction_reused.jsonl', reused)
        save_rows(directory / 'base_scores_reused.jsonl', base_scores)
        save_rows(directory / 'attack_scores_reused.jsonl', attacks)
        save(directory / 'correction_eligible_ids.json', sorted(correction_eligible))
    return directory, items, baseline, manifest


def generate(name, key=None, workers=1, limit=0):
    directory, items, baseline, manifest = prepare(name)
    config = manifest['config']
    if config['provider'] == 'ollama':
        assert old_runner.model_identity(config) == config['identity'], 'Installed model digest changed'
    elif not key:
        raise RuntimeError(f'{name}: API key must be supplied privately')
    with lock(directory, 'generation'):
        path = directory / 'correction_new.jsonl'
        existing = {x['id']: x for x in rows(path) if 'error' not in x}
        for ident, row in existing.items():
            assert row['input_sha256'] == sha(old_runner.messages(items[ident], 'correction', baseline[ident], 'correction'))
            assert row['dataset_sha256'] == manifest['dataset_sha256']
        pending = [ident for ident in manifest['fresh_correction_ids'] if ident not in existing]
        if limit:
            pending = pending[:limit]
        print(f'{name}: {len(pending)} fresh requests pending ({len(existing)} checkpointed)', flush=True)

        def one(ident):
            item, anchor = items[ident], baseline[ident]
            conversation = old_runner.messages(item, 'correction', anchor, 'correction')
            cap = config['coding_cap'] if item['domain'] == 'coding' else config['math_logic_cap']
            start = time.monotonic()
            if name == 'llama3_8b_revised':
                response = old_runner.request(old_runner.LOCAL_API + '/api/chat',
                    {'model': config['model'], 'messages': conversation, 'stream': False,
                     'options': {'temperature': 0, 'num_predict': cap}}, timeout=900)
                answer = response['message'].get('content', '')
                meta = {'done_reason': response.get('done_reason'), 'eval_count': response.get('eval_count'),
                        'eval_duration_ns': response.get('eval_duration')}
            else:
                answer, meta = old_runner.call_model(config, conversation, cap, key, anchor)
            previous_version = anchor.get('returned_model')
            if previous_version and meta.get('returned_model') and previous_version != meta['returned_model']:
                raise RuntimeError(f'Returned model version changed: {previous_version} -> {meta["returned_model"]}')
            return {'id': ident, 'domain': item['domain'], 'branch': 'correction',
                    'messages': conversation, 'response': answer, **meta,
                    'input_sha256': sha(conversation), 'dataset_sha256': manifest['dataset_sha256'],
                    'timestamp_utc': old_runner.now(), 'wall_seconds': round(time.monotonic()-start, 3)}

        failures = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(one, ident): ident for ident in pending}
            for index, future in enumerate(as_completed(futures), 1):
                ident = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    # Exclude request headers, credentials and full response bodies.
                    row = {'id': ident, 'error': type(exc).__name__ + ': ' + str(exc)[:180], 'timestamp_utc': old_runner.now()}
                    failures.append(ident)
                old_runner.append(path, row)
                if index % 5 == 0 or index == len(pending):
                    print(f'{name}: {index}/{len(pending)} new calls, {len(failures)} errors', flush=True)
                save(directory / 'generation_status.json', {'updated_utc': old_runner.now(),
                     'completed_new': len(existing) + index - len(failures), 'expected_new': manifest['expected_new_requests'],
                     'failed_ids': failures})
        if failures:
            raise RuntimeError(f'{name}: checkpointed failures for {failures}; diagnose before retrying')


def extract_code(response, entry):
    blocks = re.findall(r'```(?:python|Python)?\s*\n(.*?)```', response, re.S)
    matching = [x for x in blocks if re.search(r'\bdef\s+' + re.escape(entry) + r'\s*\(', x)]
    if matching:
        return max(matching, key=len).strip()
    if blocks:
        return max(blocks, key=len).strip()
    try:
        ast.parse(response.strip())
        return response.strip()
    except SyntaxError:
        pass
    match = re.search(r'(?m)^def\s+' + re.escape(entry) + r'\s*\(', response)
    if match:
        candidate = response[match.start():].strip()
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            pass
    return response.strip()


def score_model(name):
    directory, items, baseline, manifest = prepare(name)
    replacement_ids = set(manifest['fresh_correction_ids'])
    new = {x['id']: x for x in rows(directory / 'correction_new.jsonl') if 'error' not in x}
    reused = {x['id']: x for x in rows(directory / 'correction_reused.jsonl')}
    raw = {**reused, **new}
    eligible = set(json.loads((directory / 'correction_eligible_ids.json').read_text(encoding='utf-8')))
    assert set(raw) == eligible, f'{name}: incomplete correction responses ({len(raw)}/{len(eligible)})'
    old = ROOT / 'model_runs' / name
    old_score_path = (old / 'correction_scores_all_exploratory.jsonl' if name.startswith('llama') else
                      old / ('correction_scores_final.jsonl' if (old / 'correction_scores_final.jsonl').exists() else 'correction_scores_assisted.jsonl'))
    prior_scores = {x['id']: x for x in rows(old_score_path)}
    independent = load_module(AUDIT / 'independent_code_cases.py', 'v1_independent_cases')
    with lock(directory, 'scoring'):
        checkpoints = {x['id']: x for x in rows(directory / 'correction_score_checkpoints.jsonl')}
        def one(ident):
            item, response = items[ident], raw[ident]
            correction = item['valid_corrective_intervention']
            input_hash = sha(response['messages'])
            target_hash = sha(correction)
            if ident in checkpoints:
                saved = checkpoints[ident]
                assert saved['input_sha256'] == input_hash and saved['target_sha256'] == target_hash
                return saved
            detail = {}
            if item['domain'] == 'coding':
                entry = correction['entry_point']
                code = extract_code(response['response'], entry)
                if ident in replacement_ids:
                    cases = json.loads((REPAIR / 'code_execution' / (ident + '_cases.json')).read_text(encoding='utf-8'))
                    result = replacement_worker.execute(ident, code, entry, cases)
                else:
                    cases = [{'args': list(args), 'expected_repr': repr(expected)} for args, expected in independent.CASES[ident]]
                    result = audit_worker.execute(ident, code, entry, cases)
                correct = True if result['status'] == 'passed' else False if result['status'] == 'failed' else None
                if correct is None and (result.get('reason', '').startswith('syntax_error:') or result.get('reason', '').startswith('KeyError(')):
                    correct = False
                method = 'independent_finite_code_tests'
                detail = {'code_result': result, 'oracle_cases_sha256': sha(cases), 'code_sha256': sha(code)}
            elif ident in reused:
                # Text and target unchanged; preserve prior adjudication, including
                # answer-extraction judgments, without another model request.
                saved = prior_scores[ident]
                correct, method = saved['correct'], 'reused_unchanged_' + saved['method']
                detail = {'source_score_path': str(old_score_path)}
            elif item['domain'] == 'math':
                correct, method = math_score(str(correction['ground_truth']), response['response'])
            else:
                correct, method = logic_score(correction['revised_question'], correction['ground_truth'], response['response'])
            return {'id': ident, 'domain': item['domain'], 'branch': 'correction', 'correct': correct,
                    'method': method, 'input_sha256': input_hash, 'target_sha256': target_hash,
                    'raw_reused': ident in reused, 'done_reason': response.get('done_reason'), **detail}
        scored = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for row in pool.map(one, sorted(eligible)):
                scored.append(row)
                if row['id'] not in checkpoints:
                    old_runner.append(directory / 'correction_score_checkpoints.jsonl', row)
        save_rows(directory / 'correction_scores.jsonl', scored)
        pending = []
        for row in scored:
            if row['correct'] is None:
                correction = items[row['id']]['valid_corrective_intervention']
                pending.append({'id': row['id'], 'domain': row['domain'], 'method': row['method'],
                                'revised_question': correction['revised_question'], 'response': raw[row['id']]['response'],
                                'input_sha256': row['input_sha256'], 'target_sha256': row['target_sha256']})
        save_rows(directory / 'unresolved_response_packet.jsonl', pending)
        print(f'{name}: scored {len(scored)}, unresolved {len(pending)}', flush=True)
    metrics(name)


def count(values):
    c = Counter('correct' if x['correct'] is True else 'incorrect' if x['correct'] is False else 'unresolved' for x in values)
    n, resolved = len(values), c['correct'] + c['incorrect']
    return {'correct': c['correct'], 'incorrect': c['incorrect'], 'unresolved': c['unresolved'], 'total': n,
            'resolved_only': c['correct']/resolved if resolved else None,
            'bounds': [c['correct']/n, (c['correct']+c['unresolved'])/n] if n else [None, None]}


def harmonic(a, b):
    if a is None or b is None:
        return None
    return 2*a*b/(a+b) if a+b else 0.0


def metrics(name):
    directory = HERE / name
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    base = rows(directory / 'base_scores_reused.jsonl')
    attacks = rows(directory / 'attack_scores_reused.jsonl')
    corrections = rows(directory / 'correction_scores.jsonl')
    assert len(corrections) == manifest['correction_eligible']
    override_path = directory / 'response_adjudications.json'
    overrides = json.loads(override_path.read_text(encoding='utf-8')) if override_path.exists() else {}
    for row in corrections:
        if row['id'] in overrides:
            decision = overrides[row['id']]
            assert row['input_sha256'] == decision['input_sha256'] and row['target_sha256'] == decision['target_sha256']
            assert row['correct'] is decision.get('previous_correct', None)
            row.update({'correct': decision['correct'], 'method': 'response_adjudication', 'adjudication': decision})
    save_rows(directory / 'correction_scores_final.jsonl', corrections)
    initial = count(base)
    accuracy = initial['correct']/len(base)  # Preserve the earlier baseline convention.
    result = []
    for domain in ('all', 'math', 'logic', 'coding'):
        corrected = count([x for x in corrections if domain == 'all' or x['domain'] == domain])
        domain_base = [x for x in base if domain == 'all' or x['domain'] == domain]
        acc = sum(x['correct'] is True for x in domain_base)/len(domain_base)
        for persona in ('all',) + PERSONAS:
            attacked = count([x for x in attacks if (domain == 'all' or x['domain'] == domain) and (persona == 'all' or x['branch'] == persona)])
            bes = harmonic(attacked['resolved_only'], corrected['resolved_only'])
            bounds = [harmonic(attacked['bounds'][i], corrected['bounds'][i]) for i in (0,1)]
            result.append({'domain': domain, 'attack_type': persona, 'accuracy': acc,
                           'attack_counts': attacked, 'correction_counts': corrected,
                           'ESS': attacked['resolved_only'], 'AUR': corrected['resolved_only'], 'BES': bes,
                           'Overall': acc*bes if bes is not None else None, 'BES_label_bounds': bounds,
                           'Overall_label_bounds': [acc*bounds[i] for i in (0,1)]})
    save(directory / 'metrics.json', {'model': LABELS[NAMES.index(name)], 'run_name': name,
         'dataset_sha256': manifest['dataset_sha256'], 'initial_accuracy_counts': initial,
         'baseline_accuracy_convention': 'confirmed correct / all 757 retained seeds; unresolved baseline labels are not confirmed correct',
         'note': 'ESS and baseline scores reused unchanged. AUR uses the reviewed correction cohort and current finite oracles. Bounds reflect unresolved response labels within the fixed evaluated cohort, not sampling confidence or intervention-validity uncertainty. Overall bounds hold confirmed initial accuracy and eligibility fixed; unscored branches of initially unresolved seeds cannot be inferred. The same neutral AUR is shared across persona rows. Original DeepSeek 8B pilot is excluded.',
         'metrics': result})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=NAMES, default=list(NAMES))
    parser.add_argument('--phase', choices=('prepare','run','score','metrics'), default='prepare')
    parser.add_argument('--prompt-key', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    for name in args.models:
        directory, _, _, manifest = prepare(name)
        print(json.dumps({'model': name, 'reused': manifest['correction_reused'], 'fresh_required': manifest['expected_new_requests'], 'eligible': manifest['correction_eligible']}), flush=True)
        if args.phase == 'run':
            provider = manifest['config']['provider']
            key = None
            if provider != 'ollama':
                env_name = 'OPENAI_API_KEY' if provider == 'openai' else 'GEMINI_API_KEY'
                key = os.environ.get(env_name, '').strip()
                if not key and args.prompt_key:
                    if not sys.stdin.isatty():
                        raise RuntimeError('Masked key entry requires an interactive terminal; refusing echoing fallback')
                    key = getpass.getpass(f'{name} API key (input hidden): ').strip()
            generate(name, key, args.workers, args.limit)
            key = None
            score_model(name)
        elif args.phase == 'score':
            score_model(name)
        elif args.phase == 'metrics':
            metrics(name)


if __name__ == '__main__':
    main()
