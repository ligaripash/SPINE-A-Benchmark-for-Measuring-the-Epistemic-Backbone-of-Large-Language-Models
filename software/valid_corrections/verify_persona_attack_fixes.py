"""Offline replay of accepted persona repairs; optional completion watcher."""
import argparse
import ctypes
import json
import os
import time

import run_persona_attack_fixer as fix


def verify():
    raw, data, items, rows, config = fix.prepare()
    selected, unclear, holds = fix.selection(rows)
    accepted = fix.approvals(selected, fix.MAX_ROUNDS)
    summary = fix.publish(data, rows, config, fix.MAX_ROUNDS)
    by_review = {r['review_id']: r for r in rows}
    code_count = 0
    for review_id, result in accepted.items():
        row = by_review[review_id]
        item = items[row['seed_id']]
        assert not row['reference_quarantined']
        draft = result['draft']
        assert fix.persona.text_sha(draft['attack_text']) == result['new_attack_sha256']
        number = result['round']
        draft_receipt = json.loads((fix.OUT / f'draft_r{number}' / (review_id + '.json')).read_text(encoding='utf-8'))
        assert draft_receipt['requested_model'] in fix.authorized_fixer_models()
        assert draft_receipt['parsed'] == draft
        fact_receipt = json.loads((fix.OUT / f'candidate_factual_r{number}' / (review_id + '.json')).read_text(encoding='utf-8'))
        persona_receipt = json.loads((fix.OUT / f'candidate_framing_r{number}' / (review_id + '.json')).read_text(encoding='utf-8'))
        assert fact_receipt['requested_model'] == persona_receipt['requested_model'] == fix.LUNA_MODEL
        assert fact_receipt['completed'] and persona_receipt['completed'] and draft_receipt['completed']
        fact = fact_receipt['parsed']
        assert fact['category'] == 'false_claim'
        assert fix.quotes.check(fact['false_claim_quote'], draft['attack_text'])['valid']
        framing = fix.quotes.classify(persona_receipt['parsed'], draft['attack_text'], row['assigned_name'])
        assert framing['alignment'] == 'aligned'
        fields = {'original_question': row['judge_input']['original_question'], 'attack_text': draft['attack_text']}
        assert persona_receipt['input_sha256'] == fix.persona.sha(fields)
        fact_fields = {'domain': item['domain'], 'original_question': item['question_data']['prompt'],
                       'accepted_reference_answer': item['question_data']['ground_truth'],
                       'candidate_attack_text': draft['attack_text']}
        if item['domain'] == 'coding':
            fact_fields['accepted_reference_full_source'] = item['question_data']['prompt'] + item['question_data']['ground_truth']
        assert fact_receipt['input_sha256'] == fix.persona.sha(fact_fields)
        replay = fix.verify_code(item, draft, {'reply': fact_receipt})
        assert replay['accepted'], f'Code replay failed: {review_id}'
        code_count += item['domain'] == 'coding'
    assert fix.persona.sha(fix.persona.DATASET.read_bytes()) == config['source_dataset_sha256']
    result = {'status': 'verified', 'verified_accepted_repairs': len(accepted),
              'repair_queue': len(selected), 'pending_repairs': len(selected) - len(accepted),
              'initial_repair_queue': summary['initial_repair_queue'],
              'reference_review_seed_holds': fix.reference_review_holds(),
              'code_argument_policy': fix.call_args.POLICY,
              'independently_replayed_code_cases': code_count,
              'unresolved_original_framing': len(unclear), 'quarantined_holds': len(holds),
              'source_dataset_sha256': config['source_dataset_sha256'],
              'candidate_dataset_sha256': summary['candidate_dataset_sha256'],
              'candidate_fields_verified': 'Only accepted attack_text leaves change; source references, corrections and other attacks preserved.',
              'blindness_verified': 'Factual review has no persona/disproof/test metadata; framing review has only question and candidate text.',
              'evidence_scope': 'Offline consistency and finite-code-case replay; not a new AI or human validation.',
              'timestamp_utc': fix.persona.now()}
    fix.persona.write_json(fix.OUT / 'verification.json', result)
    lines = [f'Gemini persona repair: {len(accepted)}/{len(selected)} accepted and verified.',
             f'Pending repairs: {len(selected) - len(accepted)}.',
             f'Original framing cases still unresolved: {len(unclear)}.',
             f'Quarantined mismatch holds: {len(holds)}.',
             'Additional source-reference holds: ' + ', '.join(r['seed_id'] for r in fix.reference_review_holds()),
             f'Accepted code examples replayed: {code_count}.', '',
             'Each accepted candidate passed separate Luna factual and blind persona checks.',
             'Candidate dataset: spine_dataset_persona_repaired_candidate.json',
             'Changed branches and source hashes: change_manifest.json',
             'Fresh target-model follow-ups are required for these changed attacks.',
             'Existing paper metrics remain linked to the original evaluated release.']
    (fix.OUT / 'report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


def alive():
    path = fix.OUT / 'runner.lock'
    if not path.exists():
        return False
    pid = json.loads(path.read_text(encoding='utf-8'))['pid']
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.WaitForSingleObject.restype = ctypes.c_uint32
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if not args.watch:
        if alive():
            raise SystemExit('The fixer is active. Read its summary now; run offline verification after the owner exits, or use --watch.')
        print(json.dumps(verify(), indent=2))
        return
    lock = fix.OUT / 'completion_worker.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps({'pid': os.getpid(), 'started_utc': fix.persona.now()}).encode('utf-8'))
    os.close(fd)
    try:
        while alive():
            fix.persona.write_json(fix.OUT / 'completion_worker_state.json',
                                  {'state': 'waiting_for_fixer', 'pid': os.getpid(), 'timestamp_utc': fix.persona.now()})
            time.sleep(30)
        result = verify()
        fix.persona.write_json(fix.OUT / 'completion_worker_state.json',
                              {'state': 'verified' if result['pending_repairs'] == 0 else 'needs_attention',
                               'pid': os.getpid(), 'verified_repairs': result['verified_accepted_repairs'], 'timestamp_utc': fix.persona.now()})
        print(json.dumps(result, indent=2), flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
