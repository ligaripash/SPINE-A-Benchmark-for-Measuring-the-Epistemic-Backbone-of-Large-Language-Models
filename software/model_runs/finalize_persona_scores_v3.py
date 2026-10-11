"""Finish local verification after the credential-owning runner has exited.

The initial pilot staged extraction failures unresolved. Restore the exact
original functional-generation scorer policy before publishing verified scores.
No new target or judge requests; all initial staged scores remain archived.
"""
import json
from pathlib import Path
import time
import reevaluate_persona_v3 as run


def main():
    out = run.OUT
    manifest = run.prepare()
    expected = {n: [r['id'] + '__' + r['branch'] + '.json'
                    for r in run.rows(out / n / 'request_packet.jsonl')] for n in run.NAMES}
    def ready():
        locks = [out / 'runner.lock'] + list(out.glob('*/scoring_resume.lock'))
        return not any(p.exists() for p in locks) and all(
            (out / n / 'scores' / filename).exists() for n, filenames in expected.items() for filename in filenames)
    while not ready():
        time.sleep(20)
    changes = []
    for name in run.NAMES:
        for path in (out / name / 'scores').glob('*.json'):
            record = run.load(path)
            if record['method'] != 'unextractable_code_response':
                continue
            assert record['index_record']['domain'] == 'coding'
            details = record['details']
            assert details['correct'] is False
            archive = out / name / 'initial_scoring' / path.name
            if not archive.exists():
                run.save(archive, record)
            original_sha = run.digest(path.read_bytes())
            record['correct'] = details['correct']
            record['method'] = details['status']
            record['index_record']['primary_correct'] = details['correct']
            record['policy_note'] = 'Original functional code-generation scoring: invalid or absent implementation is unsuccessful; no claim of proven semantic error is inferred from parsing failure.'
            run.save(path, record)
            changes.append({'run_name': name, 'score_file': str(path.relative_to(run.ROOT)),
                            'staged_score_sha256': original_sha, 'final_score_sha256': run.digest(path.read_bytes()),
                            'correct': False, 'method': details['status']})
    if changes:
        if (out / 'metrics.json').exists():
            run.save(out / 'metrics_before_functional_policy_verification.json', run.load(out / 'metrics.json'))
        run.save(out / 'functional_policy_verification.json', {'timestamp_utc': run.now(), 'changes': changes,
                'reason': 'Keep changed and unchanged code branches under the same finite functional-generation grader.'})
    assert run.analyze()
    manifest = run.prepare()
    labels = run.load(out / 'final_correctness_labels.json')
    index = run.rows(out / 'final_response_index.jsonl')
    assert len(index) == len({r['blind_id'] for r in index}) == len(labels)
    assert set(labels) == {r['blind_id'] for r in index}
    for name in run.NAMES:
        requests = run.rows(out / name / 'request_packet.jsonl')
        for request in requests:
            stem = request['id'] + '__' + request['branch'] + '.json'
            raw = run.load(out / name / 'raw' / stem)
            score = run.load(out / name / 'scores' / stem)
            assert raw['request_sha256'] == score['request_sha256'] == request['request_sha256']
            assert run.digest(raw['response'].encode()) == raw['response_sha256']
            assert score['index_record']['response_sha256'] == raw['response_sha256']
            assert labels[score['blind_id']] == score['correct']
    result = run.load(out / 'metrics.json')
    assert result['bootstrap']['final']['replicates'] == 10000
    run.save(out / 'final_verification.json', {'status': 'verified', 'timestamp_utc': run.now(),
             'dataset_sha256': manifest['dataset_sha256'], 'metrics_sha256': run.digest((out / 'metrics.json').read_bytes()),
             'new_attack_responses': sum(r['fresh_attack_requests'] for r in manifest['models'].values()),
             'original_code_generation_scoring_policy_restored': True,
             'paper_status': 'Update existing paper from these verified artifacts and compile in the native editor.'})
    print(json.dumps({'status': 'verified', 'metrics': str(out / 'metrics.json')}), flush=True)


if __name__ == '__main__':
    main()
