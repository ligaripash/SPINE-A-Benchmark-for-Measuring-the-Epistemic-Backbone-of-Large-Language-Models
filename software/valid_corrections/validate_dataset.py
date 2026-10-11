"""Validate authored data and execute ONLY bundled, inspected reference code.

This is not a sandbox for model-generated code. Candidate code needs an isolated
runner. This script makes no claim of independent scientific/human validation.
"""
import ast
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def equivalent(a, b):
    # Python considers True == 1; the revised contracts often require a bool.
    if isinstance(b, bool):
        return type(a) is bool and a == b
    if isinstance(b, list):
        return isinstance(a, list) and len(a) == len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if isinstance(b, (int, float)):
        return type(a) in (int, float) and abs(a-b) < 1e-9
    return type(a) is type(b) and a == b


def load_function(source, name):
    ast.parse(source)
    namespace = {}
    exec(compile(source, '<authored-reference>', 'exec'), namespace)
    return namespace[name]


def validate():
    raw = (ROOT / 'spine_dataset.json').read_bytes()
    original = json.loads(raw)
    full = json.loads((ROOT / 'spine_dataset_with_valid_corrections.json').read_text(encoding='utf-8'))
    pilot = json.loads((ROOT / 'valid_corrections/pilot_dataset.json').read_text(encoding='utf-8'))
    messages = json.loads((ROOT / 'valid_corrections/target_followups.json').read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / 'valid_corrections/manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(raw).hexdigest() == manifest['source_sha256']
    assert len(original) == len(full) == 764
    assert len(pilot) == len(messages) == 60
    assert Counter(x['domain'] for x in pilot) == {'math':20, 'logic':20, 'coding':20}
    assert [x for x in full if x['valid_corrective_intervention']] == pilot
    for old, new in zip(original, full):
        assert {k: new[k] for k in old} == old, old['id']
        assert len(new['adversarial_attacks']) == 6
        if new['valid_corrective_intervention'] is None:
            assert new['valid_correction_status'] == 'not_authored_exclude_from_update_evaluation'
    report = {'source_unchanged': True, 'original_fields_preserved': True,
              'authored_items': 60, 'unauthored_items': 704,
              'independent_human_review': 'pending',
              'code_test_cases': 0, 'code_items_with_original_failure': 0,
              'checks': []}
    for item, message in zip(pilot, messages):
        c = item['valid_corrective_intervention']
        assert c['ground_truth'] != item['question_data']['ground_truth'], item['id']
        assert message['id'] == item['id']
        assert set(message) == {'id','domain','intervention_id','followup_user_message'}
        assert message['followup_user_message'] == c['intervention_text']
        assert c['validation']['independent_human_review'] == 'pending'
        assert c['intervention_text'] and c['rationale']
        check = {'id':item['id'], 'domain':item['domain']}
        if item['domain'] == 'coding':
            revised = load_function(c['ground_truth'], c['entry_point'])
            old = load_function(item['question_data']['prompt']+item['question_data']['ground_truth'], c['entry_point'])
            failures = []
            for i, test in enumerate(c['test_cases']):
                actual = revised(*copy.deepcopy(test['args']))
                assert equivalent(actual, test['expected']), (item['id'], i, actual, test)
                old_actual = old(*copy.deepcopy(test['args']))
                if not equivalent(old_actual, test['expected']):
                    failures.append(i)
            assert failures, f"Original solution still passes all tests: {item['id']}"
            check.update(revised_passes=len(c['test_cases']), original_fails=len(failures),
                         distinguishing_test_indices=failures)
            report['code_test_cases'] += len(c['test_cases'])
            report['code_items_with_original_failure'] += 1
        elif item['domain'] == 'math':
            from decimal import Decimal
            assert Decimal(c['ground_truth']) != Decimal(c['original_ground_truth'].replace(',',''))
            assert c['edit']['new'] in c['revised_question']
            check['check'] = 'changed_numeric_answer_and_explicit_premise_edit'
            if item['id'] == 'math_12':
                n = int(c['ground_truth'])
                assert n*(7*1.5-3)-105 > 0
                assert (n-1)*(7*1.5-3)-105 <= 0
            if item['id'] == 'math_214':
                s = int(c['ground_truth'])
                assert 4*s+6*4 == 2*(s+6*4)
        else:
            assert c['options'][c['correct_option']] == c['ground_truth']
            assert len(c['options']) == len(set(c['options'].values())) == 4
            check['check'] = 'revised_option_key_consistent; semantic_review_pending'
        report['checks'].append(check)
    # Always retain the original semantic answer/program: R=1 and U=0 on
    # the authored changed-answer golds. This is a baseline, not a model result.
    report['always_retain_baseline'] = {'R': 1.0, 'U': 0.0, 'evidence_discrimination': 0.0,
                                       'status': 'by_construction; executable_original_code_checked'}
    (ROOT / 'valid_corrections/validation_report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'checks'}, indent=2))


if __name__ == '__main__':
    validate()
