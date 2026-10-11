"""Resolve two concrete coverage concerns found during response inspection.

code_32 must preserve nonzero-constant polynomial behavior. code_149 requires
full reverse lexicographic order, not a comparison of only final characters.
Top-level demonstration assertions/calls are omitted; function bodies stay intact.
"""
import argparse
import ast
import json
from pathlib import Path
import reevaluate as v

CASES = {
    'code_32': [([[1, 1]], -1.0), ([[2, 2]], -1.0)],
    'code_149': [([['ab', 'ba']], ['ba', 'ab']), ([['az', 'ba']], ['ba', 'az'])],
}


def cleaned_function(code):
    tree = ast.parse(code)
    tree.body = [node for node in tree.body if not isinstance(node, ast.Assert) and
                 not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call))]
    return ast.unparse(tree)


def with_original_helpers(code, item):
    if item['id'] != 'code_32':
        return code
    # The revised task explicitly retains the original poly helper. The source
    # prompt ends at an empty find_zero declaration, so parse its helper prefix.
    prefix = item['question_data']['prompt'].split('def find_zero(')[0]
    supplied = ast.parse(code)
    names = {n.name for n in supplied.body if isinstance(n, ast.FunctionDef)}
    if 'poly' not in names:
        return prefix + '\n' + code
    return code


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', choices=v.NAMES, default=list(v.NAMES))
    args = p.parse_args()
    items = v.data()
    independent = v.load_module(v.AUDIT/'independent_code_cases.py', 'supplemental_cases_v1')
    for name in args.models:
        directory = v.HERE/name
        if not (directory/'correction_scores.jsonl').exists() or (directory/'scoring.lock').exists():
            continue
        scores = {x['id']:x for x in v.rows(directory/'correction_scores.jsonl')}
        raw = {x['id']:x for x in v.rows(directory/'correction_reused.jsonl')}
        raw.update({x['id']:x for x in v.rows(directory/'correction_new.jsonl') if 'error' not in x})
        path = directory/'response_adjudications.json'
        overrides = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        evidence = []
        for ident, extra in CASES.items():
            if ident not in scores:
                continue
            item = items[ident]
            target = item['valid_corrective_intervention']
            cases = [{'args':list(a),'expected_repr':repr(e)} for a,e in independent.CASES[ident]]
            cases += [{'args':a,'expected_repr':repr(e)} for a,e in extra]
            reference_code = with_original_helpers(target['ground_truth'], item)
            reference = v.audit_worker.execute(ident, reference_code, target['entry_point'], cases)
            assert reference['status']=='passed', (ident, reference)
            try:
                code = with_original_helpers(cleaned_function(v.extract_code(raw[ident]['response'],target['entry_point'])),item)
            except SyntaxError:
                continue
            result = v.audit_worker.execute(ident,code,target['entry_point'],cases)
            evidence.append({'id':ident,'reference_result':reference,'model_result':result,
                             'supplemental_cases':extra,'function_code_sha256':v.sha(code),'cases_sha256':v.sha(cases),
                             'extraction':'Removed top-level demonstration assertions/calls only; function bodies unchanged'})
            if result['status'] not in ('passed','failed'):
                continue
            scored = scores[ident]
            correct = result['status']=='passed'
            if scored['correct'] is not correct or ident in overrides:
                overrides[ident]={'correct':correct,'previous_correct':scored['correct'],
                    'reason':'Executed original independent cases plus task-derived boundary/counterexample cases after omitting top-level demonstrations. See supplemental_code_checks.json.',
                    'response_category':'finite_test_answer','adjudicator':'independent_expected_output_execution',
                    'input_sha256':scored['input_sha256'],'target_sha256':scored['target_sha256']}
        v.save(directory/'supplemental_code_checks.json',evidence)
        v.save(path,overrides)
        v.metrics(name)
        print(name,[(x['id'],x['model_result']['status']) for x in evidence])


if __name__=='__main__':
    main()
