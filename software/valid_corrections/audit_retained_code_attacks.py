"""Offline finite audit of retained GPT-4o code attacks; never rewrites data.

Literal attack quotations are linked to executable proposals. Missing or broad
claims are flagged rather than converted into invented input/output assertions.
"""
from __future__ import annotations

import argparse
import ast
import copy
import concurrent.futures
import gzip
import hashlib
import json
import re
import subprocess
import sys
import textwrap
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'valid_corrections/retained_code_attack_audit'
DATASET = ROOT / 'spine_dataset_corrective_branch_luna_reviewed_v2.json'
MANIFEST = ROOT / 'valid_corrections/attack_false_claim_repair_v2/applied_manifest.json'
LUNA = ROOT / 'valid_corrections/ai_review_gpt-6-luna_full/labels.jsonl'
TESTS = ROOT / 'original_validation/source_cache/HumanEvalPlus-Mini.jsonl.gz'
WORKER = Path(__file__).with_name('retained_code_attack_worker.py')
EDGE_INPUTS = {
    'code_2': [[3.999999999999999]],
    'code_16': [['ςΣ']],
    'code_42': [[[9007199254740992]]],
    'code_65': [[12, 4]],
    'code_76': [[0, 2]],
    'code_99': [['1.4999999999999999']],
    'code_149': [[['a', 'b']]],
    'code_153': [['K', ['Ⅻ', 'A']]],
    'code_158': [[['éαz', 'bb']]],
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def load_packet():
    data = json.loads(DATASET.read_bytes())
    changed = {(r['id'], r['persona']) for r in json.loads(MANIFEST.read_bytes())['changes']}
    labels = {(r['id'], r['persona']): r for r in rows(LUNA) if r['unit'] == 'attack'}
    packet = []
    for item in data:
        if item['domain'] != 'coding':
            continue
        for persona, attack in item['adversarial_attacks'].items():
            if (item['id'], persona) in changed:
                continue
            text = attack['attack_text']
            label = labels[(item['id'], persona)]
            assert label['label'] == 'false_claim'
            assert sha(text.encode()) == label['followup_sha256']
            packet.append({'id': item['id'], 'persona': persona,
                           'attack_text': text, 'attack_sha256': sha(text.encode()),
                           'source': item['question_data']['prompt'] + item['question_data']['ground_truth'],
                           'prompt': item['question_data']['prompt'],
                           'luna_reason': label['parsed']['reason'],
                           'luna_concrete_check': label['parsed']['concrete_check']})
    assert len(packet) == 504 and len({(r['id'], r['persona']) for r in packet}) == 504
    return packet


def quotations(text):
    found = []
    for match in re.finditer(r'```(?:python|py)?\s*\n?(.*?)```', text, re.S):
        found.append((match.group(0), textwrap.dedent(match.group(1)).strip(), match.start(), match.end()))
    fenced = [(start, end) for _, _, start, end in found]
    for match in re.finditer(r'(?<!`)`([^`\n]+)`(?!`)', text):
        if not any(start <= match.start() < end for start, end in fenced):
            found.append((match.group(0), match.group(1).strip(), match.start(), match.end()))
    return found


def literal_calls(text, entry):
    calls = []
    for match in re.finditer(r'\b' + re.escape(entry) + r'\s*\(', text):
        # Find balanced close using Python tokenize/AST progressively, preserving
        # the exact source quotation. No evaluation of supplied expressions.
        for end in range(match.end(), min(len(text), match.start() + 1200)):
            if text[end] != ')':
                continue
            quote = text[match.start():end + 1]
            try:
                call = ast.parse(quote, mode='eval').body
                if not isinstance(call, ast.Call):
                    continue
                args = [ast.literal_eval(arg) for arg in call.args]
                kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
                if kwargs and args or None in kwargs:
                    continue
                calls.append({'quote': quote, 'args': kwargs or args,
                              'tail': text[end + 1:end + 1 + 180]})
                break
            except (ValueError, SyntaxError, TypeError):
                continue
    return calls


def proposals(row, entry):
    ref = ast.parse(row['source'])
    func = next(n for n in ref.body if isinstance(n, ast.FunctionDef) and n.name == entry)
    signature = 'def ' + entry + '(' + ast.unparse(func.args) + '):\n'
    found = []
    seen = set()
    for quote, code, start, end in quotations(row['attack_text']):
        # Inline Markdown sometimes has a first line at column zero but its
        # remaining lines are uniformly indented by four spaces.
        raw_lines = code.splitlines()
        if len(raw_lines) > 1 and all(not line.strip() or line.startswith('    ') for line in raw_lines[1:]):
            try:
                ast.parse(code)
            except SyntaxError:
                code = raw_lines[0] + '\n' + textwrap.dedent('\n'.join(raw_lines[1:]))
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        mode, source = None, None
        if any(isinstance(n, ast.FunctionDef) and n.name == entry for n in tree.body):
            # Only definitions/imports, no top-level side effects or test calls.
            if all(isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom)) for n in tree.body):
                mode, source = 'quoted_function', code
        elif len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef):
            proposed = copy.deepcopy(tree.body[0])
            if len(proposed.args.args) == len(func.args.args):
                old_name = proposed.name
                proposed.name = entry
                class Rename(ast.NodeTransformer):
                    def visit_Name(self, node):
                        if node.id == old_name:
                            node.id = entry
                        return node
                proposed = Rename().visit(proposed)
                mode, source = 'quoted_function_renamed_entry', ast.unparse(proposed)
        elif len(tree.body) == 1 and isinstance(tree.body[0], ast.Return):
            # A standalone quoted return is a candidate complete return rule.
            mode, source = 'quoted_return_rule', signature + textwrap.indent(code, '    ')
            referenced = {n.id for n in ast.walk(tree.body[0]) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            local_assignments = {n.targets[0].id for n in func.body if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
            if referenced & local_assignments and isinstance(func.body[-1], ast.Return):
                # Reuse the original straight-line setup for variables such as
                # max_number or s. Do not omit guards and helpers.
                revised = copy.deepcopy(func)
                revised.body[-1] = copy.deepcopy(tree.body[0])
                mode, source = 'quoted_final_return_replacement', ast.unparse(revised)
        elif len(tree.body) == 1 and isinstance(tree.body[0], ast.If) and not tree.body[0].orelse:
            # An isolated guard does not replace the whole function. Insert it
            # before the original body so false branches retain task behavior.
            revised = copy.deepcopy(func)
            revised.body = [copy.deepcopy(tree.body[0])] + revised.body
            mode, source = 'quoted_guard_insertion', ast.unparse(revised)
        elif tree.body and all(isinstance(n, (ast.For, ast.While, ast.If, ast.Try, ast.Return, ast.Assign, ast.AugAssign, ast.Expr, ast.FunctionDef, ast.Import, ast.ImportFrom)) for n in tree.body) and any(isinstance(n, ast.Return) for n in ast.walk(tree)):
            mode, source = 'quoted_function_body', signature + textwrap.indent(code, '    ')
        if source is not None:
            # Preserve task helpers and imports when replacing its function.
            # Otherwise an isolated return could fail only because a helper
            # defined in the original prompt was omitted from the audit.
            replacement = ast.parse(source)
            module = copy.deepcopy(ref)
            module.body = [n for n in module.body if not (isinstance(n, ast.FunctionDef) and n.name == entry)] + replacement.body
            source = ast.unparse(ast.fix_missing_locations(module))
        if source is not None and source not in seen:
            try:
                ast.parse(source)
            except SyntaxError:
                continue
            found.append({'kind': mode, 'quote': quote, 'quote_span': [start, end],
                          'source': source, 'source_sha256': sha(source.encode()),
                          'needs_semantic_review': True})
            seen.add(source)
        # Interpret a literal assignment to a variable already assigned in the
        # reference as a local replacement, retaining all surrounding logic.
        if len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign) and len(tree.body[0].targets) == 1 and isinstance(tree.body[0].targets[0], ast.Name):
            target = tree.body[0].targets[0].id
            class Replace(ast.NodeTransformer):
                replaced = 0
                def visit_Assign(self, node):
                    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == target:
                        self.replaced += 1
                        return copy.deepcopy(tree.body[0])
                    return self.generic_visit(node)
            replacer = Replace()
            module = replacer.visit(copy.deepcopy(ref))
            if replacer.replaced == 1:
                source = ast.unparse(ast.fix_missing_locations(module))
                if source not in seen:
                    found.append({'kind': 'quoted_assignment_replacement', 'quote': quote, 'quote_span': [start, end],
                                  'source': source, 'source_sha256': sha(source.encode()), 'needs_semantic_review': True})
                    seen.add(source)
    return found


def run_worker(payload):
    try:
        result = subprocess.run([sys.executable, '-X', 'utf8', '-I', '-S', str(WORKER)],
                                input=json.dumps(payload), text=True, encoding='utf-8',
                                capture_output=True, timeout=10, check=False)
        if result.returncode:
            return {'status': 'worker_error', 'reason': result.stderr[-300:]}
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return {'status': 'timeout'}
    except json.JSONDecodeError:
        return {'status': 'worker_invalid_output'}


def audit(row, ep):
    record = {k: row[k] for k in ('id', 'persona', 'attack_sha256', 'attack_text',
                                 'luna_reason', 'luna_concrete_check')}
    record['reference_sha256'] = sha(row['source'].encode())
    if row['id'] == 'code_39':
        record.update(status='reference_quarantined', reason='Original reference is quarantined; no falsity conclusion from its execution.', attempts=[])
        return record
    entry = ep['entry_point']
    inputs = []
    for name in ('base_input', 'plus_input'):
        for index, args in enumerate(ep[name]):
            if len(json.dumps(args)) < 12000:
                inputs.append({'args': args, 'provenance': f'HumanEvalPlus-Mini:{ep["task_id"]}:{name}:{index}'})
    for args in EDGE_INPUTS.get(row['id'], []):
        inputs.append({'args': args, 'provenance': 'audit_authored_edge_input'})
    direct_calls = literal_calls(row['attack_text'], entry)
    for call in direct_calls:
        inputs.insert(0, {'args': call['args'], 'provenance': 'literal_call_in_attack', 'call_quote': call['quote']})
    record['literal_calls'] = direct_calls
    candidates = proposals(row, entry)
    record['attempts'] = []
    for candidate in candidates:
        attempt = dict(candidate)
        attempt['execution'] = run_worker({'reference': row['source'], 'proposal': candidate['source'],
                                          'entry_point': entry, 'cases': inputs, 'atol': ep.get('atol', 0),
                                          'task_id': ep['task_id']})
        record['attempts'].append(attempt)
    record['status'] = 'pending_semantic_review' if any(a['execution'].get('counterexample') for a in record['attempts']) else 'not_established'
    record['reason'] = ('Counterexample needs contextual confirmation that the tested quotation is an endorsed proposal.'
                        if record['status'] == 'pending_semantic_review' else
                        'No executable proposal found.' if not candidates else
                        'No reliable counterexample established by the executable proposals tested; see attempt details.')
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    packet = load_packet()
    (OUT / 'packet.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in packet), encoding='utf-8')
    ep = {r['task_id']: r for r in (json.loads(line) for line in gzip.open(TESTS, 'rt', encoding='utf-8'))}
    if args.inventory:
        for row in packet:
            e = ep['HumanEval/' + row['id'].split('_')[-1]]
            candidates = proposals(row, e['entry_point'])
            print(json.dumps({'id': row['id'], 'persona': row['persona'], 'quotes': [c['quote'] for c in candidates],
                              'calls': literal_calls(row['attack_text'], e['entry_point'])}, ensure_ascii=False))
        return
    path = OUT / 'execution_records.jsonl'
    existing = {(r['id'], r['persona']): r for r in rows(path)} if path.exists() else {}
    for row in packet:
        prior = existing.get((row['id'], row['persona']))
        if prior:
            assert prior['attack_sha256'] == row['attack_sha256']
            assert prior['reference_sha256'] == sha(row['source'].encode())
    pending = [row for row in packet if (row['id'], row['persona']) not in existing]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool, path.open('a', encoding='utf-8') as handle:
        futures = {pool.submit(audit, row, ep['HumanEval/' + row['id'].split('_')[-1]]): row for row in pending}
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()
            existing[(record['id'], record['persona'])] = record
            if len(existing) % 50 == 0:
                print(json.dumps({'complete': len(existing), 'status': dict(Counter(r['status'] for r in existing.values()))}), flush=True)
    print(json.dumps({'complete': len(existing), 'status': dict(Counter(r['status'] for r in existing.values()))}), flush=True)


if __name__ == '__main__':
    main()
