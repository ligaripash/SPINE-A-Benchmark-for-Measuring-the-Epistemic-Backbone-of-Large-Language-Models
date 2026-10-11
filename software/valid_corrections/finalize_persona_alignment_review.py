"""Verify literal quotation fragments without repeating any Luna requests.

The original strict-quotation labels remain immutable. A separate derived ledger
can accept an abbreviated quotation only when its literal fragments occur in
order in the attack. No paraphrase, fuzzy semantic match or new judge is used.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

import persona_alignment_review as review

POLICY = {
    'version': 'literal_ordered_quotation_fragments_v2',
    'preserves': 'The Luna requests, raw judgments, strict labels and frozen review manifest remain unchanged.',
    'extension': 'An ellipsis-abbreviated quotation is accepted only when all nonempty literal fragments occur in source order after whitespace normalization; at least two fragments must each have at least eight characters and together at least twenty characters.',
    'excludes': 'No fuzzy matching, paraphrase matching, changed persona definitions or new model requests.',
    'expanded_quote': 'The exact normalized source span between the first and last matched fragment is recorded for inspection.',
    'purpose': 'Separate quotation formatting from framing disagreement.',
}


def match_quote(quote, attack):
    source = review.normalize_quote(attack)
    q = review.normalize_quote(quote)
    if q and q in source:
        return {'valid': True, 'method': 'contiguous_literal', 'verified_source_span': q}
    fragments = [s.strip() for s in re.split(r'\.{3,}|…+', q) if s.strip()]
    meaningful = [f for f in fragments if len(f) >= 8]
    if len(meaningful) < 2 or sum(map(len, meaningful)) < 20:
        return {'valid': False, 'method': 'unsupported_quote', 'verified_source_span': ''}
    cursor = 0
    first = None
    for fragment in fragments:
        position = source.find(fragment, cursor)
        if position < 0:
            return {'valid': False, 'method': 'unsupported_quote', 'verified_source_span': ''}
        if first is None:
            first = position
        cursor = position + len(fragment)
    return {'valid': True, 'method': 'ordered_literal_fragments',
            'verified_source_span': source[first:cursor]}


def derive(record, item):
    source = item['judge_input']['attack_text']
    judgment = record['judgment']
    checks = [{'persona': e['persona'], 'original_quote': e['quote'], **match_quote(e['quote'], source)}
              for e in judgment['evidence']]
    supported = {c['persona'] for c in checks if c['valid']}
    dom = set(judgment['dominant_personas'])
    secondary = set(judgment['secondary_personas'])
    if not (dom | secondary) <= supported:
        alignment, cause = 'unresolved', 'quotation_validation_failed'
    elif judgment['decision'] == 'uncertain':
        alignment, cause = 'unresolved', 'uncertain_framing'
    elif judgment['decision'] == 'no_clear_persona':
        alignment, cause = 'not_aligned', 'no_clear_persona'
    elif item['assigned_name'] in dom:
        alignment, cause = 'aligned', 'assigned_persona_is_dominant'
    elif item['assigned_name'] in secondary:
        alignment, cause = 'not_aligned', 'assigned_persona_only_secondary'
    else:
        alignment, cause = 'not_aligned', 'different_dominant_persona'
    return {**record, 'strict_alignment_v1': record['alignment'],
            'strict_alignment_reason_v1': record['alignment_reason'],
            'alignment': alignment, 'alignment_reason': cause,
            'quotation_policy_version': POLICY['version'],
            'source_quotation_checks': checks}


def finish():
    manifest = json.loads((review.OUT / 'manifest.json').read_text(encoding='utf-8'))
    assert review.sha(review.DATASET.read_bytes()) == manifest['dataset_sha256']
    assert review.sha((review.OUT / 'packet.jsonl').read_bytes()) == manifest['packet_sha256']
    packet = review.read_jsonl(review.OUT / 'packet.jsonl')
    items = {r['review_id']: r for r in packet}
    raw = review.checked_labels(packet)
    derived = [derive(row, items[row['review_id']]) for row in raw]
    assert len(derived) == len({r['review_id'] for r in derived})
    for row in derived:
        for check in row['source_quotation_checks']:
            if check['valid']:
                assert check['verified_source_span'] in review.normalize_quote(items[row['review_id']]['judge_input']['attack_text'])
    flags = [r for r in derived if r['alignment'] != 'aligned']
    # Separate names ensure the runner's strict summaries are never overwritten.
    output_text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in derived)
    flags_text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in flags)
    for filename, text in [('validated_labels.jsonl', output_text), ('final_flags.jsonl', flags_text)]:
        path = review.OUT / filename
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(text, encoding='utf-8')
        os.replace(temporary, path)
    groups = {}
    for field in ('assigned_name', 'domain', 'creator'):
        groups[field] = {name: dict(Counter(r['alignment'] for r in derived if r[field] == name))
                         for name in sorted({r[field] for r in packet})}
    result = {'status': 'complete' if len(raw) == len(packet) else 'incomplete',
              'completed': len(raw), 'total': len(packet), 'pending': len(packet) - len(raw),
              'requested_model': review.MODEL, 'reasoning_effort': review.EFFORT,
              'dataset_sha256': manifest['dataset_sha256'], 'packet_sha256': manifest['packet_sha256'],
              'rubric_sha256': manifest['rubric_sha256'], 'quotation_policy': POLICY,
              'alignment_counts': dict(Counter(r['alignment'] for r in derived)),
              'strict_alignment_counts_v1': dict(Counter(r['alignment'] for r in raw)),
              'flag_reasons': dict(Counter(r['alignment_reason'] for r in flags)),
              'by_persona': groups['assigned_name'], 'by_domain': groups['domain'], 'by_creator': groups['creator'],
              'co_dominant': sum(len(r['judgment']['dominant_personas']) > 1 for r in derived),
              'quotation_only_reclassifications': sum(r['alignment'] != r['strict_alignment_v1'] for r in derived),
              'records_using_literal_ellipsis_validation': sum(any(c['method'] == 'ordered_literal_fragments' and c['valid']
                                                                  for c in r['source_quotation_checks']) for r in derived),
              'validated_labels_sha256': review.text_sha(output_text), 'flags_sha256': review.text_sha(flags_text),
              'generated_utc': review.now(),
              'limitations': 'AI persona-framing assessment, not human validation. Factual validity and benchmark outputs were not reevaluated. Source assignments and attacks remain unchanged.'}
    review.write_json(review.OUT / 'quotation_validation_policy_v2.json', POLICY)
    review.write_json(review.OUT / 'final_summary.json', result)
    report = [f"Luna persona-alignment review: {result['status']}",
              f"Completed: {len(raw)} / {len(packet)}; pending: {len(packet) - len(raw)}.",
              'Counts use source-verified quotations, including ordered literal ellipsis fragments.',
              'These are AI framing judgments; factual validity and benchmark metrics are separate.', '']
    for name in review.PERSONAS:
        counts = result['by_persona'][name]
        report.append(f"{name}: aligned={counts.get('aligned', 0)}, flagged={counts.get('not_aligned', 0)}, unresolved={counts.get('unresolved', 0)}")
    report += ['', 'Raw Luna judgments: labels.jsonl',
               'Derived source-verified judgments: validated_labels.jsonl',
               'Non-aligned and unresolved records: final_flags.jsonl',
               'Full summary: final_summary.json',
               'The original dataset, assignments, references and model responses were not changed.']
    (review.OUT / 'report.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
    return result


def owner_alive():
    lock = review.OUT / 'runner.lock'
    if not lock.exists():
        return False
    pid = json.loads(lock.read_text(encoding='utf-8'))['pid']
    if os.name == 'nt':
        # Never use os.kill(pid, 0) for Windows process probing.
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.WaitForSingleObject.restype = ctypes.c_uint32
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied is not proof of death.
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
    parser.add_argument('--watch', action='store_true', help='Wait for the existing runner; never makes API requests.')
    args = parser.parse_args()
    if not args.watch:
        print(json.dumps(finish(), indent=2))
        return
    lock_path = review.OUT / 'completion_worker.lock'
    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps({'pid': os.getpid(), 'started_utc': review.now()}).encode('utf-8'))
    os.close(fd)
    try:
        while True:
            # The runner appends whole JSON lines; read at a stable batch boundary.
            state = json.loads((review.OUT / 'run_state.json').read_text(encoding='utf-8'))
            if state['state'] in ('running', 'waiting_for_credential') and owner_alive():
                review.write_json(review.OUT / 'completion_worker_state.json',
                                  {'state': 'waiting_for_runner', 'pid': os.getpid(), 'timestamp_utc': review.now()})
                time.sleep(30)
                continue
            result = finish()
            review.write_json(review.OUT / 'completion_worker_state.json',
                              {'state': result['status'] if result['status'] == 'complete' else 'runner_needs_attention',
                               'pid': os.getpid(), 'completed': result['completed'], 'timestamp_utc': review.now()})
            print(json.dumps(result, indent=2), flush=True)
            return
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
