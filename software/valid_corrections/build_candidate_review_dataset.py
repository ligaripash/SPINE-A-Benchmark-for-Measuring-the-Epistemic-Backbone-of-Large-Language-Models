"""Materialize all correction records into a review-only dataset artifact."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from candidate_store import load_candidates

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'spine_dataset_correction_review.json'


def main():
    raw = (ROOT / 'spine_dataset.json').read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    original = json.loads(raw)
    augmented = json.loads((ROOT / 'spine_dataset_with_valid_corrections.json').read_text(encoding='utf-8'))
    drafts = load_candidates()
    smoke_path = ROOT / 'valid_corrections/code_candidate_smoke.jsonl'
    smoke = {x['id']: x for x in (json.loads(line) for line in
             smoke_path.read_text(encoding='utf-8').splitlines() if line.strip())} if smoke_path.exists() else {}
    if len(original) != len(augmented) or len(drafts) != 704:
        raise ValueError('incomplete source or candidate set')
    items = []
    for old, enriched in zip(original, augmented):
        if old['id'] != enriched['id'] or any(enriched[k] != value for k,value in old.items()):
            raise ValueError('original fields changed: ' + old['id'])
        row = drafts.get(old['id'])
        if enriched['valid_corrective_intervention'] is None:
            if not row or row['source_sha256'] != source_hash:
                raise ValueError('candidate missing or source mismatch: ' + old['id'])
            candidate = row.get('candidate') or {}
            review = {'status': 'no_usable_draft_original_task_adjudication_required'
                      if candidate.get('unable') or not candidate else 'candidate_only_not_validated',
                      'generator': row['generator'],
                      'screen_flags': row['screen_flags'],
                      'candidate': candidate,
                      'code_smoke': smoke.get(old['id']),
                      'supersedes_generator': row.get('supersedes_generator')}
        else:
            if row:
                raise ValueError('candidate overlaps pilot: ' + old['id'])
            review = None
        items.append({**enriched, 'correction_candidate_for_review': review})
    counts = Counter('pilot' if x['valid_corrective_intervention'] else
                     x['correction_candidate_for_review']['status'] for x in items)
    packet = {'artifact_type': 'SPINE correction review packet, not benchmark gold',
              'source_sha256': source_hash, 'record_count': len(items),
              'counts': dict(counts), 'items': items}
    OUT.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'counts': dict(counts)}))


if __name__ == '__main__':
    main()
