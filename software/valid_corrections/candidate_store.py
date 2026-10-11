"""Load immutable model drafts with explicit assistant-authored repair overlays."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DRAFTS = HERE / 'reasoned_candidate_drafts.jsonl'
MANUAL_REPAIRS = HERE / 'manual_candidate_repairs.jsonl'


def load_candidates() -> dict[str, dict]:
    rows = {}
    if MODEL_DRAFTS.exists():
        for line in MODEL_DRAFTS.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                if row['id'] in rows:
                    raise ValueError('duplicate model draft: ' + row['id'])
                rows[row['id']] = row
    if MANUAL_REPAIRS.exists():
        for line in MANUAL_REPAIRS.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                if row['id'] not in rows:
                    raise ValueError('repair has no model draft: ' + row['id'])
                row['supersedes_generator'] = rows[row['id']]['generator']
                rows[row['id']] = row
    return rows
