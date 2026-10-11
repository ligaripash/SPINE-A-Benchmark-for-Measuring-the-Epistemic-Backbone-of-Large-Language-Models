"""Verify packaged bytes, provenance, and the final response/label alignment."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_verified():
    manifest = json.loads((ROOT / 'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    for name, spec in manifest.items():
        path = ROOT / name
        assert path.is_file(), f'Missing artifact: {name}'
        data = path.read_bytes()
        assert len(data) == spec['bytes'] and sha(data) == spec['sha256'], f'Artifact differs: {name}'
    raw = gzip.decompress((ROOT / 'results/final_response_index.jsonl.gz').read_bytes())
    verification = json.loads((ROOT / 'results/final_verification.json').read_text())
    for name, expected in verification['hashes'].items():
        content = raw if name.endswith('.jsonl') else (ROOT / 'results' / name).read_bytes()
        assert sha(content) == expected, f'Source hash differs: {name}'
    index = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    labels = json.loads((ROOT / 'results/final_correctness_labels.json').read_text())
    assert len(index) == len({r['blind_id'] for r in index}) == verification['unique_records']
    assert set(labels) == {r['blind_id'] for r in index}
    for row in index:
        response = row['judge_input']['final_response']
        assert row['response_sha256'] in {sha(response.encode()), sha(json.dumps(response, ensure_ascii=False, sort_keys=True).encode())}
        assert labels[row['blind_id']] is row['primary_correct']
    return index, labels


if __name__ == '__main__':
    rows, labels = load_verified()
    print(f'PASS: package hashes, {len(rows):,} unique responses, response hashes, and final labels.')
