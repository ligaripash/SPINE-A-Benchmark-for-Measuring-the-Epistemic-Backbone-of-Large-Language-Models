"""Recompute all published metric rows and both 10,000-replicate bootstraps offline."""
import argparse
import json
import math
import platform
from pathlib import Path
import numpy as np
from metrics_core import NAMES, metric_rows, bootstrap_persona
from verify_release import ROOT, load_verified


def compare(actual, expected, path='root'):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path + ': keys differ'
        for key in expected:
            compare(actual[key], expected[key], path + '.' + key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path + ': length differs'
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f'{path}[{i}]')
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), path + ': value differs'
    else:
        assert actual == expected, path + ': value differs'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-bootstrap', action='store_true', help='Quick point-estimate check only')
    args = parser.parse_args()
    index, labels = load_verified()
    published = json.loads((ROOT / 'results/metrics.json').read_text())
    gates = {n: {r['id'] for r in index if r['run_name'] == n and r['phase'] == 'correction'} for n in NAMES}
    common = set.intersection(*gates.values())
    result = {'metrics': metric_rows(index, labels, gates),
              'matched_cohort_metrics': metric_rows(index, labels, {n: common for n in NAMES})}
    for key in result:
        compare(result[key], published[key], key)
    if not args.skip_bootstrap:
        for key, cohort in [('bootstrap', index), ('matched_cohort_bootstrap', [r for r in index if r['id'] in common])]:
            result[key] = bootstrap_persona(cohort, {'final': labels}, 10000)
            compare(result[key], published[key], key)
    output = ROOT / 'reproduced'
    output.mkdir(exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'verification.json').write_text(json.dumps({'status': 'verified', 'responses': len(index),
        'shared_seeds': len(common), 'metric_rows': len(result['metrics']),
        'bootstrap_checked': not args.skip_bootstrap, 'numpy': np.__version__,
        'python': platform.python_version(), 'tolerance': 'relative 1e-10, absolute 1e-12'}, indent=2) + '\n')
    print('PASS: recomputed metric rows, counts, bounds, and ' + ('point estimates only.' if args.skip_bootstrap else 'both 10,000-replicate bootstraps match published results.'))


if __name__ == '__main__':
    main()
