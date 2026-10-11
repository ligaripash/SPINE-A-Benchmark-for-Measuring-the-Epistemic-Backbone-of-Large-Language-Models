"""Check analysis math and synthetic-export rejection without making study data."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import analyze_science_human_review_v3 as audit

manifest = audit.load(audit.AUDIT / 'manifest.json')
private = audit.load(audit.AUDIT / 'analyst_private.json')
ids = manifest['primary_ids']
weights = {i: private[i]['analysis_weight'] for i in ids}
assert len(ids) == 120 and len(set(ids)) == 120
assert set(manifest['secondary_ids']) <= set(ids)
assert abs(sum(weights.values()) - manifest['population_n']) < 1e-6
same = {i: True for i in ids}
different = {i: False for i in ids}
assert audit.paired_stats(ids, same, same, weights)['agreement'] == 1
assert audit.paired_stats(ids, same, different, weights)['agreement'] == 0
assert audit.paired_stats(ids, same, different, weights)['kappa'] == 0
interval = audit.design_interval(ids, same, private, manifest)
assert abs(interval['estimate'] - 1) < 1e-12 and interval['standard_error'] == 0
with TemporaryDirectory() as temp:
    path = Path(temp) / 'synthetic_rejected.json'
    path.write_text(json.dumps({'schema': manifest['schema'], 'packet_id': manifest['packet_id'],
                                'role': 'primary', 'reviewer': 'synthetic_UI_test',
                                'sample_ids': ids, 'decisions': {}}), encoding='utf-8')
    try:
        audit.validate(path, manifest, 'primary')
        raise AssertionError('Synthetic export accepted as real evidence')
    except ValueError as error:
        assert 'synthetic' in str(error)
result = {'status': 'passed', 'sampling_weights_sum': sum(weights.values()),
          'checks': ['unique sample membership and overlap', 'weights reproduce population size',
                     'agreement and kappa arithmetic', 'stratified interval arithmetic',
                     'synthetic export rejected as study evidence'], 'human_judgments_created': 0}
(audit.AUDIT / 'analysis_verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result))
