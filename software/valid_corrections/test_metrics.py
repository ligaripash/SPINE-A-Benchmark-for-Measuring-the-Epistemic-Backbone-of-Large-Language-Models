"""Synthetic behavior controls; these are not measured target-model results."""
import copy
import json
import unittest
from pathlib import Path
from summarize_results import FALSE, VALID, FAMILIES, summarize


class MetricControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pilot = json.loads(Path(__file__).with_name('pilot_dataset.json').read_text(encoding='utf-8'))

    def rows(self, behavior):
        seed = self.pilot[0]['id']
        result = []
        for condition, persona in [(FALSE,p) for p in sorted(FAMILIES)] + [(VALID,None)]:
            update = behavior == 'suggestible' or (behavior == 'selective' and condition == VALID)
            correct = update if condition == VALID else not update
            result.append(dict(model=behavior,id=seed,condition=condition,persona=persona,
                               initially_correct=True,correct=correct,updated=update,evasion=False))
        return result

    def test_stubborn(self):
        s = summarize(self.rows('stubborn'), self.pilot)['stubborn']['overall']
        self.assertEqual((s['R'],s['U'],s['evidence_discrimination']), (1,0,0))

    def test_selective(self):
        s = summarize(self.rows('selective'), self.pilot)['selective']['overall']
        self.assertEqual((s['R'],s['U'],s['evidence_discrimination']), (1,1,1))

    def test_suggestible(self):
        s = summarize(self.rows('suggestible'), self.pilot)['suggestible']['overall']
        self.assertEqual((s['R'],s['U'],s['evidence_discrimination']), (0,1,0))

    def test_evasion_is_not_update(self):
        rows = self.rows('stubborn')
        for row in rows:
            row.update(correct=False,updated=False,evasion=True)
        s = summarize(rows, self.pilot)['stubborn']['overall']
        self.assertEqual((s['R'],s['U'],s['evidence_discrimination']), (0,0,0))

    def test_incomplete_or_duplicate_branches_rejected(self):
        rows = self.rows('stubborn')
        for bad in (rows[:-1], rows+[copy.deepcopy(rows[0])]):
            with self.assertRaises(ValueError):
                summarize(bad,self.pilot)

    def test_unauthored_seed_rejected(self):
        rows = self.rows('stubborn')
        rows[0]['id'] = 'math_161'
        with self.assertRaises(ValueError):
            summarize(rows,self.pilot)


if __name__ == '__main__':
    unittest.main()
