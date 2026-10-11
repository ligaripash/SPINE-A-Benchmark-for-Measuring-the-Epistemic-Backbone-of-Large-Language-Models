import unittest
import run_judges as api
import run_sol_adjudication as a
import analyze_judges as baseline
import analyze_sol_adjudication as final


class AdjudicationChecks(unittest.TestCase):
    def setUp(self):
        self.item = {'option_labels': ['A', 'B'],
                     'judge_input': {'final_response': '**B** is my final answer.'}}

    def answer(self, **changes):
        value = {'choice': 'B', 'answer_text': 'second option',
                 'evidence': 'B is my final answer.', 'reason': 'Explicit final answer.'}
        value.update(changes)
        return value

    def test_cosmetic_formatting_does_not_lose_a_supported_answer(self):
        exact, normalized = a.validate(self.answer(), self.item)
        self.assertFalse(exact); self.assertTrue(normalized)

    def test_invented_support_is_rejected(self):
        with self.assertRaises(api.JudgeValidationError):
            a.validate(self.answer(evidence='The answer is A.'), self.item)

    def test_nonexistent_option_is_rejected(self):
        with self.assertRaises(api.JudgeValidationError):
            a.validate(self.answer(choice='C'), self.item)

    def test_ambiguity_can_remain_unresolved(self):
        a.validate(self.answer(choice='UNCLEAR', answer_text='', evidence=''), self.item)

    def test_clear_unlisted_answer_has_an_incorrect_category(self):
        item = {'option_labels': ['A', 'B', 'OTHER'],
                'judge_input': {'final_response': 'The answer is (E) cryothermic.'}}
        a.validate(self.answer(choice='OTHER', answer_text='cryothermic',
                   evidence='(E) cryothermic.'), item)
        self.assertFalse(baseline.correctness('OTHER', 'B'))

    def test_packet_is_blinded_and_exactly_covers_disputes(self):
        m = a.prepare()
        self.assertEqual(m['selected'], 1497)
        for item in a.runner.rows(a.OUT / 'packet.jsonl'):
            self.assertEqual(set(item['judge_input']), {'task', 'branch_kind', 'original_task',
                'initial_answer', 'followup_message', 'final_response'})
            self.assertNotIn('target_option', item)
            self.assertNotIn('judge_A', item['judge_input'])

    def test_agreed_unclear_extension_is_disjoint_and_uses_same_blinded_policy(self):
        import run_sol_agreed_unclear as extension
        import run_sol_clarification as clarification
        self.assertEqual(extension.prepare()['selected'], 92)
        disputed = {r['blind_id'] for r in a.runner.rows(a.OUT / 'packet.jsonl')}
        packet = a.runner.rows(extension.OUT / 'packet.jsonl')
        self.assertFalse(disputed & {r['blind_id'] for r in packet})
        self.assertEqual(extension.RUBRIC, clarification.RUBRIC)
        self.assertEqual(extension.SCHEMA, clarification.SCHEMA)
        for item in packet:
            self.assertEqual(set(item['judge_input']), {'task', 'branch_kind', 'original_task',
                'initial_answer', 'followup_message', 'final_response'})
            self.assertNotIn('target_option', item)

    def test_final_metric_denominators_match_frozen_scores_and_drop_demoted_anchor(self):
        index = baseline.rows(a.HERE / 'private_response_index.jsonl')
        labels = {r['blind_id']: r['primary_correct'] for r in index}
        gates = {name: {r['id'] for r in index if r['run_name'] == name and r['phase'] == 'correction'}
                 for name in baseline.NAMES}
        actual = final.metric_rows(index, labels, gates)
        expected = baseline.metrics(index, labels)
        for r in expected:
            found = next(x for x in actual if x['domain'] == 'all'
                         and x['run_name'] == r['run_name'] and x['attack_type'] == r['attack_type'])
            for key in ['accuracy', 'ESS', 'AUR', 'BES', 'Overall']:
                self.assertAlmostEqual(found[key], r[key])
        name = baseline.NAMES[0]; ident = next(iter(gates[name]))
        base = next(r for r in index if r['run_name'] == name and r['phase'] == 'base' and r['id'] == ident)
        labels[base['blind_id']] = False; gates[name].remove(ident)
        dropped = next(r for r in final.metric_rows(index, labels, gates)
                       if r['run_name'] == name and r['domain'] == r['attack_type'] == 'all')
        before = next(r for r in actual if r['run_name'] == name and r['domain'] == r['attack_type'] == 'all')
        self.assertEqual(dropped['attack_counts']['total'], before['attack_counts']['total'] - 6)
        self.assertEqual(dropped['correction_counts']['total'], before['correction_counts']['total'] - 1)
        self.assertAlmostEqual(dropped['accuracy'], before['accuracy'] - 1/757)


if __name__ == '__main__':
    unittest.main()
