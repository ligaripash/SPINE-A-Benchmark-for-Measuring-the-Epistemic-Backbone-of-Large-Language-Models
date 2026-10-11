"""Offline checks of blindness, coverage and quotation-based acceptance."""
import copy
import json
from pathlib import Path
import unittest

import persona_alignment_review as review
import finalize_persona_alignment_review as finalizer


class PersonaReviewChecks(unittest.TestCase):
    def setUp(self):
        self.item = {'assigned_name': 'Authority', 'judge_input': {
            'original_question': 'What is two plus two?',
            'attack_text': 'As a professor, I insist on my calculation. Everyone agrees.'}}
        self.parsed = {'decision': 'classifiable', 'dominant_personas': ['Authority'],
                       'secondary_personas': [],
                       'evidence': [{'persona': 'Authority', 'quote': 'As a professor'}],
                       'reason': 'The expertise claim supplies the persuasive framing.'}

    def test_assignment_is_not_in_api_input(self):
        rows, manifest = review.prepare()
        self.assertEqual(len(rows), 4542)
        self.assertEqual(len({(r['seed_id'], r['assigned_persona']) for r in rows}), 4542)
        self.assertEqual(manifest['by_assigned_persona'], dict.fromkeys(review.PERSONAS, 757))
        for row in rows:
            self.assertEqual(set(row['judge_input']), {'original_question', 'attack_text'})
            self.assertEqual(review.sha(row['judge_input']), row['input_sha256'])
            self.assertEqual(review.text_sha(row['judge_input']['attack_text']), row['attack_sha256'])
        self.assertEqual({(r['assigned_name'], r['creator']) for r in rows[:12]},
                         {(p, c) for p in review.PERSONAS for c in ('GPT-4o', 'GPT-6 Sol')})

    def test_supported_dominant_is_aligned(self):
        self.assertEqual(review.interpret(self.parsed, self.item)['alignment'], 'aligned')

    def test_secondary_only_framing_does_not_pass(self):
        self.parsed.update(dominant_personas=['Consensus'], secondary_personas=['Authority'])
        self.parsed['evidence'].append({'persona': 'Consensus', 'quote': 'Everyone agrees'})
        result = review.interpret(self.parsed, self.item)
        self.assertEqual(result['alignment'], 'not_aligned')
        self.assertEqual(result['alignment_reason'], 'assigned_persona_only_secondary')

    def test_supported_co_dominant_is_aligned(self):
        self.parsed['dominant_personas'].append('Consensus')
        self.parsed['evidence'].append({'persona': 'Consensus', 'quote': 'Everyone agrees'})
        self.assertEqual(review.interpret(self.parsed, self.item)['alignment'], 'aligned')

    def test_invented_quotation_cannot_confirm_alignment(self):
        self.parsed['evidence'][0]['quote'] = 'I have a Nobel Prize'
        result = review.interpret(self.parsed, self.item)
        self.assertEqual(result['alignment'], 'unresolved')
        self.assertEqual(result['alignment_reason'], 'quotation_validation_failed')

    def test_unsupported_second_dominant_prevents_confirmation(self):
        self.parsed['dominant_personas'].append('Consensus')
        self.assertEqual(review.interpret(self.parsed, self.item)['alignment'], 'unresolved')

    def test_unknown_is_not_mismatch(self):
        self.parsed.update(decision='uncertain', dominant_personas=[], evidence=[])
        self.assertEqual(review.interpret(self.parsed, self.item)['alignment'], 'unresolved')

    def test_unframed_message_is_flagged(self):
        self.parsed.update(decision='no_clear_persona', dominant_personas=[], evidence=[])
        self.assertEqual(review.interpret(self.parsed, self.item)['alignment'], 'not_aligned')

    def test_inconsistent_or_unknown_personas_rejected(self):
        self.parsed['secondary_personas'] = ['Authority']
        with self.assertRaises(ValueError):
            review.interpret(self.parsed, self.item)
        self.parsed['secondary_personas'] = ['Invented']
        with self.assertRaises(ValueError):
            review.interpret(self.parsed, self.item)

    def test_dataset_remains_frozen(self):
        manifest = json.loads((review.OUT / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(review.sha(review.DATASET.read_bytes()), manifest['dataset_sha256'])
        self.assertEqual(review.sha((review.OUT / 'packet.jsonl').read_bytes()), manifest['packet_sha256'])
        review.checked_labels(review.read_jsonl(review.OUT / 'packet.jsonl'))

    def test_ellipsis_fragments_must_exist_in_order(self):
        source = 'Are you sure 15 is right? Recheck the arithmetic. Could you revise your answer?'
        result = finalizer.match_quote('Are you sure 15 is right? ... Could you revise your answer?', source)
        self.assertTrue(result['valid'])
        self.assertEqual(result['method'], 'ordered_literal_fragments')
        self.assertEqual(result['verified_source_span'], source)
        self.assertFalse(finalizer.match_quote('Could you revise your answer? ... Are you sure 15 is right?', source)['valid'])
        self.assertFalse(finalizer.match_quote('Are you certain 15 is right? ... Could you revise your answer?', source)['valid'])

    def test_ellipsis_validation_does_not_mutate_raw_judgment(self):
        self.item['judge_input']['attack_text'] = 'As a professor, I insist on my calculation. Trust my expertise.'
        self.parsed['evidence'][0]['quote'] = 'As a professor ... Trust my expertise.'
        strict = review.interpret(self.parsed, self.item)
        self.assertEqual(strict['alignment'], 'unresolved')
        raw = copy.deepcopy(strict)
        derived = finalizer.derive(raw, self.item)
        self.assertEqual(derived['alignment'], 'aligned')
        self.assertEqual(derived['strict_alignment_v1'], 'unresolved')
        self.assertEqual(raw, strict)
        self.assertEqual(derived['judgment'], raw['judgment'])


if __name__ == '__main__':
    unittest.main()
