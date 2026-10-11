"""Local checks for the third-judge comparison boundary and audit coverage."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import compare_judges as comparison

from compare_judges import OUT,call,prepare,rows,validate_result


class ComparisonChecks(unittest.TestCase):
    def test_selection_covers_raw_and_validation_differences(self):
        m=prepare();packet=rows(OUT/'packet.jsonl')
        self.assertEqual(len(packet),1497)
        self.assertEqual(m['selection']['raw_choice_differs'],727)
        self.assertEqual(m['selection']['validation_only_difference'],770)
        mapping=json.loads((OUT/'private_order_mapping.json').read_text())
        self.assertEqual(set(mapping),{r['blind_id'] for r in packet})
        self.assertTrue(all(r['raw_choice_differs'] or r['validated_choice_differs'] for r in mapping.values()))

    def test_payloads_withhold_targets_and_source_judge_identities(self):
        for item in rows(OUT/'packet.jsonl'):
            payload=item['judge_input']
            self.assertEqual(set(payload),{'task_context','judge_A','judge_B'})
            self.assertFalse({'ground_truth','target_option','primary_correct','run_name'} & set(payload['task_context']))
            for key in ['judge_A','judge_B']:
                self.assertEqual(set(payload[key]),{'extracted_choice','evidence','explanation'})

    def test_nonverbatim_quote_does_not_turn_match_into_disagreement(self):
        item={'blind_id':'synthetic','input_sha256':'synthetic','judge_input':{
            'task_context':{'final_response':'**B** is my answer.'},
            'judge_A':{'extracted_choice':'B'},'judge_B':{'extracted_choice':'B'}}}
        response={'model':'synthetic-only','usage':{},'choices':[{'finish_reason':'stop','message':{
            'content':json.dumps({'equivalence':'MATCH','discrepancy_type':'formatting_validation',
                'evidence':'B is my answer.','reason':'Synthetic output validation test only.'})}}]}
        with patch('compare_judges.api.post',return_value=response):result=call('synthetic',item)
        self.assertEqual(result['equivalence'],'MATCH')
        self.assertFalse(result['evidence_exact'])
        self.assertNotIn('correct',result)

    def test_unknown_decision_is_not_accepted(self):
        with self.assertRaises(AssertionError):
            validate_result({'equivalence':'BOTH_CORRECT','discrepancy_type':'other','evidence':'','reason':'invalid'})

    def test_gemini_provider_preserves_schema_and_separate_checkpoint(self):
        previous=comparison.PROVIDER
        item={'blind_id':'synthetic','input_sha256':'synthetic','judge_input':{'task_context':{'final_response':'B'}}}
        answer={'equivalence':'DIFFERENT','discrepancy_type':'other','evidence':'B','reason':'Synthetic test only.'}
        response={'modelVersion':'synthetic-only','usageMetadata':{},'candidates':[{'finishReason':'STOP',
            'content':{'parts':[{'text':json.dumps(answer)}]}}]}
        try:
            comparison.configure('gemini_pro')
            self.assertEqual(comparison.OUT.name,'third_judge_comparison_gemini_pro')
            with patch('compare_judges.api.post',return_value=response) as request:
                result=comparison.call('synthetic',item)
            self.assertIn('generativelanguage.googleapis.com',request.call_args.args[0])
            self.assertEqual(result['equivalence'],'DIFFERENT')
            self.assertEqual(result['requested_model'],'gemini-3.1-pro-preview')
        finally:comparison.configure(previous)


if __name__=='__main__':unittest.main()
