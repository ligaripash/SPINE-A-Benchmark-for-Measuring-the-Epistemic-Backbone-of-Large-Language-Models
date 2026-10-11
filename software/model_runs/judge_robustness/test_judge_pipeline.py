"""Local tests of abstention, blinding, evidence, and frozen-input invariants."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from analyze_judges import agreement, correctness, vote
from run_judges import call

HERE=Path(__file__).resolve().parent


class JudgePipelineChecks(unittest.TestCase):
    def test_abstention_cannot_cast_deciding_vote(self):
        self.assertIsNone(vote([True,None]))
        self.assertIsNone(vote([True,False]))
        self.assertIsNone(vote([False,None,None]))
        self.assertIs(vote([True,True,None]),True)
        self.assertIs(vote([False,False,True]),False)

    def test_nonanswers_are_not_unresolved(self):
        self.assertIs(correctness('NONE','B'),False)
        self.assertIs(correctness('MULTIPLE','B'),False)
        self.assertIsNone(correctness('UNCLEAR','B'))

    def test_wrong_choices_can_agree_on_correctness_but_not_extraction(self):
        self.assertEqual(agreement({'x':'A'},{'x':'C'})['agreement'],0)
        self.assertEqual(agreement({'x':False},{'x':False})['agreement'],1)

    def test_resolved_extraction_requires_quote_from_response(self):
        item={'blind_id':'synthetic','input_sha256':'synthetic', 'option_labels':['A','B'],
              'judge_input':{'task':'Options: (A) one, (B) two', 'final_response':'I choose B.'}}
        def response(evidence):
            return {'model':'synthetic-test-only','usage':{},'choices':[{'finish_reason':'stop',
                'message':{'content':json.dumps({'choice':'A','evidence':evidence,'reason':'synthetic test'})}}]}
        with patch('run_judges.post',return_value=response('I choose A.')):
            self.assertEqual(call('luna','synthetic','synthetic',item)['choice'],'UNCLEAR')
        with patch('run_judges.post',return_value=response('I choose B.')) as mocked:
            out=call('luna','synthetic','synthetic',item)
            self.assertTrue(out['evidence_valid'])
            sent=mocked.call_args.args[1]['messages'][0]['content']
            self.assertNotIn('ground_truth',sent)
            self.assertNotIn('primary_correct',sent)

    def test_human_sample_and_packet_metadata_are_blinded(self):
        human=json.loads((HERE/'human_sample_blinded.json').read_text(encoding='utf-8'))
        self.assertEqual(len(human),120)
        self.assertTrue(all(r['domain'] in ['math','logic'] for r in human))
        self.assertTrue(all('run_name' not in r and 'primary_correct' not in r for r in human))
        packet=[json.loads(s) for s in (HERE/'judge_packet.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(packet),12119)
        for r in packet:
            self.assertEqual(set(r),{'blind_id','judge_input','option_labels','input_sha256'})
            self.assertNotIn('reference_answer',r['judge_input'])
            self.assertNotIn('ground_truth',r['judge_input'])

    def test_nonexistent_option_is_unresolved_without_guessing_a_mapping(self):
        item={'blind_id':'synthetic','input_sha256':'synthetic','option_labels':['A','B','C','D'],
              'judge_input':{'task':'Options A through D','final_response':'I choose the second option.'}}
        response={'model':'synthetic-test-only','usage':{},'choices':[{'finish_reason':'stop',
            'message':{'content':json.dumps({'choice':'2','evidence':'second option','reason':'synthetic test'})}}]}
        with patch('run_judges.post',return_value=response):
            out=call('luna','synthetic','synthetic',item)
        self.assertEqual(out['choice'],'UNCLEAR')
        self.assertEqual(out['raw_choice'],'2')
        self.assertFalse(out['choice_valid_for_task'])


if __name__=='__main__':unittest.main()
