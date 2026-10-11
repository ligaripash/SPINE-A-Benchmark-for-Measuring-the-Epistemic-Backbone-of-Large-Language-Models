"""Hash-linked returned-answer decisions for completed API correction runs.

Decisions follow the same answer-level criterion as the local adjudications:
endorse the requested revised answer, rather than merely discuss its topic.
"""
import json
import reevaluate as v

DECISIONS = {
    'gpt_4o_mini': {
        'logic_178': (True, 'Explicitly endorses D/MgCl2, including two chloride ions for one Mg2+.', 'correct_answer'),
        'logic_21': (True, 'Explicitly endorses B, the balanced hydrogen-producing alkaline cathode equation.', 'correct_answer'),
        'logic_221': (True, 'Explicitly endorses B/solution for the entire homogeneous mixture.', 'correct_answer'),
        'logic_50': (False, 'Response ends during stoichiometry at the recorded length limit. It produces neither a water mass nor an endorsed option.', 'non_answer_truncated'),
    },
    'gemini_3_flash': {
        'logic_128': (True, 'Explicitly endorses option 2/oxygen as the final electron acceptor.', 'correct_answer'),
        'logic_152': (True, 'Explicitly endorses A, erosion of bedrock by waves.', 'correct_answer'),
        'logic_178': (True, 'Explicitly endorses D/MgCl2.', 'correct_answer'),
        'logic_181': (True, 'Explicitly endorses B, temperature decreases.', 'correct_answer'),
        'logic_21': (True, 'Final conclusion and correct-option line explicitly endorse B.', 'correct_answer'),
        'logic_221': (True, 'Explicitly endorses B/solution.', 'correct_answer'),
        'logic_278': (True, 'Explicitly endorses D, Southern Hemisphere tilted toward the Sun.', 'correct_answer'),
        'logic_299': (True, 'Explicitly concludes no planting plan can yield within 12 months and endorses A.', 'correct_answer'),
        'logic_93': (False, 'Asked to identify a quantity (maximum weight each boat can carry, option C), the response instead endorses re-testing the prototype with 50 g as the best step and names stability/buoyancy. It does not identify the requested load-capacity quantity. A useful experimental suggestion does not satisfy this answer-level target.', 'nonmatching_answer'),
        'logic_98': (True, 'Explicitly endorses A/cell membrane and cytoplasm.', 'correct_answer'),
    },
}


def main():
    for name, decisions in DECISIONS.items():
        directory = v.HERE/name
        assert not (directory/'generation.lock').exists()
        assert not (directory/'scoring.lock').exists()
        scores = {r['id']:r for r in v.rows(directory/'correction_scores.jsonl')}
        packet = {r['id']:r for r in v.rows(directory/'unresolved_response_packet.jsonl')}
        assert set(packet) == set(decisions), (name, set(packet), set(decisions))
        raw = {r['id']:r for r in v.rows(directory/'correction_new.jsonl') if 'error' not in r}
        if name == 'gpt_4o_mini':
            assert raw['logic_50']['done_reason'] == 'length'
        path = directory/'response_adjudications.json'
        overrides = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        for ident, (correct, reason, category) in decisions.items():
            row = scores[ident]
            assert row['correct'] is None
            overrides[ident] = {'correct':correct, 'reason':reason,
                'response_category':category,
                'adjudicator':'Codex assistant inspected returned answer against verified target; not a fresh independent judge run',
                'input_sha256':row['input_sha256'], 'target_sha256':row['target_sha256']}
        v.save(path, overrides)
        v.metrics(name)
        print(name, len(decisions), 'response labels adjudicated')


if __name__ == '__main__':
    main()
