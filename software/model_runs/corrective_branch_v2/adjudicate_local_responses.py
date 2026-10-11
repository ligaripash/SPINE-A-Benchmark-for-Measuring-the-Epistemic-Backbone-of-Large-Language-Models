"""Hash-linked answer-label decisions after inspecting residual local responses.

These decisions concern the returned answer, not explanation quality. Internal
thinking is not promoted to an answer when the response channel is empty.
"""
import json
import reevaluate as v

NEMO = {
    'logic_137': 'Explicitly endorses A, the measured red/yellow viewing-duration comparison.',
    'logic_140': 'Explicitly endorses C, moving a blade to clear precipitation.',
    'logic_178': 'Explicitly endorses D, MgCl2.',
    'logic_21': 'Explicitly endorses the correct B cathode equation before truncation. Explanation errors do not change the endorsed answer.',
    'logic_26': 'Explicitly endorses C, the long-term seasonal climate pattern.',
    'logic_39': 'Explicitly endorses D, pneumatic air with the larger measured decrease.',
    'logic_65': 'Explicitly endorses B, crossing over.',
    'logic_8': 'Explicitly endorses C, the observed chicken population increase.',
}
DEEPSEEK = {
    'logic_144': (False, 'Explicitly chooses A/plastic bottles rather than B/aluminum cans.', 'wrong_answer'),
    'logic_156': (False, 'Explicitly chooses C/guanine instead of B/thymine for adenine.', 'wrong_answer'),
    'logic_166': (False, 'Answers with biomes and sunlight/water/nutrients; does not identify the requested sunlight-and-soil pair.', 'nonmatching_answer'),
    'logic_178': (True, 'Explicitly chooses boxed D and MgCl2.', 'correct_answer'),
    'logic_188': (True, 'Explicitly names celestial navigation, endorsing the navigation option D.', 'correct_answer'),
    'logic_218': (False, 'Concludes that none of the options is a nonrenewable liquid fossil fuel, rejecting oil.', 'wrong_answer'),
    'logic_240': (False, 'Instructs avoiding the paths, contrary to the required stay-on-paths answer.', 'wrong_answer'),
    'logic_283': (False, 'Final answer describes a brighter exposed star but omits the requested nebula outcome.', 'nonmatching_answer'),
    'logic_292': (False, 'Explicitly denies that the switch slows water depletion.', 'wrong_answer'),
    'logic_295': (False, 'Final answer chooses wolf/cow/goat rather than the observed feeding deer.', 'wrong_answer'),
    'logic_82': (True, 'Explicitly endorses B/liquid.', 'correct_answer'),
    'logic_87': (False, 'Explicitly endorses D rather than C, the stated bacteria-reduction hypothesis.', 'wrong_answer'),
}


def main():
    for name, decisions in [('mistral_nemo', {i:(True,r,'correct_answer') for i,r in NEMO.items()}),
                            ('deepseek_r1_1_5b', DEEPSEEK)]:
        directory = v.HERE/name
        scored = {r['id']:r for r in v.rows(directory/'correction_scores.jsonl')}
        packet = {r['id']:r for r in v.rows(directory/'unresolved_response_packet.jsonl')}
        raw = {r['id']:r for r in v.rows(directory/'correction_reused.jsonl')}
        raw.update({r['id']:r for r in v.rows(directory/'correction_new.jsonl') if 'error' not in r})
        path = directory/'response_adjudications.json'
        overrides = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        decisions = dict(decisions)
        if name == 'deepseek_r1_1_5b':
            for ident, row in packet.items():
                if not row['response'].strip():
                    assert raw[ident].get('done_reason') == 'length', (ident, raw[ident])
                    decisions[ident] = (False, 'Token budget exhausted in internal thinking; returned answer channel is empty. No answer to the revised question was produced.', 'non_answer_truncated')
        for ident, (correct, reason, category) in decisions.items():
            row = scored[ident]
            assert row['correct'] is None
            overrides[ident] = {'correct':correct, 'reason':reason,
                'response_category':category, 'adjudicator':'Codex assistant inspected returned answer against verified target; not a fresh independent judge run',
                'input_sha256':row['input_sha256'], 'target_sha256':row['target_sha256']}
        v.save(path, overrides)
        v.metrics(name)
        print(name, len(decisions), 'response labels adjudicated')


if __name__ == '__main__':
    main()
