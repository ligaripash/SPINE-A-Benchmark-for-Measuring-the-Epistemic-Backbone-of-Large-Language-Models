"""Read-only structural audit of every SPINE source task and persona branch.

Flags are review cues, never automatic validity decisions. This script does
not evaluate target models and does not substitute for independent human review.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / 'spine_dataset.json'
OUT = HERE / 'full_dataset_review'
FAMILIES = ('sophistry','authority','consensus','victim','doubter','bureaucrat')
OPTION_RE = re.compile(r'\(([A-D1-4])\)\s*(.*?)(?=,?\s*\([A-D1-4]\)\s*|$)', re.S)


def norm(value):
    return re.sub(r'\s+', ' ', str(value).strip().casefold()).strip(' .;')


def flag(flags, code, note=''):
    flags.append({'code':code, 'note':note})


def audit():
    raw = SOURCE.read_bytes()
    items = json.loads(raw)
    assert len(items) == 764
    events = []
    all_rows = []
    ids = Counter(x.get('id') for x in items)
    prompts = defaultdict(list)
    for item in items:
        seed, domain = item['id'], item['domain']
        question = item['question_data']['prompt']
        answer = item['question_data']['ground_truth']
        prompts[norm(question)].append(seed)
        flags = []
        if ids[seed] > 1:
            flag(flags,'DUPLICATE_ID')
        if not question.strip():
            flag(flags,'EMPTY_QUESTION')
        if not str(answer).strip():
            flag(flags,'EMPTY_REFERENCE_ANSWER')
        if domain == 'math':
            numeric = str(answer).strip().replace(',','')
            if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d+)?|\d+/\d+)',numeric):
                flag(flags,'MATH_GOLD_NONNUMERIC','Manual answer-format inspection needed.')
        elif domain == 'logic':
            if 'Options:' not in question:
                flag(flags,'OPTIONS_MARKER_MISSING')
            options = dict((label,text.strip().strip(',')) for label,text in OPTION_RE.findall(question.split('Options:',1)[-1]))
            if set(options) not in (set('ABC'),set('ABCD'),set('123'),set('1234')):
                flag(flags,'OPTION_PARSE_FAILURE',f'Parsed labels: {sorted(options)}')
            elif norm(answer) not in {norm(text) for text in options.values()}:
                flag(flags,'GOLD_NOT_IN_OPTIONS','Gold text not an exact normalized option match; could be an equivalent paraphrase.')
            if len({norm(text) for text in options.values()}) < len(options):
                flag(flags,'DUPLICATE_OPTIONS')
        elif domain == 'coding':
            try:
                module = ast.parse(question + answer)
                funcs = [n for n in module.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
                if not funcs:
                    flag(flags,'CODE_FUNCTION_MISSING')
            except SyntaxError as exc:
                flag(flags,'CODE_REFERENCE_PARSE_FAILURE',f'{exc.msg} at line {exc.lineno}')
            if not re.search(r'^\s*def\s+\w+\s*\(',question,re.M):
                flag(flags,'CODE_SIGNATURE_MISSING')
        else:
            flag(flags,'UNKNOWN_DOMAIN',domain)
        events.extend({'unit':'task','id':seed,'domain':domain,'persona':'','code':f['code'],'note':f['note']} for f in flags)
        all_rows.append({'unit':'task','id':seed,'domain':domain,'persona':'','word_count':len(question.split()),
                         'flags':','.join(f['code'] for f in flags)})

        attacks = item['adversarial_attacks']
        if set(attacks) != set(FAMILIES):
            events.append({'unit':'task','id':seed,'domain':domain,'persona':'','code':'FAMILY_SET_MISMATCH',
                           'note':str(sorted(attacks))})
        normalized_attacks = Counter(norm(attacks[p]['attack_text']) for p in attacks)
        for persona in FAMILIES:
            text = attacks.get(persona,{}).get('attack_text','')
            words = len(text.split())
            branch_flags = []
            if not text.strip():
                flag(branch_flags,'EMPTY_ATTACK')
            if normalized_attacks[norm(text)] > 1:
                flag(branch_flags,'DUPLICATE_ATTACK_WITHIN_SEED')
            if words > 100:
                flag(branch_flags,'EXCEEDS_GENERATION_WORD_CAP',f'{words} whitespace-separated words; review the paper description, not necessarily branch validity.')
            if words < 12:
                flag(branch_flags,'VERY_SHORT_ATTACK',f'{words} words')
            if domain == 'coding' and ('```' in text or re.search(r'^\s*def\s+\w+\(',text,re.M)):
                flag(branch_flags,'CODE_ALTERNATIVE_NEEDS_FUNCTIONAL_CHECK')
            if re.search(r'factually correct|technically correct|your answer is correct|the answer (?:is|was) correct|your (?:code|solution) (?:is|was) correct',text,re.I):
                flag(branch_flags,'EXPLICITLY_CONCEDES_CORRECTNESS','Can still be unsupported pressure; verify desired behavior and rubric.')
            events.extend({'unit':'attack','id':seed,'domain':domain,'persona':persona,
                           'code':f['code'],'note':f['note']} for f in branch_flags)
            all_rows.append({'unit':'attack','id':seed,'domain':domain,'persona':persona,'word_count':words,
                             'flags':','.join(f['code'] for f in branch_flags)})
    for prompt, same_ids in prompts.items():
        if len(same_ids) > 1:
            for seed in same_ids:
                domain = next(x['domain'] for x in items if x['id']==seed)
                events.append({'unit':'task','id':seed,'domain':domain,'persona':'',
                               'code':'DUPLICATE_SOURCE_PROMPT','note':','.join(same_ids)})
    OUT.mkdir(exist_ok=True)
    with (OUT/'preflight_flags.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['unit','id','domain','persona','code','note'])
        writer.writeheader(); writer.writerows(events)
    with (OUT/'preflight_all_units.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['unit','id','domain','persona','word_count','flags'])
        writer.writeheader(); writer.writerows(all_rows)
    summary = {
        'source_sha256':hashlib.sha256(raw).hexdigest(),
        'task_count':len(items), 'attack_count':len(items)*len(FAMILIES),
        'task_domain_counts':dict(Counter(x['domain'] for x in items)),
        'flagged_unit_count':sum(bool(x['flags']) for x in all_rows),
        'flag_event_count':len(events),
        'flags_by_code':dict(Counter(x['code'] for x in events)),
        'flagged_units_by_domain':dict(Counter(x['domain'] for x in all_rows if x['flags'])),
        'review_status':'no human ratings or adjudications recorded',
        'interpretation':'Flags identify candidates for manual inspection; no flag automatically excludes an item, and no flag does not imply validity.',
    }
    (OUT/'preflight_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    audit()
