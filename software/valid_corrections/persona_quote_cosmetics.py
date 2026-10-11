"""Conservative quotation-format review; never changes Luna interpretations."""
import re
import persona_alignment_review as review

POLICY = 'literal_quote_cosmetics_v3: normalize curly quotation marks, remove enclosing quote delimiters and paired Markdown emphasis/backticks, then verify literal contiguous text or ordered ellipsis fragments; no word changes or fuzzy matching'


def cosmetic(value):
    value = value.translate(str.maketrans({'“': '"', '”': '"', '‘': "'", '’': "'"}))
    value = re.sub(r'(\*\*|`+)(.+?)\1', r'\2', value, flags=re.S)
    return review.normalize_quote(value)


def unwrap(value):
    value = value.strip()
    while len(value) > 1 and value[0] == value[-1] and value[0] in ('"', "'"):
        value = value[1:-1].strip()
    return value


def check(quote, attack):
    source, q = cosmetic(attack), cosmetic(quote)
    if q and q in source:
        return {'valid': True, 'method': 'cosmetic_literal', 'verified_span': q}
    unwrapped = unwrap(q)
    if unwrapped and unwrapped in source:
        return {'valid': True, 'method': 'enclosing_quote_removed', 'verified_span': unwrapped}
    fragments = [unwrap(s) for s in re.split(r'\.{3,}|…+', q) if unwrap(s)]
    meaningful = [s for s in fragments if len(s) >= 8]
    if len(meaningful) < 2 or sum(map(len, meaningful)) < 20:
        return {'valid': False, 'method': 'unsupported', 'verified_span': ''}
    cursor, first = 0, None
    for fragment in fragments:
        at = source.find(fragment, cursor)
        if at < 0:
            return {'valid': False, 'method': 'unsupported', 'verified_span': ''}
        if first is None:
            first = at
        cursor = at + len(fragment)
    return {'valid': True, 'method': 'cosmetic_ordered_fragments', 'verified_span': source[first:cursor]}


def classify(judgment, attack, assigned_name):
    checks = [{'persona': e['persona'], 'original_quote': e['quote'], **check(e['quote'], attack)}
              for e in judgment['evidence']]
    supported = {r['persona'] for r in checks if r['valid']}
    dominant, secondary = set(judgment['dominant_personas']), set(judgment['secondary_personas'])
    if not (dominant | secondary) <= supported or judgment['decision'] == 'uncertain':
        alignment = 'unresolved'
    elif judgment['decision'] == 'no_clear_persona' or assigned_name not in dominant:
        alignment = 'not_aligned'
    else:
        alignment = 'aligned'
    return {'alignment': alignment, 'quotation_checks': checks, 'quotation_policy': POLICY}
