"""Recompute explicit numeric equations without evaluating arbitrary code.

This validates arithmetic in recorded reasoning, not the semantic translation
of every word problem. Equations with variables are left unassessed.
"""
import ast
import json
import operator
import re
from fractions import Fraction
from pathlib import Path

HERE=Path(__file__).resolve().parent
OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.Pow:operator.pow}

def calculate(text):
    node=ast.parse(text.strip(),mode='eval').body
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):
            return Fraction(str(n.value))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
            return visit(n.operand) * (-1 if isinstance(n.op,ast.USub) else 1)
        if isinstance(n,ast.BinOp) and type(n.op) in OPS:
            a,b=visit(n.left),visit(n.right)
            if isinstance(n.op,ast.Pow) and (b.denominator!=1 or abs(b)>12):
                raise ValueError('unsupported_exponent')
            return OPS[type(n.op)](a,b)
        raise ValueError('not_numeric_arithmetic')
    return visit(node)

def equations(text):
    text=text.replace('×','*').replace('÷','/').replace('−','-').replace('$','')
    text=text.replace('^','**')
    text=re.sub(r'(?<=\d),(?=\d{3}\b)','',text)
    found=[]
    for m in re.finditer(r'[\d\s.()+*/%=\-]+',text):
        chain=m.group().strip().rstrip('.')
        if '=' not in chain or not re.match(r'[\d(]',chain):
            continue
        # A bare fraction after division is ambiguous under programming
        # precedence, e.g. human prose '10 / 1/2'. Leave it unassessed.
        if re.search(r'/\s*\d+\s*/\s*\d+',chain):
            continue
        parts=[p.strip() for p in chain.split('=')]
        # A bare number on the left may be a tail cut off after a word such
        # as 'of' in '70% of 240 = 168'; it is not a complete expression.
        if not re.search(r'[+*/\-]',parts[0]):
            continue
        try:
            values=[]
            for p in parts:
                if p.endswith('%') and re.search(r'\*\s*100\s*$',parts[0]):
                    p=p[:-1] # conventional percentage output after times 100
                values.append(calculate(re.sub(r'(\d+(?:\.\d+)?)%',r'(\1/100)',p)))
            actual=values[0]
            for right,expected in zip(parts[1:],values[1:]):
                decimals=len(right.split('.')[1]) if re.fullmatch(r'\d+\.\d+',right) else 0
                exact=actual==expected
                rounded=decimals>0 and abs(actual-expected)<=Fraction(1,2*10**decimals)
                found.append({'expression':parts[0],'reported':right,'computed_exact':str(actual),
                              'status':'exact' if exact else 'rounded' if rounded else 'mismatch'})
        except (SyntaxError,ValueError,ZeroDivisionError,OverflowError):
            continue
    return found

def main():
    rows=[json.loads(s) for s in (HERE/'target_audit.jsonl').read_text(encoding='utf-8').splitlines()]
    results=[]
    for row in rows:
        if row['domain']!='math' or row['status']=='missing':
            continue
        parsed=row['ai_review']['parsed']
        checks=equations(parsed.get('reason','')+'\n'+parsed.get('concrete_check',''))
        results.append({'id':row['id'],'followup_sha256':row['followup_sha256'],'equations':checks,
                        'status':'arithmetic_disagreement' if any(x['status']=='mismatch' for x in checks)
                        else 'numeric_equations_recomputed' if checks else 'no_supported_numeric_equation'})
    (HERE/'arithmetic_evidence.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    from collections import Counter
    print(json.dumps({'items':len(results),'counts':dict(Counter(x['status'] for x in results)),
                     'numeric_equations':sum(len(x['equations']) for x in results),
                     'disagreements':[x for x in results if x['status']=='arithmetic_disagreement']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
