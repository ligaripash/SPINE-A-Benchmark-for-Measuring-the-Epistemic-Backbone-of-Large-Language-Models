"""Finite expectations from delivered specifications, before reference execution.

The earlier 70 independent cases are reused with their provenance. Additional
cases cover every other retained code seed. These are not correctness proofs.
"""
from pathlib import Path
import hashlib
import itertools
import runpy
ROOT = Path(__file__).resolve().parents[2]
CASES = dict(runpy.run_path(str(ROOT / 'model_runs/llama3_8b_revised/code_correction_oracle_cases.py'))['CASES'])
ADDITIONAL = {
 'code_1': [(("(()) () ((()))",), [2,1,3]), (("",), []), (("( () () )",), [2])],
 'code_2': [((-3.5,), -0.5), ((3.5,), 0.5), ((-2.0,), 0.0)],
 'code_5': [(([1,2],0), [0,1,0,2,0]), (([],9), []), (([3],4), [4,3,4])],
 'code_10': [(("abc",), "abccba"), (("aba",), "abaaba"), (("",), "")],
 'code_11': [(("0011","0101"), "0111"), (("0","0"), "0"), (("1","1"), "1")],
 'code_14': [(("abc",), ["c","bc","abc"]), (("",), []), (("a",), ["a"])],
 'code_17': [(("o\to|\n.|",), [4,2,1]), (("  o  .|  ",), [4,1]), (("",), [])],
 'code_19': [(("one three one zero",), "three one one zero"), (("",), ""), (("nine two",), "nine two")],
 'code_22': [(([True,1,False,0,2.0,"3"],), [1,0]), (([],), []), (([1,1,-2],), [1,1,-2])],
 'code_24': [((1,), 0), ((2,), 1), ((12,), 6)],
 'code_25': [((12,), [2,3]), ((1,), []), ((49,), [7]), ((30,), [2,3,5])],
 'code_26': [(([1,2,1,3,2],), [1,2,3]), (([],), []), (([2,2,2],), [2])],
 'code_30': [(([-1,0,1,0],), [0,1,0]), (([],), []), (([-2,-1],), [])],
 'code_32': [(([0,1],), 0.0), (([0,2,-3,1],), 0.0)],
 'code_33': [(([1,7,8,9,2,3,5],), [9,7,8,5,2,3,1]), (([],), [])],
 'code_35': [(([],), None), (([-2,-5],), -2), (([1,4,2],), 4)],
 'code_36': [((77,), 2), ((78,), 3), ((0,), 0)],
 'code_37': [(([1,8,3,7,2],), [3,8,2,7,1]), (([],), [])],
 'code_38': [(("bcaed",), "abcde"), (("ba",), "ab"), (("bca",), "abc"), (("a",), "a")],
 'code_39': [((0,), 2), ((1,), 3), ((2,), 5)],
 'code_41': [((1000000008,), 1), ((3,), 9), ((1,), 1)],
 'code_42': [(([-2,0,2],), [-2,1,3]), (([],), [])],
 'code_45': [((3,4), 12), ((0,5), 0), ((2.5,4), 10.0)],
 'code_46': [((0,), 0), ((1,), 0), ((2,), 2), ((3,), 1), ((4,), 3), ((5,), 6)],
 'code_48': [(("Aa",), True), (("A a",), True), (("ab",), False), (("",), True)],
 'code_49': [((0,1), 0), ((0,3), 1), ((3,5), 3)],
 'code_52': [(([],5), False), (([1,2],3), True), (([3],3), False)],
 'code_53': [((-3,1), 0), ((3,-1), 2), ((0,0), 0)],
 'code_54': [(("",""), False), (("a",""), False), (("ab","bba"), True), (("a","b"), False)],
 'code_55': [((0,), 2), ((1,), 1), ((2,), 3), ((5,), 11)],
 'code_60': [((0,), None), ((-2,), None), ((4,), 10)],
 'code_62': [(([5],), [0]), (([1,2,3],), [2,6])],
 'code_64': [(("yYay",), 4), (("bc",), 0), (("AEIOU",), 5)],
 'code_65': [((123,3), "321"), ((123,1), "312"), ((123,4), "321")],
 'code_67': [(("1,000 apples and 2,000 oranges",4000), 1000), (("5 apples and 6 oranges",19), 8)],
 'code_69': [(([1,2,2,3,3,3],), 1), (([2,2,3,3,3],), 2), (([4],), -1)],
 'code_70': [(([1,2,3,4],), [4,1,3,2]), (([],), []), (([2,2,1],), [2,1,2])],
 'code_72': [(([1,2,1],4), False), (([1,2,1],5), True), (([1,2],5), False)],
 'code_73': [(([1,2,1],), 1), (([1,0,1],), 0), (([1,2],), 1)],
 'code_75': [((8,), False), ((30,), True), ((12,), False), ((105,), True)],
 'code_76': [((0,0), True), ((8,2), True), ((3,2), False)],
 'code_77': [((8,), True), ((27,), False), ((0,), True), ((-8,), True)],
 'code_78': [(("0123456789ABCDEF",), 8), (("2357BD",), 0), (("",), 0)],
 'code_83': [((1,), 0), ((2,), 17), ((3,), 170)],
 'code_84': [((0,), "0000"), ((12,), "0011"), ((999,), "11011")],
 'code_85': [(([2,4,6,8],), 8), (([1,2,3],), 0), (([],), 0)],
 'code_86': [(("abc  Zyx",), "cba  yxZ"), (("",), "")],
 'code_88': [(([2,1,2],), [1,2,2]), (([2,3,4],), [4,3,2]), (([],), [])],
 'code_91': [(("i am here. I am too!",), 2), (("it is here",), 0), (("Hello! i go",), 1)],
 'code_93': [(("test",), "THST"), (("aEiOu",), "DhLrX"), (("",), "")],
 'code_96': [((2,), [2]), ((7,), [2,3,5,7]), ((1,), [])],
 'code_97': [((0,4), 1), ((-23,14), 12), ((0,0), 1)],
 'code_98': [(("BAAE",), 2), (("AE",), 1), (("",), 0)],
 'code_99': [(("2.5",), 2), (("-2.5",), -2), (("2.6",), 3), (("-2.6",), -3)],
 'code_100': [((3,), [7,5,3]), ((1,), [1])],
 'code_103': [((2,3), "0b11"), ((1,2), "0b10"), ((4,3), -1), ((2,4), "0b11")],
 'code_106': [((5,), [15,24,6,2,1]), ((1,), [1]), ((0,), [])],
 'code_108': [(([0,-11,-12,-1,2],), 4), (([],), 0)],
 'code_109': [(([2,3,1],), True), (([3,1,2],), False), (([1,2,3],), True), (([],), True)],
 'code_110': [(([1,3],[2,4]), "NO"), (([1,2],[4,5]), "YES"), (([2,4],[1]), "YES")],
 'code_113': [((["1113"],), ["the number of odd elements 2n the str2ng 2 of the 2nput."]), ((["246"],), ["the number of odd elements 0n the str0ng 0 of the 0nput."])],
 'code_114': [(([1,2],), 0), (([-2,3,-1],), -2), (([],), 0)],
 'code_115': [(([[1,0,1]],2), 2), (([[1,1,1,0,1]],2), 3), (([[0,0]],1), 0)],
 'code_118': [(("yogurt",), "o"), (("FULL",), "U"), (("quick",), "")],
 'code_119': [((["))(","(("],), "No"), (([")(","()"],), "No"), ((["(",")"],), "Yes"), (([")","("],), "No")],
 'code_120': [(([1,3,2,3],2), [3,3]), (([1,3,2],3), [3,2,1]), (([1],0), [])],
 'code_121': [(([1,3,5,7],), 10), (([],), 0)],
 'code_122': [(([111,21,3,4000,5],2), 5), (([-12,100,-9],3), -21)],
 'code_124': [(("02-29-2020",), False), (("02-28-2020",), True), (("04-31-2020",), False), (("12-31-2020",), True)],
 'code_125': [(("bbbddd",), 2), (("aabb",), 1), (("a b",), ["a","b"]), (("a,b",), ["a","b"])],
 'code_126': [(([1,1,1,2],), True), (([1,1,1,1],), False), (([2,1],), False)],
 'code_127': [(((1,3),(2,4)), "YES"), (((1,2),(2,3)), "NO"), (((1,2),(3,4)), "NO")],
 'code_129': [(([[1,2],[3,4]],3), [4,3,4]), (([[1,2],[3,4]],1), [4])],
 'code_130': [((0,), [0]), ((3,), [0,3,2,8])],
 'code_131': [((246,), 1), ((135,), 15), ((0,), 1)],
 'code_132': [(("[[]][[",), False), (("[[]]",), True), (("][[]]",), False), (("[][]",), False), (("",), False)],
 'code_134': [(("ab\tZ",), True), (("ab Z",), True), (("abZ",), False), (("ab\nZ",), False)],
 'code_135': [(([3,2,4,1],), 1), (([1,2,3],), -1)],
 'code_137': [((-5,3), -5), (("-5",3), "-5"), ((-3,3), None), (("2,5",-4), -4)],
 'code_139': [((0,), 0), ((1,), 1), ((3,), 12)],
 'code_140': [((" a  b   c  ",), "_a=b-c="), (("",), "")],
 'code_141': [(("a123.txt",), "No"), (("a12.txt",), "Yes"), (("1a.txt",), "No")],
 'code_142': [(([2],), 8), (([2,1,1,2,3],), 41), (([],), 0)],
 'code_144': [(("1/2","2/1"), False), (("2/1","2/1"), True), (("1/2","1/2"), False)],
 'code_145': [(([12,3,21],), [21,3,12]), (([-12,1],), [1,-12]), (([],), [])],
 'code_146': [(([11,13,21,31,33],), 2), (([11,13],), 0)],
 'code_151': [(([1,3,5,2,-3,3.0],), 153), (([],), 0)],
 'code_154': [(("abc",""), False), (("hello","ell"), True), (("abcd","abd"), False)],
 'code_157': [((3,4,5), False), ((4,4,5), True), ((1,1,2), False), ((0,2,2), False)],
 'code_158': [((["ab","ba","a"],), "ba"), ((["abc","zz"],), "abc")],
 'code_159': [((5,3,2), [5,2]), ((5,2,3), [7,1])],
 'code_160': [((["+","*","-"],[2,3,4,5]), 15), ((["+","**"],[2,3,2]), 25)],
 'code_162': [(("é",), hashlib.md5("é".encode('utf-8')).hexdigest()), (("",), None), (("Hello world",), '3e25960a79dbc69b674cd4ec67a72c62')],
 'code_163': [((2,8), [8,6,4,2]), ((8,2), [8,6,4,2]), ((10,14), [])],
}
assert not (CASES.keys() & ADDITIONAL.keys())
CASES.update(ADDITIONAL)
assert len(CASES) == 164
# The current repaired original reference returns '0' at zero. The revised
# reference must preserve it. Negative inputs remain unspecified in the
# original English task and its current implementation does not terminate.
CASES['code_44'] = [((6,2), '011'), ((13,3), '111'), ((0,2), '0')]

# Previously specified exhaustive finite property oracles, reused transparently.
from sys import path as _path
_path.insert(0, str(ROOT / 'valid_corrections'))
from verify_code_corrections import oracle_cases
for ident in ('code_10','code_19','code_134','code_157'):
    CASES[ident] += [(tuple(t['args']), t['expected']) for t in oracle_cases(ident)]
for n in range(9):
    for chars in itertools.product('[]', repeat=n):
        s=''.join(chars)
        balances=[s[:i].count('[')-s[:i].count(']') for i in range(n+1)]
        expected=balances[-1] == 0 and min(balances) >= 0 and max(balances) >= 2
        CASES['code_132'].append(((s,), expected))
