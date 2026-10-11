"""Independent finite oracles for the five replacement code contracts.

The bounded prime_fib domain is covered completely. Other contracts have
finite coverage, not a proof of every possible submitted implementation.
"""
import itertools


def cases(ident):
    if ident == 'code_22':
        values = [-3, -2, 0, 1, 2, True, False, 2.0, '2']
        result = []
        for size in range(4):
            for seq in itertools.product(values, repeat=size):
                expected = []
                for value in seq:
                    if isinstance(value, int) and not isinstance(value, bool):
                        if value & 1 == 0:
                            expected.append(value)
                result.append(([list(seq)], expected))
        result.append(([[{}, [], None, 4, 4, -6, -5]], [4, 4, -6]))
        return result
    if ident == 'code_39':
        # Enumerate the Fibonacci sequence, then test every possible divisor
        # (not the square-root stopping rule of the revised reference).
        fib = [0, 1]
        while fib[-1] < 28657:
            fib.append(sum(fib[-2:]))
        primes = [value for value in fib if value > 1 and
                  all(value % divisor for divisor in range(2, value))]
        assert primes == [2, 3, 5, 13, 89, 233, 1597, 28657]
        return [([n], primes[:n]) for n in range(9)]
    if ident == 'code_44':
        def conventional_digits(x, base):
            if x < base:
                return str(x)
            return conventional_digits(x // base, base) + str(x % base)
        result = []
        for x in list(range(251)) + [1000, 10001, 65536]:
            for base in range(2, 10):
                expected = conventional_digits(x, base)[::-1]
                assert int(expected[::-1], base) == x
                assert x == 0 or expected[-1] != '0'
                result.append(([x, base], expected))
        return result
    if ident == 'code_49':
        result = []
        for n in list(range(26)) + [100, 1101]:
            for p in range(1, 26):
                value = 1 % p
                for _ in range(n):
                    value = (value + value + value) % p
                result.append(([n, p], value))
        return result
    if ident == 'code_76':
        result = []
        for n in range(13):
            # For n>=2, 2**10>300. Thus all qualifying x<=300
            # occur among these powers. For n in {0,1}, powers are constant.
            possible = {n ** exponent for exponent in range(2, 19, 2)}
            for x in range(301):
                result.append(([x, n], x in possible))
        return result
    raise KeyError(ident)


def packed_cases(ident):
    return [{'args': args, 'expected_repr': repr(expected)}
            for args, expected in cases(ident)]
