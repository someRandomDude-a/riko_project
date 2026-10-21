"""The calculator evaluates model-controlled text, so it must never execute anything but arithmetic."""
import math
import time

import pytest

from process.app_core.tools.builtin.scientific_calculator import Tool, evaluate


@pytest.mark.parametrize('expression, expected', [
    ('sin(pi/4) + log(100)', repr(math.sin(math.pi / 4) + math.log(100))),
    ('sqrt(2**10)', '32.0'), ('5 * (3 + 2) - 4', '21'), ('atan2(3, 4) * 180 / pi', repr(math.atan2(3, 4) * 180 / math.pi)),
    ('pow(2, 8)', '256.0'), ('log10(1000)', '3.0'), ('exp(1)', repr(math.e)), ('-3 % 5', '2'), ('7 // 2', '3'),
    ('2 ** -2', '0.25'), ('hypot(3, 4)', '5.0'), ('+tau - 2*pi', '0.0'),
])
def test_ordinary_arithmetic_still_works(expression, expected):
    assert Tool({}).execute(expression=expression) == expected


@pytest.mark.parametrize('expression', [
    "().__class__.__base__.__subclasses__()",
    "[c for c in ().__class__.__base__.__subclasses__() if c.__name__ == 'catch_warnings'][0]()._module"
    ".__builtins__['__import__']('os').system('touch {marker}')",
    "__import__('os').system('touch {marker}')", "(lambda: 1)()", "open('{marker}', 'w')", "'a' * 10",
    "sqrt.__self__", "pi.real", "x", "sin(x=1)", "[1, 2]", "{{1: 2}}", "True + 1", "1 if 1 else 2", "1 < 2",
])
def test_code_execution_and_non_arithmetic_syntax_are_rejected(expression, tmp_path):
    marker = tmp_path / 'pwned'
    with pytest.raises(ValueError): evaluate(expression.format(marker=marker))
    assert not marker.exists()


@pytest.mark.parametrize('expression', ['9**9**9', '10**4000', '(10**3000) * (10**3000)', '2**10**4', '2**10000', '3**9999'])
def test_huge_results_are_refused_quickly(expression):
    started = time.perf_counter()
    with pytest.raises(ValueError, match='too large'): evaluate(expression)
    assert time.perf_counter() - started < 1


def test_results_up_to_the_limit_agree_however_they_are_written():
    assert evaluate('2**9998') == evaluate('2**4999 * 2**4999')
    assert evaluate('10**3000') == evaluate('(10**1500) * (10**1500)')


def test_long_flat_sums_and_products_are_not_mistaken_for_nesting():
    assert evaluate('+'.join(str(n) for n in range(1, 201))) == sum(range(1, 201))  # 691 characters, 200 terms
    assert evaluate('*'.join(str(n) for n in range(1, 103))) == math.factorial(102)


@pytest.mark.parametrize('expression, message', [
    ('-' * 500 + '1', 'nested too deeply'), ('1+' * 600 + '1', 'longer than'), ('1/0', 'division by zero'),
    ('(-8) ** 0.5', 'complex'), ('sqrt(-1)', 'Math domain error in sqrt'), ('log(0)', 'Math domain error in log'),
    ('exp(1000)', 'range'), ('sqrt(1, 2)', 'argument'), ('', 'Provide'), ('1 +', 'Invalid'),
    ('[' * 199 + ']' * 199 + ' 1', 'Invalid expression: .+'), ('(-' * 199 + ')' * 199, 'Invalid expression: .+'),
    ('2^10', 'use \\*\\*'), ('factorial(5)', "Unknown function 'factorial'"), ('x + 1', "Unknown name 'x'"),
    ('sin(x=1)', 'positional arguments only'), ("'a'", 'Only numbers'),
])
def test_bad_input_is_a_clear_error(expression, message):
    with pytest.raises(ValueError, match=message): evaluate(expression)
