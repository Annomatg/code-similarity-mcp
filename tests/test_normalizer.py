"""Tests for code normalizer."""

import pytest
from code_similarity_mcp.normalizer import normalize_code


def test_strips_line_comments():
    code = "def f(x):\n    y = x  # this is a comment\n    return y"
    result = normalize_code(code)
    assert "#" not in result
    assert "this is a comment" not in result


def test_normalizes_double_quoted_strings():
    code = 'def f():\n    s = "hello world"\n    return s'
    result = normalize_code(code)
    assert "hello world" not in result
    assert "STR_LITERAL" in result


def test_normalizes_single_quoted_strings():
    code = "def f():\n    s = 'hello'\n    return s"
    result = normalize_code(code)
    assert "hello" not in result
    assert "STR_LITERAL" in result


def test_normalizes_integer_literals():
    code = "def f():\n    x = 42\n    return x"
    result = normalize_code(code)
    assert "42" not in result
    assert "NUM_LITERAL" in result


def test_normalizes_float_literals():
    code = "def f():\n    y = 3.14\n    return y"
    result = normalize_code(code)
    assert "3.14" not in result
    assert "NUM_LITERAL" in result


def test_normalizes_function_name():
    code = "def calculate_total(a, b):\n    return a + b"
    result = normalize_code(code)
    assert "calculate_total" not in result
    assert "FUNC_NAME" in result


def test_normalizes_parameters():
    code = "def add(price, tax):\n    return price + tax"
    result = normalize_code(code)
    assert "price" not in result
    assert "tax" not in result
    assert "v1" in result
    assert "v2" in result


def test_normalizes_var_assignments():
    code = "def f():\n    result = 1\n    final = result + 1\n    return final"
    result = normalize_code(code)
    assert "result" not in result
    assert "final" not in result
    assert "v1" in result
    assert "v2" in result


def test_params_numbered_before_vars():
    code = "def f(a, b):\n    x = a + b\n    return x"
    result = normalize_code(code)
    # a->v1, b->v2, x->v3
    assert "v1" in result
    assert "v2" in result
    assert "v3" in result


def test_self_excluded_from_params():
    code = "def f(self, value):\n    return self.data + value"
    result = normalize_code(code)
    assert "self" in result      # self preserved
    assert "value" not in result
    assert "v1" in result        # value -> v1


def test_preserves_keywords():
    code = "def f(a):\n    x = a\n    return x"
    result = normalize_code(code)
    assert "def" in result
    assert "return" in result


def test_normalizes_whitespace():
    code = "def  f(x):    return  x"
    result = normalize_code(code)
    assert "  " not in result


def test_annotated_assignment_normalized():
    code = "def f(a):\n    result: int = a + 1\n    return result"
    result = normalize_code(code)
    assert "result" not in result
    assert "v1" in result   # a -> v1
    assert "v2" in result   # result -> v2


def test_equivalent_functions_same_normalized_output():
    """Two functions with different names/params but same logic normalize identically."""
    code_a = "def calculate(price, tax):\n    total = price + price * tax\n    return total"
    code_b = "def compute(cost, rate):\n    amount = cost + cost * rate\n    return amount"
    assert normalize_code(code_a) == normalize_code(code_b)


def test_different_logic_different_output():
    code_a = "def f(a, b):\n    return a + b"
    code_b = "def g(a, b):\n    return a * b"
    assert normalize_code(code_a) != normalize_code(code_b)


def test_unsupported_language_raises():
    with pytest.raises(ValueError, match="Unsupported language"):
        normalize_code("some code", language="cobol")
