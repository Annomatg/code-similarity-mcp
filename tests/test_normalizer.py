"""Tests for code normalizer."""

import pytest
from code_similarity_mcp.normalizer import normalize_code


def test_strips_line_comments():
    code = "var x = 1  # this is a comment\nvar y = 2"
    result = normalize_code(code)
    assert "#" not in result
    assert "this is a comment" not in result


def test_normalizes_double_quoted_strings():
    code = 'var s = "hello world"'
    result = normalize_code(code)
    assert "hello world" not in result
    assert "STR_LITERAL" in result


def test_normalizes_single_quoted_strings():
    code = "var s = 'hello'"
    result = normalize_code(code)
    assert "hello" not in result
    assert "STR_LITERAL" in result


def test_normalizes_integer_literals():
    code = "var x = 42"
    result = normalize_code(code)
    assert "42" not in result
    assert "NUM_LITERAL" in result


def test_normalizes_float_literals():
    code = "var y = 3.14"
    result = normalize_code(code)
    assert "3.14" not in result
    assert "NUM_LITERAL" in result


def test_normalizes_function_name():
    code = "func calculate_total(a, b):\n    return a + b"
    result = normalize_code(code)
    assert "calculate_total" not in result
    assert "FUNC_NAME" in result


def test_normalizes_parameters():
    code = "func add(price, tax):\n    return price + tax"
    result = normalize_code(code)
    assert "price" not in result
    assert "tax" not in result
    assert "v1" in result
    assert "v2" in result


def test_normalizes_var_declarations():
    code = "func f():\n    var result = 1\n    var final = result + 1\n    return final"
    result = normalize_code(code)
    assert "result" not in result
    assert "final" not in result
    assert "v1" in result
    assert "v2" in result


def test_params_numbered_before_vars():
    code = "func f(a, b):\n    var x = a + b\n    return x"
    result = normalize_code(code)
    # a->v1, b->v2, x->v3
    assert "v1" in result
    assert "v2" in result
    assert "v3" in result


def test_preserves_keywords():
    code = "func f(a):\n    var x = a\n    return x"
    result = normalize_code(code)
    assert "func" in result
    assert "return" in result
    assert "var" in result


def test_normalizes_whitespace():
    code = "var  x  =  1"
    result = normalize_code(code)
    assert "  " not in result


def test_equivalent_functions_same_normalized_output():
    """Two functions with different names/params but same logic normalize identically."""
    code_a = """func calculate(price, tax):
    var total = price + price * tax
    return total"""

    code_b = """func compute(cost, rate):
    var amount = cost + cost * rate
    return amount"""

    assert normalize_code(code_a) == normalize_code(code_b)


def test_different_logic_different_output():
    code_a = "func f(a, b):\n    return a + b"
    code_b = "func g(a, b):\n    return a * b"
    assert normalize_code(code_a) != normalize_code(code_b)
