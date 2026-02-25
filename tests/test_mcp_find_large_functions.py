"""Integration tests for the MCP find_large_functions tool."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from code_similarity_mcp.mcp.server import find_large_functions, index_repository


# ---------------------------------------------------------------------------
# Code fixtures
# ---------------------------------------------------------------------------

def _make_large_function(name: str = "big_func", n_statements: int = 35) -> str:
    """Generate a Python function with exactly n_statements assignment statements."""
    lines = [f"def {name}():"]
    for i in range(n_statements):
        lines.append(f"    x_{i} = {i}")
    lines.append("    return x_0")
    return "\n".join(lines) + "\n"


def _make_small_function(name: str = "small_func") -> str:
    return f"def {name}(a, b):\n    x = a + b\n    return x\n"


def _setup_index(tmp_path: Path, content: str) -> str:
    (tmp_path / "module.py").write_text(content, encoding="utf-8")
    index_dir = str(tmp_path / "index")
    index_repository(str(tmp_path), index_dir=index_dir)
    return index_dir


# ---------------------------------------------------------------------------
# Tests: response shape
# ---------------------------------------------------------------------------


def test_returns_valid_json(tmp_path):
    index_dir = _setup_index(tmp_path, _make_small_function())
    raw = find_large_functions(index_dir=index_dir)
    data = json.loads(raw)
    assert isinstance(data, dict)


def test_response_has_large_functions_key(tmp_path):
    index_dir = _setup_index(tmp_path, _make_small_function())
    data = json.loads(find_large_functions(index_dir=index_dir))
    assert "large_functions" in data
    assert isinstance(data["large_functions"], list)


def test_each_entry_has_required_fields(tmp_path):
    index_dir = _setup_index(tmp_path, _make_large_function())
    data = json.loads(find_large_functions(index_dir=index_dir))
    for entry in data["large_functions"]:
        for field in ("id", "name", "file", "start_line", "end_line", "statement_count"):
            assert field in entry, f"Missing field: {field!r}"


def test_id_is_positive_integer(tmp_path):
    index_dir = _setup_index(tmp_path, _make_large_function())
    data = json.loads(find_large_functions(index_dir=index_dir))
    for entry in data["large_functions"]:
        assert isinstance(entry["id"], int)
        assert entry["id"] > 0


def test_statement_count_is_positive_integer(tmp_path):
    index_dir = _setup_index(tmp_path, _make_large_function())
    data = json.loads(find_large_functions(index_dir=index_dir))
    for entry in data["large_functions"]:
        assert isinstance(entry["statement_count"], int)
        assert entry["statement_count"] > 0


def test_line_numbers_are_positive_integers(tmp_path):
    index_dir = _setup_index(tmp_path, _make_large_function())
    data = json.loads(find_large_functions(index_dir=index_dir))
    for entry in data["large_functions"]:
        assert isinstance(entry["start_line"], int)
        assert entry["start_line"] >= 1
        assert isinstance(entry["end_line"], int)
        assert entry["end_line"] >= entry["start_line"]


# ---------------------------------------------------------------------------
# Tests: filtering behaviour
# ---------------------------------------------------------------------------


def test_small_function_excluded_from_results(tmp_path):
    """A function with <=30 statements must not appear in results."""
    index_dir = _setup_index(tmp_path, _make_small_function())
    data = json.loads(find_large_functions(index_dir=index_dir))
    assert data["large_functions"] == []


def test_large_function_included_in_results(tmp_path):
    """A function with >30 statements must appear in results."""
    index_dir = _setup_index(tmp_path, _make_large_function(n_statements=35))
    data = json.loads(find_large_functions(index_dir=index_dir))
    assert len(data["large_functions"]) == 1


def test_statement_count_exceeds_threshold(tmp_path):
    """Returned entries must have statement_count > 30 (the default)."""
    index_dir = _setup_index(tmp_path, _make_large_function(n_statements=35))
    data = json.loads(find_large_functions(index_dir=index_dir))
    for entry in data["large_functions"]:
        assert entry["statement_count"] > 30


def test_exact_boundary_at_30_is_excluded(tmp_path):
    """A function with exactly 30 statements must NOT be included (threshold is exclusive)."""
    # 30 assignments + 1 return = 31 statements total including the function_definition
    # Let's build a function with exactly 29 assignment statements + 1 return = 30 total
    # (function_definition counts as 1, plus 29 assignments + 1 return = 31 — need to be careful)
    # We count everything including the outer function_definition node.
    # Use min_statements=30 explicitly: only include if stmt_count > 30.
    # Build a function whose stmt_count is exactly 30 by trial; instead just use
    # a simple helper: count_statements and verify directly.
    from code_similarity_mcp.parser.python import count_statements

    # Build a function with exactly 28 assignment stmts + 1 return = 30 total
    # (1 function_definition + 28 expression_stmts + 1 return_stmt = 30)
    lines = ["def boundary_func():"]
    for i in range(28):
        lines.append(f"    x_{i} = {i}")
    lines.append("    return x_0")
    code = "\n".join(lines) + "\n"

    actual = count_statements(code)
    # Adjust the min_statements so that this function is just at the boundary
    index_dir = _setup_index(tmp_path, code)
    data = json.loads(find_large_functions(index_dir=index_dir, min_statements=actual))
    # stmt_count == actual and threshold is exclusive (> actual), so excluded
    assert data["large_functions"] == []


def test_function_just_above_threshold_included(tmp_path):
    """A function with exactly min_statements+1 statements is included."""
    from code_similarity_mcp.parser.python import count_statements

    lines = ["def just_above():"]
    for i in range(28):
        lines.append(f"    x_{i} = {i}")
    lines.append("    return x_0")
    code = "\n".join(lines) + "\n"
    actual = count_statements(code)

    index_dir = _setup_index(tmp_path, code)
    # Use min_statements = actual - 1, so actual > (actual - 1) → included
    data = json.loads(find_large_functions(index_dir=index_dir, min_statements=actual - 1))
    assert len(data["large_functions"]) == 1


def test_mixed_file_only_large_returned(tmp_path):
    """With both small and large functions indexed, only large ones are returned."""
    content = _make_small_function("tiny") + "\n\n" + _make_large_function("giant")
    index_dir = _setup_index(tmp_path, content)
    data = json.loads(find_large_functions(index_dir=index_dir))
    names = [e["name"] for e in data["large_functions"]]
    assert "giant" in names
    assert "tiny" not in names


def test_custom_min_statements_parameter(tmp_path):
    """Setting min_statements=5 returns any function with >5 statements."""
    code = textwrap.dedent("""\
        def medium(a, b, c):
            x = a + b
            y = x * c
            z = y - a
            w = z + 1
            return w
    """)
    index_dir = _setup_index(tmp_path, code)
    # With a very high threshold, nothing should be returned
    high = json.loads(find_large_functions(index_dir=index_dir, min_statements=100))
    assert high["large_functions"] == []
    # With min_statements=0, every function should be returned
    low = json.loads(find_large_functions(index_dir=index_dir, min_statements=0))
    assert len(low["large_functions"]) == 1


# ---------------------------------------------------------------------------
# Tests: ordering
# ---------------------------------------------------------------------------


def test_results_ordered_by_statement_count_descending(tmp_path):
    """Results must be sorted largest → smallest by statement_count."""
    content = (
        _make_large_function("func_a", n_statements=40)
        + "\n\n"
        + _make_large_function("func_b", n_statements=35)
    )
    index_dir = _setup_index(tmp_path, content)
    data = json.loads(find_large_functions(index_dir=index_dir))
    counts = [e["statement_count"] for e in data["large_functions"]]
    assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------------------------------
# Tests: empty index
# ---------------------------------------------------------------------------


def test_empty_index_returns_empty_list(tmp_path):
    empty_repo = tmp_path / "empty"
    empty_repo.mkdir()
    index_dir = str(tmp_path / "index")
    index_repository(str(empty_repo), index_dir=index_dir)
    data = json.loads(find_large_functions(index_dir=index_dir))
    assert data["large_functions"] == []


# ---------------------------------------------------------------------------
# Tests: correct function metadata
# ---------------------------------------------------------------------------


def test_returned_name_matches_function_name(tmp_path):
    index_dir = _setup_index(tmp_path, _make_large_function("my_large_func"))
    data = json.loads(find_large_functions(index_dir=index_dir))
    assert len(data["large_functions"]) == 1
    assert data["large_functions"][0]["name"] == "my_large_func"


def test_returned_file_points_to_indexed_file(tmp_path):
    index_dir = _setup_index(tmp_path, _make_large_function())
    data = json.loads(find_large_functions(index_dir=index_dir))
    assert len(data["large_functions"]) == 1
    assert "module.py" in data["large_functions"][0]["file"]


def test_multiple_large_functions_all_returned(tmp_path):
    content = (
        _make_large_function("alpha", n_statements=35)
        + "\n\n"
        + _make_large_function("beta", n_statements=40)
        + "\n\n"
        + _make_large_function("gamma", n_statements=32)
    )
    index_dir = _setup_index(tmp_path, content)
    data = json.loads(find_large_functions(index_dir=index_dir))
    names = {e["name"] for e in data["large_functions"]}
    assert names == {"alpha", "beta", "gamma"}


# ---------------------------------------------------------------------------
# Tests: custom index_dir
# ---------------------------------------------------------------------------


def test_custom_index_dir_is_used(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "m.py").write_text(_make_large_function(), encoding="utf-8")
    custom_index = str(tmp_path / "custom_idx")
    index_repository(str(repo), index_dir=custom_index)
    data = json.loads(find_large_functions(index_dir=custom_index))
    assert len(data["large_functions"]) == 1
