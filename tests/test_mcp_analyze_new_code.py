"""Integration tests for the MCP analyze_new_code tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from code_similarity_mcp.mcp.server import analyze_new_code, index_repository


# ---------------------------------------------------------------------------
# Code fixtures
# ---------------------------------------------------------------------------

_INDEXED_PY = """\
def add(a, b):
    return a + b


def subtract(x, y):
    return x - y


def multiply(a, b):
    result = a * b
    return result
"""

# Semantically equivalent to indexed 'add' — normalizer makes it identical
_SNIPPET_EQUIVALENT = """\
def add_numbers(x, y):
    return x + y
"""

# Byte-for-byte copy of indexed 'add' — guaranteed exact match
_SNIPPET_EXACT = """\
def add(a, b):
    return a + b
"""

# Snippet with two methods both present in the index
_SNIPPET_MULTI = """\
def add(a, b):
    return a + b


def subtract(x, y):
    return x - y
"""

# Code that is NOT a function definition
_SNIPPET_NO_METHODS = """\
x = 1 + 2
print(x)
"""

# Method with different structure (extra local variable, different body)
_SNIPPET_DIFFERENT = """\
def compute(a, b):
    total = a + b + 100
    prefix = total * 2
    suffix = prefix - a
    return suffix
"""


def _setup_index(tmp_path: Path, content: str = _INDEXED_PY) -> str:
    """Write a Python file and index it. Returns the index_dir path string."""
    (tmp_path / "module.py").write_text(content, encoding="utf-8")
    index_dir = str(tmp_path / "index")
    index_repository(str(tmp_path), index_dir=index_dir)
    return index_dir


# ---------------------------------------------------------------------------
# Tests: response shape
# ---------------------------------------------------------------------------


def test_returns_valid_json(tmp_path):
    index_dir = _setup_index(tmp_path)
    raw = analyze_new_code(_SNIPPET_EQUIVALENT, language="python", index_dir=index_dir)
    data = json.loads(raw)
    assert "new_methods" in data


def test_new_methods_is_list(tmp_path):
    index_dir = _setup_index(tmp_path)
    data = json.loads(
        analyze_new_code(_SNIPPET_EQUIVALENT, language="python", index_dir=index_dir)
    )
    assert isinstance(data["new_methods"], list)


def test_method_entry_has_name_and_parameters(tmp_path):
    index_dir = _setup_index(tmp_path)
    data = json.loads(
        analyze_new_code(_SNIPPET_EQUIVALENT, language="python", index_dir=index_dir)
    )
    entry = data["new_methods"][0]
    assert "name" in entry
    assert "parameters" in entry
    assert isinstance(entry["parameters"], list)


def test_candidate_has_required_fields(tmp_path):
    index_dir = _setup_index(tmp_path)
    data = json.loads(
        analyze_new_code(_SNIPPET_EQUIVALENT, language="python", index_dir=index_dir)
    )
    candidates = data["new_methods"][0]["candidates"]
    assert len(candidates) > 0
    c = candidates[0]
    for field in ("file", "method", "line", "score", "exact_match",
                  "embedding_similarity", "ast_similarity",
                  "differences", "refactoring_hints"):
        assert field in c, f"Missing field: {field!r}"


def test_differences_is_list(tmp_path):
    index_dir = _setup_index(tmp_path)
    data = json.loads(
        analyze_new_code(_SNIPPET_EQUIVALENT, language="python", index_dir=index_dir)
    )
    for m in data["new_methods"]:
        for c in m["candidates"]:
            assert isinstance(c["differences"], list)


def test_refactoring_hints_is_list(tmp_path):
    index_dir = _setup_index(tmp_path)
    data = json.loads(
        analyze_new_code(_SNIPPET_EQUIVALENT, language="python", index_dir=index_dir)
    )
    for m in data["new_methods"]:
        for c in m["candidates"]:
            assert isinstance(c["refactoring_hints"], list)


def test_score_is_float_between_0_and_1(tmp_path):
    index_dir = _setup_index(tmp_path)
    data = json.loads(
        analyze_new_code(_SNIPPET_EQUIVALENT, language="python", index_dir=index_dir)
    )
    for m in data["new_methods"]:
        for c in m["candidates"]:
            assert 0.0 <= c["score"] <= 1.0


def test_candidates_ordered_by_score_descending(tmp_path):
    index_dir = _setup_index(tmp_path)
    data = json.loads(
        analyze_new_code(_SNIPPET_EQUIVALENT, language="python", index_dir=index_dir)
    )
    scores = [c["score"] for c in data["new_methods"][0]["candidates"]]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Tests: method extraction from snippet
# ---------------------------------------------------------------------------


def test_single_method_snippet_produces_one_entry(tmp_path):
    index_dir = _setup_index(tmp_path)
    data = json.loads(
        analyze_new_code(_SNIPPET_EQUIVALENT, language="python", index_dir=index_dir)
    )
    assert len(data["new_methods"]) == 1
    assert data["new_methods"][0]["name"] == "add_numbers"


def test_multi_method_snippet_produces_multiple_entries(tmp_path):
    index_dir = _setup_index(tmp_path)
    data = json.loads(
        analyze_new_code(_SNIPPET_MULTI, language="python", index_dir=index_dir)
    )
    assert len(data["new_methods"]) == 2
    names = {m["name"] for m in data["new_methods"]}
    assert names == {"add", "subtract"}


# ---------------------------------------------------------------------------
# Tests: exact match detection
# ---------------------------------------------------------------------------


def test_exact_match_is_detected(tmp_path):
    """When the snippet is byte-identical to an indexed method, exact_match=True."""
    index_dir = _setup_index(tmp_path)
    data = json.loads(
        analyze_new_code(_SNIPPET_EXACT, language="python", index_dir=index_dir)
    )
    candidates = data["new_methods"][0]["candidates"]
    assert any(c["exact_match"] for c in candidates)


def test_exact_match_score_is_one(tmp_path):
    """An exact match (same normalized code) produces score=1.0."""
    index_dir = _setup_index(tmp_path)
    data = json.loads(
        analyze_new_code(_SNIPPET_EXACT, language="python", index_dir=index_dir)
    )
    candidates = data["new_methods"][0]["candidates"]
    exact = [c for c in candidates if c["exact_match"]]
    assert len(exact) > 0
    assert exact[0]["score"] == 1.0


def test_semantically_equivalent_triggers_exact_match(tmp_path):
    """Normalizer collapses 'add_numbers(x,y)' and 'add(a,b)' to same code."""
    index_dir = _setup_index(tmp_path)
    data = json.loads(
        analyze_new_code(_SNIPPET_EQUIVALENT, language="python", index_dir=index_dir)
    )
    candidates = data["new_methods"][0]["candidates"]
    # The normalizer makes them identical → exact_match is expected
    assert any(c["exact_match"] for c in candidates)


# ---------------------------------------------------------------------------
# Tests: empty / no methods
# ---------------------------------------------------------------------------


def test_empty_snippet_returns_empty_list(tmp_path):
    index_dir = _setup_index(tmp_path)
    data = json.loads(
        analyze_new_code("", language="python", index_dir=index_dir)
    )
    assert data["new_methods"] == []


def test_snippet_with_no_functions_returns_empty_list_and_note(tmp_path):
    index_dir = _setup_index(tmp_path)
    data = json.loads(
        analyze_new_code(_SNIPPET_NO_METHODS, language="python", index_dir=index_dir)
    )
    assert data["new_methods"] == []
    assert "note" in data


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


def test_unsupported_language_returns_error(tmp_path):
    index_dir = _setup_index(tmp_path)
    data = json.loads(
        analyze_new_code("def foo(): pass", language="cobol", index_dir=index_dir)
    )
    assert "error" in data


# ---------------------------------------------------------------------------
# Tests: top_k parameter
# ---------------------------------------------------------------------------


def test_top_k_one_limits_candidates(tmp_path):
    index_dir = _setup_index(tmp_path)
    data = json.loads(
        analyze_new_code(_SNIPPET_EQUIVALENT, language="python", top_k=1, index_dir=index_dir)
    )
    assert len(data["new_methods"][0]["candidates"]) <= 1


def test_top_k_default_three_caps_candidates(tmp_path):
    index_dir = _setup_index(tmp_path)
    data = json.loads(
        analyze_new_code(_SNIPPET_EQUIVALENT, language="python", index_dir=index_dir)
    )
    assert len(data["new_methods"][0]["candidates"]) <= 3


def test_top_k_two_limits_candidates(tmp_path):
    index_dir = _setup_index(tmp_path)
    data = json.loads(
        analyze_new_code(_SNIPPET_EQUIVALENT, language="python", top_k=2, index_dir=index_dir)
    )
    assert len(data["new_methods"][0]["candidates"]) <= 2


# ---------------------------------------------------------------------------
# Tests: empty index
# ---------------------------------------------------------------------------


def test_empty_index_produces_no_candidates(tmp_path):
    """Analyzing against an index with no methods returns empty candidates list."""
    # Create an empty indexed repo (no Python files)
    empty_repo = tmp_path / "empty_repo"
    empty_repo.mkdir()
    index_dir = str(tmp_path / "index")
    index_repository(str(empty_repo), index_dir=index_dir)

    data = json.loads(
        analyze_new_code(_SNIPPET_EQUIVALENT, language="python", index_dir=index_dir)
    )
    assert data["new_methods"][0]["candidates"] == []


# ---------------------------------------------------------------------------
# Tests: similarity quality
# ---------------------------------------------------------------------------


def test_similar_method_appears_in_candidates(tmp_path):
    """A semantically equivalent function should appear as a candidate."""
    index_dir = _setup_index(tmp_path)
    data = json.loads(
        analyze_new_code(_SNIPPET_EQUIVALENT, language="python", index_dir=index_dir)
    )
    candidates = data["new_methods"][0]["candidates"]
    assert len(candidates) > 0
    candidate_methods = {c["method"] for c in candidates}
    assert "add" in candidate_methods


def test_candidate_references_correct_file(tmp_path):
    """The candidate should point back to the indexed file."""
    index_dir = _setup_index(tmp_path)
    data = json.loads(
        analyze_new_code(_SNIPPET_EQUIVALENT, language="python", index_dir=index_dir)
    )
    candidates = data["new_methods"][0]["candidates"]
    assert len(candidates) > 0
    # All candidates come from the single indexed file
    assert all("module.py" in c["file"] for c in candidates)


def test_candidate_line_is_positive_integer(tmp_path):
    index_dir = _setup_index(tmp_path)
    data = json.loads(
        analyze_new_code(_SNIPPET_EQUIVALENT, language="python", index_dir=index_dir)
    )
    for m in data["new_methods"]:
        for c in m["candidates"]:
            assert isinstance(c["line"], int)
            assert c["line"] >= 1


# ---------------------------------------------------------------------------
# Tests: differences and refactoring hints content
# ---------------------------------------------------------------------------


def test_exact_match_has_no_differences(tmp_path):
    """An exact match should report no differences."""
    index_dir = _setup_index(tmp_path)
    data = json.loads(
        analyze_new_code(_SNIPPET_EXACT, language="python", index_dir=index_dir)
    )
    exact = [
        c for c in data["new_methods"][0]["candidates"]
        if c["exact_match"]
    ]
    assert len(exact) > 0
    assert exact[0]["differences"] == []
    assert exact[0]["refactoring_hints"] == []


# ---------------------------------------------------------------------------
# Tests: custom index_dir
# ---------------------------------------------------------------------------


def test_custom_index_dir_is_used(tmp_path):
    """Results come from the specified custom index, not the default one."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "m.py").write_text(_INDEXED_PY, encoding="utf-8")
    custom_index = str(tmp_path / "custom_idx")
    index_repository(str(repo), index_dir=custom_index)

    data = json.loads(
        analyze_new_code(_SNIPPET_EQUIVALENT, language="python", index_dir=custom_index)
    )
    assert len(data["new_methods"]) == 1
    assert len(data["new_methods"][0]["candidates"]) > 0


def test_two_separate_indexes_are_independent(tmp_path):
    """Two separately indexed repos should each return only their own methods."""
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    repo_a.mkdir()
    repo_b.mkdir()

    (repo_a / "ops.py").write_text(_INDEXED_PY, encoding="utf-8")
    (repo_b / "other.py").write_text(
        "def totally_different(x, y, z):\n    return x * y - z\n",
        encoding="utf-8",
    )

    idx_a = str(tmp_path / "idx_a")
    idx_b = str(tmp_path / "idx_b")
    index_repository(str(repo_a), index_dir=idx_a)
    index_repository(str(repo_b), index_dir=idx_b)

    # Analyzing against idx_a should find 'add'; idx_b has 'totally_different'
    data_a = json.loads(
        analyze_new_code(_SNIPPET_EQUIVALENT, language="python", index_dir=idx_a)
    )
    data_b = json.loads(
        analyze_new_code(_SNIPPET_EQUIVALENT, language="python", index_dir=idx_b)
    )

    methods_a = {c["method"] for c in data_a["new_methods"][0]["candidates"]}
    methods_b = {c["method"] for c in data_b["new_methods"][0]["candidates"]}

    assert "add" in methods_a
    assert "add" not in methods_b
