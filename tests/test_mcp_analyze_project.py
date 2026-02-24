"""Integration tests for the MCP analyze_project tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from code_similarity_mcp.mcp.server import analyze_project, index_repository


# ---------------------------------------------------------------------------
# Code fixtures
# ---------------------------------------------------------------------------

# Two semantically identical functions (normalizer collapses them to same code) — 4 lines each
_IDENTICAL_PY = """\
def add(a, b):
    total = a + b
    x = total
    return x


def sum_values(x, y):
    total = x + y
    v = total
    return v
"""

# Three functions: two identical + one structurally different — 4+ lines each
_MIXED_PY = """\
def add(a, b):
    total = a + b
    x = total
    return x


def sum_values(x, y):
    total = x + y
    v = total
    return v


def multiply(a, b):
    result = a * b
    squared = result * result
    return squared
"""

# Two functions with very different param counts — fast filter will exclude each other
_DIFFERENT_PY = """\
def add(a, b):
    return a + b


def process_data(items, key, value, extra):
    result = []
    for item in items:
        if item.get(key) == value:
            result.append(item[extra])
    return result
"""

# Two short functions (2 lines each) — below default min_lines threshold
_SHORT_PY = """\
def add(a, b):
    return a + b


def sum_two(x, y):
    return x + y
"""


def _setup_index(tmp_path: Path, content: str) -> str:
    """Write a Python file and index it. Returns the index_dir path string."""
    (tmp_path / "module.py").write_text(content, encoding="utf-8")
    index_dir = str(tmp_path / "index")
    index_repository(str(tmp_path), index_dir=index_dir)
    return index_dir


# ---------------------------------------------------------------------------
# Tests: response shape
# ---------------------------------------------------------------------------


def test_returns_valid_json(tmp_path):
    index_dir = _setup_index(tmp_path, _IDENTICAL_PY)
    raw = analyze_project(index_dir=index_dir)
    data = json.loads(raw)
    assert isinstance(data, dict)


def test_response_has_total_methods(tmp_path):
    index_dir = _setup_index(tmp_path, _IDENTICAL_PY)
    data = json.loads(analyze_project(index_dir=index_dir))
    assert "total_methods" in data


def test_response_has_similar_pairs(tmp_path):
    index_dir = _setup_index(tmp_path, _IDENTICAL_PY)
    data = json.loads(analyze_project(index_dir=index_dir))
    assert "similar_pairs" in data
    assert isinstance(data["similar_pairs"], list)


def test_total_methods_count_is_correct(tmp_path):
    index_dir = _setup_index(tmp_path, _IDENTICAL_PY)
    data = json.loads(analyze_project(index_dir=index_dir))
    assert data["total_methods"] == 2


# ---------------------------------------------------------------------------
# Tests: pair structure
# ---------------------------------------------------------------------------


def test_pair_has_method_a_and_method_b(tmp_path):
    index_dir = _setup_index(tmp_path, _IDENTICAL_PY)
    data = json.loads(analyze_project(index_dir=index_dir))
    assert len(data["similar_pairs"]) > 0
    pair = data["similar_pairs"][0]
    assert "method_a" in pair
    assert "method_b" in pair


def test_method_entry_has_file_method_line(tmp_path):
    index_dir = _setup_index(tmp_path, _IDENTICAL_PY)
    data = json.loads(analyze_project(index_dir=index_dir))
    pair = data["similar_pairs"][0]
    for key in ("method_a", "method_b"):
        entry = pair[key]
        assert "file" in entry
        assert "method" in entry
        assert "line" in entry


def test_pair_has_required_score_fields(tmp_path):
    index_dir = _setup_index(tmp_path, _IDENTICAL_PY)
    data = json.loads(analyze_project(index_dir=index_dir))
    pair = data["similar_pairs"][0]
    for field in ("score", "exact_match", "embedding_similarity", "ast_similarity",
                  "differences", "refactoring_hints"):
        assert field in pair, f"Missing field: {field!r}"


def test_score_is_float_between_0_and_1(tmp_path):
    index_dir = _setup_index(tmp_path, _IDENTICAL_PY)
    data = json.loads(analyze_project(index_dir=index_dir))
    for pair in data["similar_pairs"]:
        assert 0.0 <= pair["score"] <= 1.0


def test_line_numbers_are_positive_integers(tmp_path):
    index_dir = _setup_index(tmp_path, _IDENTICAL_PY)
    data = json.loads(analyze_project(index_dir=index_dir))
    for pair in data["similar_pairs"]:
        assert isinstance(pair["method_a"]["line"], int)
        assert pair["method_a"]["line"] >= 1
        assert isinstance(pair["method_b"]["line"], int)
        assert pair["method_b"]["line"] >= 1


def test_differences_and_hints_are_lists(tmp_path):
    index_dir = _setup_index(tmp_path, _IDENTICAL_PY)
    data = json.loads(analyze_project(index_dir=index_dir))
    for pair in data["similar_pairs"]:
        assert isinstance(pair["differences"], list)
        assert isinstance(pair["refactoring_hints"], list)


# ---------------------------------------------------------------------------
# Tests: empty index
# ---------------------------------------------------------------------------


def test_empty_index_returns_zero_methods_and_no_pairs(tmp_path):
    empty_repo = tmp_path / "empty"
    empty_repo.mkdir()
    index_dir = str(tmp_path / "index")
    index_repository(str(empty_repo), index_dir=index_dir)
    data = json.loads(analyze_project(index_dir=index_dir))
    assert data["total_methods"] == 0
    assert data["similar_pairs"] == []


# ---------------------------------------------------------------------------
# Tests: single method — no pairs possible
# ---------------------------------------------------------------------------


def test_single_method_produces_no_pairs(tmp_path):
    (tmp_path / "single.py").write_text("def solo(x):\n    return x * 2\n", encoding="utf-8")
    index_dir = str(tmp_path / "index")
    index_repository(str(tmp_path), index_dir=index_dir)
    data = json.loads(analyze_project(index_dir=index_dir))
    assert data["total_methods"] == 1
    assert data["similar_pairs"] == []


# ---------------------------------------------------------------------------
# Tests: exact match detection
# ---------------------------------------------------------------------------


def test_identical_methods_produce_exact_match_pair(tmp_path):
    index_dir = _setup_index(tmp_path, _IDENTICAL_PY)
    data = json.loads(analyze_project(index_dir=index_dir))
    assert any(pair["exact_match"] for pair in data["similar_pairs"])


def test_exact_match_pair_score_is_one(tmp_path):
    index_dir = _setup_index(tmp_path, _IDENTICAL_PY)
    data = json.loads(analyze_project(index_dir=index_dir))
    exact_pairs = [p for p in data["similar_pairs"] if p["exact_match"]]
    assert len(exact_pairs) > 0
    assert exact_pairs[0]["score"] == 1.0


def test_exact_match_has_no_differences(tmp_path):
    index_dir = _setup_index(tmp_path, _IDENTICAL_PY)
    data = json.loads(analyze_project(index_dir=index_dir))
    exact_pairs = [p for p in data["similar_pairs"] if p["exact_match"]]
    assert len(exact_pairs) > 0
    assert exact_pairs[0]["differences"] == []
    assert exact_pairs[0]["refactoring_hints"] == []


# ---------------------------------------------------------------------------
# Tests: self-comparison excluded
# ---------------------------------------------------------------------------


def test_method_not_paired_with_itself(tmp_path):
    index_dir = _setup_index(tmp_path, _IDENTICAL_PY)
    data = json.loads(analyze_project(index_dir=index_dir))
    for pair in data["similar_pairs"]:
        a, b = pair["method_a"], pair["method_b"]
        assert not (
            a["file"] == b["file"]
            and a["method"] == b["method"]
            and a["line"] == b["line"]
        ), "A method was paired with itself"


# ---------------------------------------------------------------------------
# Tests: pair deduplication
# ---------------------------------------------------------------------------


def test_identical_pair_appears_exactly_once(tmp_path):
    """add ↔ sum_values is one pair; should not appear twice."""
    index_dir = _setup_index(tmp_path, _IDENTICAL_PY)
    data = json.loads(analyze_project(index_dir=index_dir))
    assert len(data["similar_pairs"]) == 1


def test_pairs_are_not_duplicated(tmp_path):
    index_dir = _setup_index(tmp_path, _MIXED_PY)
    data = json.loads(analyze_project(index_dir=index_dir))
    seen: set[frozenset] = set()
    for pair in data["similar_pairs"]:
        key = frozenset([
            (pair["method_a"]["file"], pair["method_a"]["method"], pair["method_a"]["line"]),
            (pair["method_b"]["file"], pair["method_b"]["method"], pair["method_b"]["line"]),
        ])
        assert key not in seen, "Duplicate pair found"
        seen.add(key)


# ---------------------------------------------------------------------------
# Tests: ordering
# ---------------------------------------------------------------------------


def test_pairs_ordered_by_score_descending(tmp_path):
    index_dir = _setup_index(tmp_path, _MIXED_PY)
    data = json.loads(analyze_project(index_dir=index_dir))
    scores = [p["score"] for p in data["similar_pairs"]]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Tests: threshold parameter
# ---------------------------------------------------------------------------


def test_high_threshold_reduces_pairs(tmp_path):
    index_dir = _setup_index(tmp_path, _MIXED_PY)
    default_data = json.loads(analyze_project(index_dir=index_dir))
    strict_data = json.loads(analyze_project(index_dir=index_dir, threshold=0.99))
    assert len(strict_data["similar_pairs"]) <= len(default_data["similar_pairs"])


def test_threshold_one_only_includes_exact_matches(tmp_path):
    index_dir = _setup_index(tmp_path, _MIXED_PY)
    data = json.loads(analyze_project(index_dir=index_dir, threshold=1.0))
    for pair in data["similar_pairs"]:
        assert pair["exact_match"]
        assert pair["score"] == 1.0


def test_zero_threshold_returns_all_scored_pairs(tmp_path):
    index_dir = _setup_index(tmp_path, _MIXED_PY)
    low_data = json.loads(analyze_project(index_dir=index_dir, threshold=0.0))
    high_data = json.loads(analyze_project(index_dir=index_dir, threshold=0.85))
    # With threshold=0, we should get at least as many pairs as with 0.85
    assert len(low_data["similar_pairs"]) >= len(high_data["similar_pairs"])


# ---------------------------------------------------------------------------
# Tests: top_k parameter
# ---------------------------------------------------------------------------


def test_top_k_one_limits_matches_per_method(tmp_path):
    """With top_k=1, each method finds at most 1 similar partner; deduplication
    ensures the total pairs ≤ number of methods."""
    (tmp_path / "many.py").write_text(
        "\n\n".join(
            f"def func_{i}(a, b):\n    total = a + b\n    x = total\n    return x"
            for i in range(6)
        ),
        encoding="utf-8",
    )
    index_dir = str(tmp_path / "index")
    index_repository(str(tmp_path), index_dir=index_dir)
    data = json.loads(analyze_project(index_dir=index_dir, top_k=1))
    assert data["total_methods"] == 6
    # With top_k=1 and deduplication, pairs ≤ total_methods
    assert len(data["similar_pairs"]) <= data["total_methods"]


# ---------------------------------------------------------------------------
# Tests: dissimilar methods produce no pairs
# ---------------------------------------------------------------------------


def test_dissimilar_methods_produce_no_pairs_above_threshold(tmp_path):
    """Methods with very different param counts are filtered before scoring."""
    index_dir = _setup_index(tmp_path, _DIFFERENT_PY)
    data = json.loads(analyze_project(index_dir=index_dir, threshold=0.85))
    assert data["similar_pairs"] == []


# ---------------------------------------------------------------------------
# Tests: custom index_dir
# ---------------------------------------------------------------------------


def test_custom_index_dir_is_used(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "m.py").write_text(_IDENTICAL_PY, encoding="utf-8")
    custom_index = str(tmp_path / "custom_idx")
    index_repository(str(repo), index_dir=custom_index)
    data = json.loads(analyze_project(index_dir=custom_index))
    assert data["total_methods"] == 2


# ---------------------------------------------------------------------------
# Tests: multiple files
# ---------------------------------------------------------------------------


def test_total_methods_spans_multiple_files(tmp_path):
    (tmp_path / "a.py").write_text(_IDENTICAL_PY, encoding="utf-8")
    (tmp_path / "b.py").write_text("def multiply(a, b):\n    return a * b\n", encoding="utf-8")
    index_dir = str(tmp_path / "index")
    index_repository(str(tmp_path), index_dir=index_dir)
    data = json.loads(analyze_project(index_dir=index_dir))
    assert data["total_methods"] == 3


def test_cross_file_pairs_are_detected(tmp_path):
    """A similar function split across two files should still be paired."""
    (tmp_path / "a.py").write_text(
        "def add(a, b):\n    total = a + b\n    x = total\n    return x\n", encoding="utf-8"
    )
    (tmp_path / "b.py").write_text(
        "def sum_values(x, y):\n    total = x + y\n    v = total\n    return v\n", encoding="utf-8"
    )
    index_dir = str(tmp_path / "index")
    index_repository(str(tmp_path), index_dir=index_dir)
    data = json.loads(analyze_project(index_dir=index_dir))
    assert len(data["similar_pairs"]) > 0
    pair = data["similar_pairs"][0]
    # The pair should span two different files
    assert pair["method_a"]["file"] != pair["method_b"]["file"]


# ---------------------------------------------------------------------------
# Tests: min_lines parameter
# ---------------------------------------------------------------------------


def test_min_lines_default_excludes_short_methods(tmp_path):
    """Default min_lines=4 skips 2-line methods — no pairs produced."""
    index_dir = _setup_index(tmp_path, _SHORT_PY)
    data = json.loads(analyze_project(index_dir=index_dir))
    assert data["similar_pairs"] == []


def test_min_lines_one_includes_short_methods(tmp_path):
    """min_lines=1 allows 2-line methods to be compared."""
    index_dir = _setup_index(tmp_path, _SHORT_PY)
    data = json.loads(analyze_project(index_dir=index_dir, min_lines=1))
    assert len(data["similar_pairs"]) > 0


def test_min_lines_at_threshold_produces_pairs(tmp_path):
    """Methods at exactly min_lines are included in comparisons."""
    index_dir = _setup_index(tmp_path, _IDENTICAL_PY)
    data = json.loads(analyze_project(index_dir=index_dir, min_lines=4))
    assert len(data["similar_pairs"]) > 0


def test_min_lines_above_threshold_excludes_all(tmp_path):
    """Setting min_lines higher than any method's line count produces no pairs."""
    index_dir = _setup_index(tmp_path, _IDENTICAL_PY)
    data = json.loads(analyze_project(index_dir=index_dir, min_lines=100))
    assert data["similar_pairs"] == []


def test_min_lines_does_not_affect_total_methods_count(tmp_path):
    """Filtered methods stay in the index — total_methods reflects the full index."""
    index_dir = _setup_index(tmp_path, _SHORT_PY)
    data = json.loads(analyze_project(index_dir=index_dir, min_lines=100))
    assert data["total_methods"] == 2
    assert data["similar_pairs"] == []


def test_min_lines_excludes_short_candidates(tmp_path):
    """Short methods are excluded as candidates, not just as queries."""
    content = """\
def process(a, b):
    total = a + b
    result = total * 2
    return result


def get_x(self):
    return self._x
"""
    index_dir = _setup_index(tmp_path, content)
    # 'process' is 4 lines (eligible); 'get_x' is 2 lines (excluded as candidate)
    data = json.loads(analyze_project(index_dir=index_dir, min_lines=4))
    # Only 'process' is eligible; no eligible candidates to compare with
    assert data["similar_pairs"] == []


def test_min_lines_default_is_four(tmp_path):
    """Omitting min_lines uses default of 4 — same as passing min_lines=4 explicitly."""
    index_dir = _setup_index(tmp_path, _IDENTICAL_PY)
    default_data = json.loads(analyze_project(index_dir=index_dir))
    explicit_data = json.loads(analyze_project(index_dir=index_dir, min_lines=4))
    assert default_data["similar_pairs"] == explicit_data["similar_pairs"]
