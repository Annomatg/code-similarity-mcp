"""Tests for SimilarityScorer."""

import pytest
from code_similarity_mcp.similarity.scorer import SimilarityScorer

_CODE_A = "func FUNC_NAME(v1, v2):\n return v1 + v2"
_CODE_B = "func FUNC_NAME(v1, v2):\n return v1 * v2"  # different logic


def _method_dict(**kwargs):
    defaults = {
        "id": 1,
        "file_path": "test.gd",
        "language": "gdscript",
        "name": "test_func",
        "parameters": ["a", "b"],
        "return_type": None,
        "normalized_code": _CODE_A,
        "code_hash": "abc123",
        "start_line": 1,
        "end_line": 3,
        "dependencies": [],
        "embedding_score": 0.95,
    }
    defaults.update(kwargs)
    return defaults


@pytest.fixture
def scorer():
    return SimilarityScorer(threshold=0.7)


def test_exact_match_scores_one(scorer):
    query = _method_dict(code_hash="HASH_X")
    cand = _method_dict(id=2, code_hash="HASH_X", embedding_score=1.0)
    results = scorer.score_candidates(query, [cand])
    assert len(results) == 1
    assert results[0].exact_match is True
    assert results[0].score == 1.0


def test_below_threshold_excluded(scorer):
    """Candidate with low embedding score AND different code is excluded."""
    query = _method_dict(code_hash="HASH_A", normalized_code=_CODE_A)
    # Use very different code for low AST score too
    cand = _method_dict(
        id=2,
        code_hash="HASH_B",
        normalized_code=_CODE_B,
        embedding_score=0.1,
    )
    results = scorer.score_candidates(query, [cand])
    assert len(results) == 0


def test_high_similarity_included(scorer):
    """Candidate with high scores across signals passes threshold."""
    query = _method_dict(code_hash="HASH_A", normalized_code=_CODE_A)
    cand = _method_dict(
        id=2,
        code_hash="HASH_B",
        normalized_code=_CODE_A,  # same normalized code -> AST = 1.0
        embedding_score=0.98,
    )
    results = scorer.score_candidates(query, [cand])
    assert len(results) == 1
    assert results[0].score > 0.7


def test_fast_filter_language_mismatch(scorer):
    query = _method_dict(language="gdscript")
    cand = _method_dict(id=2, language="python")
    results = scorer.score_candidates(query, [cand])
    assert len(results) == 0


def test_fast_filter_param_count_too_different(scorer):
    query = _method_dict(parameters=["a", "b", "c"])
    cand = _method_dict(id=2, parameters=["x"])  # diff = 2 > 1
    results = scorer.score_candidates(query, [cand])
    assert len(results) == 0


def test_fast_filter_param_count_off_by_one_passes(scorer):
    """Param count difference of 1 should still pass the filter."""
    query = _method_dict(code_hash="Q", parameters=["a", "b"], normalized_code=_CODE_A)
    cand = _method_dict(
        id=2,
        code_hash="C",
        parameters=["x"],  # diff = 1 -> passes filter
        normalized_code=_CODE_A,
        embedding_score=0.98,
    )
    results = scorer.score_candidates(query, [cand])
    # Filter passes, but param diff should show up in differences
    assert len(results) == 1
    assert any("parameter count" in d for d in results[0].differences)


def test_fast_filter_loc_ratio_too_small(scorer):
    query = _method_dict(start_line=1, end_line=30)
    cand = _method_dict(id=2, start_line=1, end_line=2)  # ratio 2/30 ≈ 0.07 < 0.7
    results = scorer.score_candidates(query, [cand])
    assert len(results) == 0


def test_results_sorted_by_score(scorer):
    query = _method_dict(code_hash="Q", normalized_code=_CODE_A)
    c1 = _method_dict(
        id=2, code_hash="C1", normalized_code=_CODE_A, embedding_score=0.80
    )
    c2 = _method_dict(
        id=3, code_hash="C2", normalized_code=_CODE_A, embedding_score=0.99
    )
    results = scorer.score_candidates(query, [c1, c2])
    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_return_type_difference_in_output(scorer):
    query = _method_dict(code_hash="Q", return_type="int", normalized_code=_CODE_A)
    cand = _method_dict(
        id=2,
        code_hash="C",
        return_type="float",
        normalized_code=_CODE_A,
        embedding_score=0.98,
    )
    results = scorer.score_candidates(query, [cand])
    assert len(results) == 1
    assert any("return type" in d for d in results[0].differences)


def test_dependency_difference_in_output(scorer):
    query = _method_dict(code_hash="Q", dependencies=["foo"], normalized_code=_CODE_A)
    cand = _method_dict(
        id=2,
        code_hash="C",
        dependencies=["bar"],
        normalized_code=_CODE_A,
        embedding_score=0.98,
    )
    results = scorer.score_candidates(query, [cand])
    assert len(results) == 1
    diffs = results[0].differences
    assert any("foo" in d for d in diffs)
    assert any("bar" in d for d in diffs)


def test_refactoring_hints_generated(scorer):
    query = _method_dict(code_hash="Q", parameters=["a"], normalized_code=_CODE_A)
    cand = _method_dict(
        id=2,
        code_hash="C",
        parameters=["a", "b"],
        normalized_code=_CODE_A,
        embedding_score=0.98,
    )
    results = scorer.score_candidates(query, [cand])
    assert len(results) == 1
    assert len(results[0].refactoring_hints) > 0


def test_empty_candidates(scorer):
    query = _method_dict()
    assert scorer.score_candidates(query, []) == []
