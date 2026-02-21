"""Tests for FilterPipeline and related MethodRegistry filter support."""

from __future__ import annotations

import numpy as np
import pytest

from code_similarity_mcp.index.registry import MethodRegistry
from code_similarity_mcp.parser.base import MethodInfo
from code_similarity_mcp.similarity.filter import FilterPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cand(
    language="python",
    params=None,
    start_line=1,
    end_line=5,
    db_id=1,
) -> dict:
    return {
        "id": db_id,
        "language": language,
        "parameters": params if params is not None else ["a", "b"],
        "start_line": start_line,
        "end_line": end_line,
    }


def _query(
    language="python",
    params=None,
    start_line=1,
    end_line=5,
) -> dict:
    return {
        "language": language,
        "parameters": params if params is not None else ["a", "b"],
        "start_line": start_line,
        "end_line": end_line,
    }


def _make_method(
    name="func",
    file_path="test.py",
    language="python",
    params=None,
    start_line=1,
    end_line=5,
) -> MethodInfo:
    params = params or ["a", "b"]
    return MethodInfo(
        file_path=file_path,
        language=language,
        name=name,
        parameters=params,
        return_type=None,
        body_code=f"def {name}({', '.join(params)}):\n    return 1",
        normalized_code=f"def FUNC_NAME({', '.join(params)}):\n    return 1",
        start_line=start_line,
        end_line=end_line,
        dependencies=[],
    )


def _random_embedding(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.random(384).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


@pytest.fixture
def pipeline() -> FilterPipeline:
    return FilterPipeline()


@pytest.fixture
def registry(tmp_path):
    reg = MethodRegistry(tmp_path / "index")
    yield reg
    reg.close()


# ---------------------------------------------------------------------------
# Individual predicate: language_matches
# ---------------------------------------------------------------------------


def test_language_matches_same(pipeline):
    assert pipeline.language_matches("python", _cand(language="python")) is True


def test_language_matches_different(pipeline):
    assert pipeline.language_matches("python", _cand(language="gdscript")) is False


def test_language_matches_case_sensitive(pipeline):
    assert pipeline.language_matches("Python", _cand(language="python")) is False


# ---------------------------------------------------------------------------
# Individual predicate: param_count_within_one
# ---------------------------------------------------------------------------


def test_param_count_exact_match(pipeline):
    assert pipeline.param_count_within_one(2, _cand(params=["a", "b"])) is True


def test_param_count_off_by_one_above(pipeline):
    assert pipeline.param_count_within_one(2, _cand(params=["a", "b", "c"])) is True


def test_param_count_off_by_one_below(pipeline):
    assert pipeline.param_count_within_one(2, _cand(params=["a"])) is True


def test_param_count_off_by_two_fails(pipeline):
    assert pipeline.param_count_within_one(2, _cand(params=["a", "b", "c", "d"])) is False


def test_param_count_zero_vs_one(pipeline):
    assert pipeline.param_count_within_one(0, _cand(params=["a"])) is True


def test_param_count_zero_vs_two_fails(pipeline):
    assert pipeline.param_count_within_one(0, _cand(params=["a", "b"])) is False


# ---------------------------------------------------------------------------
# Individual predicate: loc_within_range
# ---------------------------------------------------------------------------


def test_loc_exact_match(pipeline):
    assert pipeline.loc_within_range(10, _cand(start_line=1, end_line=10)) is True


def test_loc_ratio_exactly_at_boundary(pipeline):
    # query=10, cand=7 → 7/10 = 0.7 → passes
    assert pipeline.loc_within_range(10, _cand(start_line=1, end_line=7)) is True


def test_loc_ratio_just_below_boundary(pipeline):
    # query=10, cand=6 → 6/10 = 0.6 < 0.7 → fails
    assert pipeline.loc_within_range(10, _cand(start_line=1, end_line=6)) is False


def test_loc_ratio_larger_candidate(pipeline):
    # query=7, cand=10 → 7/10 = 0.7 → passes
    assert pipeline.loc_within_range(7, _cand(start_line=1, end_line=10)) is True


def test_loc_query_zero_always_passes(pipeline):
    # query_loc=0 → always True
    assert pipeline.loc_within_range(0, _cand(start_line=5, end_line=5)) is True


def test_loc_candidate_zero_always_passes(pipeline):
    assert pipeline.loc_within_range(10, _cand(start_line=3, end_line=2)) is True


# ---------------------------------------------------------------------------
# Combined: passes
# ---------------------------------------------------------------------------


def test_passes_all_conditions_met(pipeline):
    q = _query(language="python", params=["a", "b"], start_line=1, end_line=10)
    c = _cand(language="python", params=["a", "b"], start_line=1, end_line=10)
    assert pipeline.passes(q, c) is True


def test_passes_fails_on_language(pipeline):
    q = _query(language="python")
    c = _cand(language="gdscript")
    assert pipeline.passes(q, c) is False


def test_passes_fails_on_param_count(pipeline):
    q = _query(params=["a", "b"])
    c = _cand(params=["x", "y", "z", "w"])  # diff = 2
    assert pipeline.passes(q, c) is False


def test_passes_fails_on_loc(pipeline):
    q = _query(start_line=1, end_line=30)
    c = _cand(start_line=1, end_line=2)  # ratio 2/30 ≈ 0.067
    assert pipeline.passes(q, c) is False


def test_passes_off_by_one_param_passes(pipeline):
    q = _query(params=["a", "b"], start_line=1, end_line=10)
    c = _cand(params=["x"], start_line=1, end_line=10)  # diff = 1 → passes
    assert pipeline.passes(q, c) is True


# ---------------------------------------------------------------------------
# filter_candidates
# ---------------------------------------------------------------------------


def test_filter_candidates_removes_mismatches(pipeline):
    q = _query(language="python", params=["a"], start_line=1, end_line=5)
    good = _cand(language="python", params=["a"], start_line=1, end_line=5, db_id=1)
    bad_lang = _cand(language="gdscript", params=["a"], start_line=1, end_line=5, db_id=2)
    bad_loc = _cand(language="python", params=["a"], start_line=1, end_line=100, db_id=3)
    result = pipeline.filter_candidates(q, [good, bad_lang, bad_loc])
    assert len(result) == 1
    assert result[0]["id"] == 1


def test_filter_candidates_empty_list(pipeline):
    assert pipeline.filter_candidates(_query(), []) == []


def test_filter_candidates_all_pass(pipeline):
    q = _query()
    candidates = [_cand(db_id=i) for i in range(5)]
    result = pipeline.filter_candidates(q, candidates)
    assert len(result) == 5


# ---------------------------------------------------------------------------
# MethodRegistry.filter_by_criteria
# ---------------------------------------------------------------------------


def test_filter_by_criteria_language(registry):
    registry.add_method(_make_method("py_func", language="python"), _random_embedding(1))
    registry.add_method(_make_method("gd_func", language="gdscript"), _random_embedding(2))

    ids = registry.filter_by_criteria("python", param_count=2, loc=5)
    assert len(ids) == 1
    results = registry.get_by_file("test.py")
    assert results[0]["id"] in ids


def test_filter_by_criteria_param_count_exact(registry):
    registry.add_method(_make_method("f1", params=["a", "b"]), _random_embedding(1))
    registry.add_method(_make_method("f2", params=["a", "b", "c", "d"]), _random_embedding(2))

    # param_count=2, diff allowed ≤1 → f1 (0 diff) passes, f2 (2 diff) fails
    ids = registry.filter_by_criteria("python", param_count=2, loc=5)
    assert len(ids) == 1


def test_filter_by_criteria_param_count_off_by_one(registry):
    registry.add_method(_make_method("f1", params=["a"]), _random_embedding(1))
    registry.add_method(_make_method("f2", params=["a", "b"]), _random_embedding(2))
    registry.add_method(_make_method("f3", params=["a", "b", "c"]), _random_embedding(3))
    registry.add_method(_make_method("f4", params=["a", "b", "c", "d"]), _random_embedding(4))

    # param_count=2, allowed: 1, 2, 3 → f1, f2, f3 pass; f4 (diff=2) fails
    ids = registry.filter_by_criteria("python", param_count=2, loc=5)
    assert len(ids) == 3


def test_filter_by_criteria_loc_range(registry):
    # 5-line method (LOC=5)
    registry.add_method(_make_method("short", start_line=1, end_line=5), _random_embedding(1))
    # 50-line method (LOC=50) — ratio 5/50=0.1 < 0.7 → excluded
    registry.add_method(_make_method("long", start_line=1, end_line=50), _random_embedding(2))

    ids = registry.filter_by_criteria("python", param_count=2, loc=5)
    assert len(ids) == 1  # only 'short' passes


def test_filter_by_criteria_loc_zero_includes_all_languages(registry):
    registry.add_method(_make_method("f1", language="python"), _random_embedding(1))
    registry.add_method(_make_method("f2", language="python"), _random_embedding(2))

    ids = registry.filter_by_criteria("python", param_count=2, loc=0)
    assert len(ids) == 2


def test_filter_by_criteria_empty_registry(registry):
    ids = registry.filter_by_criteria("python", param_count=2, loc=5)
    assert ids == set()


# ---------------------------------------------------------------------------
# MethodRegistry.search with allowed_ids
# ---------------------------------------------------------------------------


def test_search_with_allowed_ids_restricts_results(registry):
    emb1 = _random_embedding(1)
    id1 = registry.add_method(_make_method("f1"), emb1)
    _id2 = registry.add_method(_make_method("f2"), _random_embedding(2))

    # Only allow id1
    results = registry.search(emb1, top_k=5, allowed_ids={id1})
    assert len(results) == 1
    assert results[0]["name"] == "f1"


def test_search_with_empty_allowed_ids_returns_nothing(registry):
    registry.add_method(_make_method("f1"), _random_embedding(1))
    results = registry.search(_random_embedding(1), top_k=5, allowed_ids=set())
    assert results == []


def test_search_with_none_allowed_ids_is_unfiltered(registry):
    registry.add_method(_make_method("f1"), _random_embedding(1))
    registry.add_method(_make_method("f2"), _random_embedding(2))
    results = registry.search(_random_embedding(1), top_k=5, allowed_ids=None)
    assert len(results) == 2


def test_search_allowed_ids_excludes_non_matching(registry):
    id1 = registry.add_method(_make_method("f1"), _random_embedding(1))
    id2 = registry.add_method(_make_method("f2"), _random_embedding(2))
    id3 = registry.add_method(_make_method("f3"), _random_embedding(3))

    results = registry.search(_random_embedding(0), top_k=5, allowed_ids={id1, id3})
    names = {r["name"] for r in results}
    assert "f1" in names
    assert "f3" in names
    assert "f2" not in names


# ---------------------------------------------------------------------------
# FilterPipeline.get_candidate_ids (integration)
# ---------------------------------------------------------------------------


def test_get_candidate_ids_returns_matching_ids(registry):
    pipeline = FilterPipeline()
    id1 = registry.add_method(
        _make_method("match", language="python", params=["a", "b"], start_line=1, end_line=5),
        _random_embedding(1),
    )
    _id2 = registry.add_method(
        _make_method("nomatch", language="gdscript", params=["a", "b"], start_line=1, end_line=5),
        _random_embedding(2),
    )

    query = _query(language="python", params=["a", "b"], start_line=1, end_line=5)
    ids = pipeline.get_candidate_ids(registry, query)
    assert id1 in ids
    assert _id2 not in ids


def test_get_candidate_ids_empty_index(registry):
    pipeline = FilterPipeline()
    query = _query()
    ids = pipeline.get_candidate_ids(registry, query)
    assert ids == set()


def test_get_candidate_ids_combined_with_search(registry):
    """Full pipeline: filter_by_criteria → search with allowed_ids."""
    pipeline = FilterPipeline()

    id_py = registry.add_method(
        _make_method("py_func", language="python", params=["x"], start_line=1, end_line=4),
        _random_embedding(10),
    )
    _id_gd = registry.add_method(
        _make_method("gd_func", language="gdscript", params=["x"], start_line=1, end_line=4),
        _random_embedding(11),
    )

    query = _query(language="python", params=["x"], start_line=1, end_line=4)
    valid_ids = pipeline.get_candidate_ids(registry, query)
    results = registry.search(_random_embedding(10), top_k=5, allowed_ids=valid_ids)

    names = [r["name"] for r in results]
    assert "py_func" in names
    assert "gd_func" not in names
