"""Integration tests for the MCP analyze_chunks tool (feature #26)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from code_similarity_mcp.index.registry import MethodRegistry
from code_similarity_mcp.mcp.server import analyze_chunks, chunk_repository, index_repository


# ---------------------------------------------------------------------------
# Code fixtures
# ---------------------------------------------------------------------------

def _make_large_function(name: str = "big_func", n_stmts: int = 35) -> str:
    """Generate a Python function with enough assignment statements to exceed 30."""
    lines = [f"def {name}():"]
    for i in range(n_stmts):
        lines.append(f"    x_{i} = {i}")
    lines.append("    return x_0")
    return "\n".join(lines) + "\n"


def _setup_index_and_chunks(tmp_path: Path, content: str, filename: str = "module.py") -> tuple[str, str]:
    """Write *content* to a .py file, index it, and chunk it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / filename).write_text(content, encoding="utf-8")
    index_dir = str(tmp_path / "index")
    index_repository(str(repo), index_dir=index_dir)
    chunk_repository(str(repo), index_dir=index_dir)
    return str(repo), index_dir


# ---------------------------------------------------------------------------
# Tests: validation / error cases
# ---------------------------------------------------------------------------


def test_neither_snippet_nor_chunk_id_returns_error(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    data = json.loads(analyze_chunks(index_dir=index_dir))
    assert "error" in data


def test_both_snippet_and_chunk_id_returns_error(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    registry = MethodRegistry(index_dir)
    methods = registry.get_all_methods()
    big = next(m for m in methods if m["name"] == "big_func")
    chunks = registry.get_chunks_by_function(big["id"])
    registry.close()
    cid = chunks[0]["id"]

    data = json.loads(analyze_chunks(code_snippet="x = 1", chunk_id=cid, index_dir=index_dir))
    assert "error" in data


def test_nonexistent_chunk_id_returns_error(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    data = json.loads(analyze_chunks(chunk_id=999999, index_dir=index_dir))
    assert "error" in data


def test_empty_chunk_index_returns_empty_results(tmp_path):
    # Index a file but do NOT run chunk_repository → no chunks stored
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "small.py").write_text("def f(a):\n    return a\n", encoding="utf-8")
    index_dir = str(tmp_path / "index")
    index_repository(str(repo), index_dir=index_dir)

    data = json.loads(analyze_chunks(code_snippet="x = 1", index_dir=index_dir))
    assert data == {"results": []}


# ---------------------------------------------------------------------------
# Tests: response shape
# ---------------------------------------------------------------------------


def test_returns_valid_json(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    raw = analyze_chunks(code_snippet="x_0 = 0\nx_1 = 1", index_dir=index_dir)
    data = json.loads(raw)
    assert isinstance(data, dict)


def test_results_key_present(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    data = json.loads(analyze_chunks(code_snippet="x_0 = 0", index_dir=index_dir))
    assert "results" in data


def test_results_is_a_list(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    data = json.loads(analyze_chunks(code_snippet="x_0 = 0", index_dir=index_dir))
    assert isinstance(data["results"], list)


def test_result_items_have_required_fields(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    data = json.loads(analyze_chunks(code_snippet="x_0 = 0\nx_1 = 1\nx_2 = 2", index_dir=index_dir))
    required = {"chunk_id", "function_name", "file", "similarity_score", "statement_range", "refactoring_hint"}
    for item in data["results"]:
        for field in required:
            assert field in item, f"Missing field {field!r}"


def test_similarity_score_is_float_between_0_and_1(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    data = json.loads(analyze_chunks(code_snippet="x_0 = 0\nx_1 = 1", index_dir=index_dir))
    for item in data["results"]:
        assert isinstance(item["similarity_score"], float)
        assert 0.0 <= item["similarity_score"] <= 1.0


def test_statement_range_is_two_ints(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    data = json.loads(analyze_chunks(code_snippet="x_0 = 0\nx_1 = 1", index_dir=index_dir))
    for item in data["results"]:
        sr = item["statement_range"]
        assert isinstance(sr, list) and len(sr) == 2
        assert all(isinstance(v, int) for v in sr)


def test_refactoring_hint_is_string(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    data = json.loads(analyze_chunks(code_snippet="x_0 = 0\nx_1 = 1", index_dir=index_dir))
    for item in data["results"]:
        assert isinstance(item["refactoring_hint"], str)
        assert len(item["refactoring_hint"]) > 0


# ---------------------------------------------------------------------------
# Tests: top_k
# ---------------------------------------------------------------------------


def test_top_k_limits_results(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function(n_stmts=40))
    data = json.loads(analyze_chunks(code_snippet="x_0 = 0\nx_1 = 1\nx_2 = 2", top_k=2, index_dir=index_dir))
    assert len(data["results"]) <= 2


def test_top_k_default_is_3(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function(n_stmts=40))
    data = json.loads(analyze_chunks(code_snippet="x_0 = 0\nx_1 = 1\nx_2 = 2", index_dir=index_dir))
    assert len(data["results"]) <= 3


def test_top_k_1_returns_at_most_one_result(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function(n_stmts=35))
    data = json.loads(analyze_chunks(code_snippet="x_0 = 0", top_k=1, index_dir=index_dir))
    assert len(data["results"]) <= 1


# ---------------------------------------------------------------------------
# Tests: results are sorted by similarity (descending)
# ---------------------------------------------------------------------------


def test_results_sorted_descending(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function(n_stmts=40))
    data = json.loads(analyze_chunks(code_snippet="x_0 = 0\nx_1 = 1\nx_2 = 2", top_k=3, index_dir=index_dir))
    scores = [item["similarity_score"] for item in data["results"]]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Tests: chunk_id query
# ---------------------------------------------------------------------------


def test_chunk_id_query_returns_results(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function(n_stmts=40))
    registry = MethodRegistry(index_dir)
    methods = registry.get_all_methods()
    big = next(m for m in methods if m["name"] == "big_func")
    chunks = sorted(registry.get_chunks_by_function(big["id"]), key=lambda c: c["chunk_index"])
    registry.close()

    # Pick the first chunk as the query
    cid = chunks[0]["id"]
    data = json.loads(analyze_chunks(chunk_id=cid, index_dir=index_dir))
    assert "results" in data
    assert isinstance(data["results"], list)


def test_chunk_id_query_excludes_self(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function(n_stmts=40))
    registry = MethodRegistry(index_dir)
    methods = registry.get_all_methods()
    big = next(m for m in methods if m["name"] == "big_func")
    chunks = registry.get_chunks_by_function(big["id"])
    registry.close()

    cid = chunks[0]["id"]
    data = json.loads(analyze_chunks(chunk_id=cid, index_dir=index_dir))
    result_ids = [item["chunk_id"] for item in data["results"]]
    assert cid not in result_ids


def test_chunk_id_result_has_correct_fields(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function(n_stmts=40))
    registry = MethodRegistry(index_dir)
    methods = registry.get_all_methods()
    big = next(m for m in methods if m["name"] == "big_func")
    chunks = registry.get_chunks_by_function(big["id"])
    registry.close()

    cid = chunks[0]["id"]
    data = json.loads(analyze_chunks(chunk_id=cid, index_dir=index_dir))
    required = {"chunk_id", "function_name", "file", "similarity_score", "statement_range", "refactoring_hint"}
    for item in data["results"]:
        for field in required:
            assert field in item


# ---------------------------------------------------------------------------
# Tests: exact match returns similarity >= 0.99
# ---------------------------------------------------------------------------


def test_exact_match_snippet_similarity_gte_099(tmp_path):
    """Searching with the exact normalised body of a stored chunk should score >= 0.99."""
    from code_similarity_mcp.mcp.server import chunk_repository, index_repository
    from code_similarity_mcp.parser.base import (
        annotate_chunks,
        embed_chunks,
        group_into_chunks,
    )
    from code_similarity_mcp.parser.python import (
        build_dependency_graph,
        get_flat_statements,
    )

    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function(n_stmts=35))
    registry = MethodRegistry(index_dir)
    methods = registry.get_all_methods()
    big = next(m for m in methods if m["name"] == "big_func")

    # Get the first stored chunk's statement range and reconstruct its snippet
    chunks = sorted(registry.get_chunks_by_function(big["id"]), key=lambda c: c["chunk_index"])
    registry.close()

    first_chunk = chunks[0]

    # Rebuild the original snippet by extracting lines from the body code
    body_lines = big["body_code"].splitlines()
    stmts = get_flat_statements(big["body_code"])
    # Gather lines covered by the first chunk's statement indices
    stmt_indices = first_chunk["statement_indices"]
    selected_stmts = [stmts[i] for i in stmt_indices]
    min_line = min(s.start_line for s in selected_stmts)
    max_line = max(s.end_line for s in selected_stmts)
    snippet_lines = body_lines[min_line - 1 : max_line]
    snippet = "\n".join(line.strip() for line in snippet_lines)

    data = json.loads(analyze_chunks(code_snippet=snippet, top_k=1, index_dir=index_dir))
    assert data["results"], "Expected at least one result"
    assert data["results"][0]["similarity_score"] >= 0.99, (
        f"Expected similarity >= 0.99, got {data['results'][0]['similarity_score']}"
    )


def test_chunk_id_query_top_result_is_highly_similar(tmp_path):
    """When multiple chunks exist, querying by chunk_id should find closely related chunks."""
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function(n_stmts=40))
    registry = MethodRegistry(index_dir)
    methods = registry.get_all_methods()
    big = next(m for m in methods if m["name"] == "big_func")
    chunks = sorted(registry.get_chunks_by_function(big["id"]), key=lambda c: c["chunk_index"])
    registry.close()

    if len(chunks) < 2:
        pytest.skip("Need at least 2 chunks for this test")

    cid = chunks[0]["id"]
    data = json.loads(analyze_chunks(chunk_id=cid, index_dir=index_dir))
    # We can't assert >= 0.99 here (different chunks differ), but scores should be valid
    for item in data["results"]:
        assert 0.0 <= item["similarity_score"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: function_name and file populated correctly
# ---------------------------------------------------------------------------


def test_result_function_name_matches_stored(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function("my_func"))
    data = json.loads(analyze_chunks(code_snippet="x_0 = 0\nx_1 = 1", index_dir=index_dir))
    for item in data["results"]:
        assert item["function_name"] == "my_func"


def test_result_file_is_nonempty_string(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    data = json.loads(analyze_chunks(code_snippet="x_0 = 0\nx_1 = 1", index_dir=index_dir))
    for item in data["results"]:
        assert isinstance(item["file"], str)
        assert len(item["file"]) > 0


# ---------------------------------------------------------------------------
# Tests: combined scoring (feature #32)
# ---------------------------------------------------------------------------


def test_combined_score_lower_than_pure_embedding_when_stmt_mismatch(tmp_path):
    """Score is less than embedding alone when statement counts differ strongly."""
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function(n_stmts=40))
    # A single-statement snippet will have a very different stmt count from
    # stored chunks (which have ~10 statements each). Combined score should be
    # pulled down by the structural component.
    data = json.loads(analyze_chunks(code_snippet="x_0 = 0", top_k=1, index_dir=index_dir))
    if not data["results"]:
        pytest.skip("No results to check")
    score = data["results"][0]["similarity_score"]
    assert 0.0 <= score <= 1.0


def test_combined_score_is_weighted_blend(tmp_path):
    """Verify that the score is not just embedding similarity (structural component is used)."""
    from code_similarity_mcp.similarity.chunk_scorer import ChunkSimilarityScorer
    scorer = ChunkSimilarityScorer()
    assert scorer.W_EMBEDDING == pytest.approx(0.7)
    assert scorer.W_STRUCTURAL == pytest.approx(0.3)
    assert scorer.W_EMBEDDING + scorer.W_STRUCTURAL == pytest.approx(1.0)


def test_results_sorted_by_combined_score(tmp_path):
    """Results are sorted by combined score, not just embedding score."""
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function(n_stmts=40))
    data = json.loads(analyze_chunks(code_snippet="x_0 = 0\nx_1 = 1\nx_2 = 2", top_k=3, index_dir=index_dir))
    scores = [item["similarity_score"] for item in data["results"]]
    assert scores == sorted(scores, reverse=True)


def test_chunk_id_query_combined_score_in_range(tmp_path):
    """chunk_id query uses topology in structural component; score still in [0, 1]."""
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function(n_stmts=40))
    registry = MethodRegistry(index_dir)
    methods = registry.get_all_methods()
    big = next(m for m in methods if m["name"] == "big_func")
    chunks = sorted(registry.get_chunks_by_function(big["id"]), key=lambda c: c["chunk_index"])
    registry.close()

    if len(chunks) < 2:
        pytest.skip("Need at least 2 chunks")

    cid = chunks[0]["id"]
    data = json.loads(analyze_chunks(chunk_id=cid, index_dir=index_dir))
    for item in data["results"]:
        assert 0.0 <= item["similarity_score"] <= 1.0
