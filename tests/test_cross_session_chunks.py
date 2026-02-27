"""Integration tests for cross-session chunk reuse (feature #33).

Chunks stored in one MethodRegistry session must be fully available — via
both the registry API and MCP tools — in a new session that opens the same
index directory, without calling chunk_repository again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from code_similarity_mcp.index.registry import MethodRegistry
from code_similarity_mcp.mcp.server import analyze_chunks, chunk_repository, get_chunk_map, index_repository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_large_function(name: str = "big_func", n_stmts: int = 35) -> str:
    lines = [f"def {name}():"]
    for i in range(n_stmts):
        lines.append(f"    x_{i} = {i}")
    lines.append("    return x_0")
    return "\n".join(lines) + "\n"


def _setup_session1(tmp_path: Path, content: str, filename: str = "module.py") -> tuple[str, str, int]:
    """Index and chunk content, then close the registry.

    Returns (repo_dir, index_dir, function_id) so session-2 tests can refer
    to the stored function without opening session-1 again.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / filename).write_text(content, encoding="utf-8")
    index_dir = str(tmp_path / "index")

    index_repository(str(repo), index_dir=index_dir)
    chunk_repository(str(repo), index_dir=index_dir)

    # Grab the function id before closing.
    registry = MethodRegistry(index_dir)
    methods = registry.get_all_methods()
    func_name = content.split("(")[0].replace("def ", "").strip()
    method = next(m for m in methods if m["name"] == func_name)
    func_id = method["id"]
    chunk_count = registry.get_chunk_count()
    registry.close()

    return str(repo), index_dir, func_id


# ---------------------------------------------------------------------------
# Session-2 registry API tests
# ---------------------------------------------------------------------------


def test_new_registry_loads_chunk_count(tmp_path):
    """A freshly opened registry must report the same chunk count as session 1."""
    content = _make_large_function("worker", n_stmts=35)
    _, index_dir, func_id = _setup_session1(tmp_path, content)

    # Session 1 count
    r1 = MethodRegistry(index_dir)
    count1 = r1.get_chunk_count()
    r1.close()

    # Session 2 — brand-new instance, same index_dir
    r2 = MethodRegistry(index_dir)
    count2 = r2.get_chunk_count()
    r2.close()

    assert count1 > 0
    assert count2 == count1


def test_new_registry_get_chunks_by_function(tmp_path):
    """get_chunks_by_function must return data in a new session."""
    content = _make_large_function("process", n_stmts=35)
    _, index_dir, func_id = _setup_session1(tmp_path, content)

    r = MethodRegistry(index_dir)
    chunks = r.get_chunks_by_function(func_id)
    r.close()

    assert len(chunks) > 0, "Expected persisted chunks to be returned in a new session"


def test_new_registry_chunk_indices_sequential(tmp_path):
    """chunk_index values in the new session must be 0, 1, 2, …"""
    content = _make_large_function("seq_func", n_stmts=35)
    _, index_dir, func_id = _setup_session1(tmp_path, content)

    r = MethodRegistry(index_dir)
    chunks = sorted(r.get_chunks_by_function(func_id), key=lambda c: c["chunk_index"])
    r.close()

    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(indices)))


def test_new_registry_chunk_has_expected_fields(tmp_path):
    """Each persisted chunk must have the expected schema fields."""
    content = _make_large_function("schema_func", n_stmts=35)
    _, index_dir, func_id = _setup_session1(tmp_path, content)

    r = MethodRegistry(index_dir)
    chunks = r.get_chunks_by_function(func_id)
    r.close()

    required = {
        "id", "function_id", "chunk_index", "statement_start", "statement_end",
        "statement_indices", "function_name", "file_path",
        "depends_on_chunks", "depended_on_by_chunks",
        "normalized_code", "code_hash", "dependency_links", "faiss_pos",
    }
    for chunk in chunks:
        missing = required - chunk.keys()
        assert not missing, f"Chunk missing fields: {missing}"


def test_new_registry_faiss_index_loaded(tmp_path):
    """The chunk FAISS index must be loaded and non-empty in a new session."""
    content = _make_large_function("faiss_func", n_stmts=35)
    _, index_dir, _ = _setup_session1(tmp_path, content)

    r = MethodRegistry(index_dir)
    total = r._chunks_index.ntotal
    r.close()

    assert total > 0


def test_new_registry_chunk_id_map_loaded(tmp_path):
    """The chunk id_map must map every FAISS position to a valid DB id."""
    content = _make_large_function("idmap_func", n_stmts=35)
    _, index_dir, func_id = _setup_session1(tmp_path, content)

    r = MethodRegistry(index_dir)
    id_map = r._chunk_id_map
    all_chunk_ids = {c["id"] for c in r.get_chunks_by_function(func_id)}
    r.close()

    # Every value in the id_map must correspond to an actual DB row
    assert len(id_map) > 0
    for faiss_pos, db_id in id_map.items():
        assert db_id in all_chunk_ids, (
            f"id_map position {faiss_pos} points to unknown chunk DB id {db_id}"
        )


def test_new_registry_search_chunks_returns_results(tmp_path):
    """search_chunks must return results using the persisted FAISS index."""
    import numpy as np
    from code_similarity_mcp.embeddings.generator import EmbeddingGenerator

    content = _make_large_function("search_func", n_stmts=35)
    _, index_dir, func_id = _setup_session1(tmp_path, content)

    # Build a query embedding for a simple code snippet
    gen = EmbeddingGenerator()
    snippet = "x_0 = 0\nx_1 = 1\nx_2 = 2"
    query_emb = gen.encode([snippet])[0]

    r = MethodRegistry(index_dir)
    results = r.search_chunks(query_emb, top_k=3)
    r.close()

    assert len(results) > 0, "search_chunks should return results from persisted index"


def test_new_registry_get_chunk_by_id(tmp_path):
    """get_chunk_by_id must retrieve a chunk stored in a previous session."""
    content = _make_large_function("byid_func", n_stmts=35)
    _, index_dir, func_id = _setup_session1(tmp_path, content)

    # Get a chunk id from a fresh registry (session 2)
    r2 = MethodRegistry(index_dir)
    chunks = r2.get_chunks_by_function(func_id)
    chunk_id = chunks[0]["id"]

    # Retrieve it by id in the same session-2 instance
    chunk = r2.get_chunk_by_id(chunk_id)
    r2.close()

    assert chunk is not None
    assert chunk["id"] == chunk_id


def test_new_registry_get_chunk_embedding(tmp_path):
    """get_chunk_embedding must reconstruct a vector from the persisted FAISS index."""
    import numpy as np

    content = _make_large_function("emb_func", n_stmts=35)
    _, index_dir, func_id = _setup_session1(tmp_path, content)

    r = MethodRegistry(index_dir)
    chunks = r.get_chunks_by_function(func_id)
    faiss_pos = chunks[0]["faiss_pos"]
    emb = r.get_chunk_embedding(faiss_pos)
    r.close()

    assert emb is not None
    assert emb.shape == (384,)
    # L2-normalised vectors have unit norm
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Session-2 MCP tool tests
# ---------------------------------------------------------------------------


def test_get_chunk_map_new_session_by_function_id(tmp_path):
    """get_chunk_map must return stored chunks in a new session (function_id query)."""
    content = _make_large_function("map_func", n_stmts=35)
    _, index_dir, func_id = _setup_session1(tmp_path, content)

    # Call the MCP tool without re-running chunk_repository
    data = json.loads(get_chunk_map(function_id=func_id, index_dir=index_dir))
    assert "functions" in data
    assert len(data["functions"]) == 1
    assert len(data["functions"][0]["chunks"]) > 0


def test_get_chunk_map_new_session_by_file_path(tmp_path):
    """get_chunk_map must return stored chunks in a new session (file_path query)."""
    content = _make_large_function("fpath_func", n_stmts=35)
    _, index_dir, func_id = _setup_session1(tmp_path, content)

    # Retrieve file path from registry in a new session
    r = MethodRegistry(index_dir)
    methods = r.get_all_methods()
    file_path = next(m for m in methods if m["name"] == "fpath_func")["file_path"]
    r.close()

    data = json.loads(get_chunk_map(file_path=file_path, index_dir=index_dir))
    assert "functions" in data
    assert len(data["functions"]) >= 1
    names = {f["function_name"] for f in data["functions"]}
    assert "fpath_func" in names


def test_get_chunk_map_chunk_count_matches_registry(tmp_path):
    """Number of chunks from get_chunk_map equals get_chunk_count in the new session."""
    content = _make_large_function("count_func", n_stmts=35)
    _, index_dir, func_id = _setup_session1(tmp_path, content)

    # MCP tool call (new session internally)
    data = json.loads(get_chunk_map(function_id=func_id, index_dir=index_dir))
    map_count = len(data["functions"][0]["chunks"])

    # Direct registry API (another new session)
    r = MethodRegistry(index_dir)
    reg_count = len(r.get_chunks_by_function(func_id))
    r.close()

    assert map_count == reg_count
    assert map_count > 0


def test_analyze_chunks_new_session_returns_results(tmp_path):
    """analyze_chunks must find persisted chunks in a new session."""
    content = _make_large_function("analyze_func", n_stmts=35)
    _, index_dir, _ = _setup_session1(tmp_path, content)

    # Query without re-running chunk_repository
    data = json.loads(
        analyze_chunks(code_snippet="x_0 = 0\nx_1 = 1\nx_2 = 2", index_dir=index_dir)
    )
    assert "results" in data
    assert len(data["results"]) > 0


def test_analyze_chunks_new_session_result_fields(tmp_path):
    """Results from analyze_chunks in a new session have all required fields."""
    content = _make_large_function("fields_func", n_stmts=35)
    _, index_dir, _ = _setup_session1(tmp_path, content)

    data = json.loads(
        analyze_chunks(code_snippet="x_0 = 0\nx_1 = 1\nx_2 = 2", index_dir=index_dir)
    )
    required = {"chunk_id", "function_name", "file", "similarity_score", "statement_range", "refactoring_hint"}
    for item in data["results"]:
        for field in required:
            assert field in item, f"Missing field {field!r}"


def test_analyze_chunks_new_session_scores_in_range(tmp_path):
    """Similarity scores from a new session must be in [0, 1]."""
    content = _make_large_function("score_func", n_stmts=35)
    _, index_dir, _ = _setup_session1(tmp_path, content)

    data = json.loads(
        analyze_chunks(code_snippet="x_0 = 0\nx_1 = 1\nx_2 = 2", index_dir=index_dir)
    )
    for item in data["results"]:
        assert 0.0 <= item["similarity_score"] <= 1.0


def test_analyze_chunks_new_session_chunk_id_query(tmp_path):
    """analyze_chunks with chunk_id works in a new session (no re-chunking required)."""
    content = _make_large_function("cid_func", n_stmts=40)
    _, index_dir, func_id = _setup_session1(tmp_path, content)

    # Get a chunk id from a fresh registry instance
    r = MethodRegistry(index_dir)
    chunks = sorted(r.get_chunks_by_function(func_id), key=lambda c: c["chunk_index"])
    r.close()

    if len(chunks) < 2:
        pytest.skip("Need at least 2 chunks")

    cid = chunks[0]["id"]
    data = json.loads(analyze_chunks(chunk_id=cid, index_dir=index_dir))
    assert "results" in data
    # The query chunk itself must not appear in the results
    result_ids = [item["chunk_id"] for item in data["results"]]
    assert cid not in result_ids


def test_no_rechunking_needed_in_new_session(tmp_path):
    """Calling chunk_repository again in session 2 must produce 0 new chunks (idempotent)."""
    content = _make_large_function("idem_func", n_stmts=35)
    repo, index_dir, _ = _setup_session1(tmp_path, content)

    # Session 1 chunk count
    r1 = MethodRegistry(index_dir)
    count1 = r1.get_chunk_count()
    r1.close()

    # Re-run chunk_repository (should be a no-op because chunks already exist)
    result = chunk_repository(repo, index_dir=index_dir)
    result_data = json.loads(result) if isinstance(result, str) else result

    # Session 2 chunk count must be unchanged
    r2 = MethodRegistry(index_dir)
    count2 = r2.get_chunk_count()
    r2.close()

    assert count2 == count1
    assert result_data.get("chunks_created", 0) == 0


def test_multiple_sessions_see_same_data(tmp_path):
    """Three successive MethodRegistry instances must all report identical chunk counts."""
    content = _make_large_function("triple_func", n_stmts=35)
    _, index_dir, func_id = _setup_session1(tmp_path, content)

    counts = []
    for _ in range(3):
        r = MethodRegistry(index_dir)
        counts.append(r.get_chunk_count())
        r.close()

    assert counts[0] > 0
    assert counts[0] == counts[1] == counts[2]
