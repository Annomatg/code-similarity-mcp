"""Integration tests for the MCP chunk_repository tool (feature #25)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from code_similarity_mcp.index.registry import MethodRegistry
from code_similarity_mcp.mcp.server import chunk_repository, index_repository


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


def _make_small_function(name: str = "small_func") -> str:
    return f"def {name}(a, b):\n    x = a + b\n    return x\n"


def _setup_index(tmp_path: Path, content: str, filename: str = "module.py") -> tuple[str, str]:
    """Write *content* to a .py file and index it. Returns (repo_dir, index_dir)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / filename).write_text(content, encoding="utf-8")
    index_dir = str(tmp_path / "index")
    index_repository(str(repo), index_dir=index_dir)
    return str(repo), index_dir


# ---------------------------------------------------------------------------
# Tests: response shape
# ---------------------------------------------------------------------------


def test_returns_valid_json(tmp_path):
    repo, index_dir = _setup_index(tmp_path, _make_small_function())
    raw = chunk_repository(repo, index_dir=index_dir)
    data = json.loads(raw)
    assert isinstance(data, dict)


def test_response_has_required_keys(tmp_path):
    repo, index_dir = _setup_index(tmp_path, _make_small_function())
    data = json.loads(chunk_repository(repo, index_dir=index_dir))
    for key in ("files_scanned", "functions_chunked", "chunks_created"):
        assert key in data, f"Missing key: {key!r}"


def test_all_counts_are_non_negative_integers(tmp_path):
    repo, index_dir = _setup_index(tmp_path, _make_small_function())
    data = json.loads(chunk_repository(repo, index_dir=index_dir))
    for key in ("files_scanned", "functions_chunked", "chunks_created"):
        assert isinstance(data[key], int)
        assert data[key] >= 0


# ---------------------------------------------------------------------------
# Tests: invalid repository_root
# ---------------------------------------------------------------------------


def test_nonexistent_root_returns_error(tmp_path):
    data = json.loads(
        chunk_repository(str(tmp_path / "does_not_exist"), index_dir=str(tmp_path / "idx"))
    )
    assert "error" in data


# ---------------------------------------------------------------------------
# Tests: filtering — only large functions are chunked
# ---------------------------------------------------------------------------


def test_small_function_not_chunked(tmp_path):
    repo, index_dir = _setup_index(tmp_path, _make_small_function())
    data = json.loads(chunk_repository(repo, index_dir=index_dir))
    assert data["functions_chunked"] == 0
    assert data["chunks_created"] == 0


def test_large_function_is_chunked(tmp_path):
    repo, index_dir = _setup_index(tmp_path, _make_large_function(n_stmts=35))
    data = json.loads(chunk_repository(repo, index_dir=index_dir))
    assert data["functions_chunked"] == 1
    assert data["chunks_created"] >= 1


def test_mixed_file_only_large_function_chunked(tmp_path):
    content = _make_small_function("tiny") + "\n\n" + _make_large_function("giant")
    repo, index_dir = _setup_index(tmp_path, content)
    data = json.loads(chunk_repository(repo, index_dir=index_dir))
    assert data["functions_chunked"] == 1


# ---------------------------------------------------------------------------
# Tests: files_scanned
# ---------------------------------------------------------------------------


def test_files_scanned_counts_unique_indexed_files(tmp_path):
    repo, index_dir = _setup_index(tmp_path, _make_small_function())
    data = json.loads(chunk_repository(repo, index_dir=index_dir))
    # One file was indexed
    assert data["files_scanned"] == 1


def test_files_scanned_multiple_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text(_make_small_function("f1"), encoding="utf-8")
    (repo / "b.py").write_text(_make_small_function("f2"), encoding="utf-8")
    index_dir = str(tmp_path / "index")
    index_repository(str(repo), index_dir=index_dir)
    data = json.loads(chunk_repository(str(repo), index_dir=index_dir))
    assert data["files_scanned"] == 2


def test_files_scanned_excludes_files_outside_root(tmp_path):
    """Methods from a different root should not count toward files_scanned."""
    # Index two separate repos into the same index
    repo_a = tmp_path / "repo_a"
    repo_a.mkdir()
    (repo_a / "a.py").write_text(_make_small_function("a"), encoding="utf-8")

    repo_b = tmp_path / "repo_b"
    repo_b.mkdir()
    (repo_b / "b.py").write_text(_make_small_function("b"), encoding="utf-8")

    index_dir = str(tmp_path / "index")
    index_repository(str(repo_a), index_dir=index_dir)
    index_repository(str(repo_b), index_dir=index_dir)

    data = json.loads(chunk_repository(str(repo_a), index_dir=index_dir))
    # Only repo_a files should be scanned
    assert data["files_scanned"] == 1


# ---------------------------------------------------------------------------
# Tests: chunks_created count
# ---------------------------------------------------------------------------


def test_chunks_created_positive_for_large_function(tmp_path):
    repo, index_dir = _setup_index(tmp_path, _make_large_function(n_stmts=35))
    data = json.loads(chunk_repository(repo, index_dir=index_dir))
    assert data["chunks_created"] >= 2  # a 35-stmt function with max_stmts=10 → ≥3 chunks


def test_chunks_created_matches_stored_chunks(tmp_path):
    """chunks_created must equal the number of chunks stored in the registry."""
    repo, index_dir = _setup_index(tmp_path, _make_large_function(n_stmts=35))
    data = json.loads(chunk_repository(repo, index_dir=index_dir))

    registry = MethodRegistry(index_dir)
    methods = registry.get_all_methods()
    total_stored = sum(len(registry.get_chunks_by_function(m["id"])) for m in methods)
    registry.close()

    assert total_stored == data["chunks_created"]


# ---------------------------------------------------------------------------
# Tests: stored chunk metadata
# ---------------------------------------------------------------------------


def test_stored_chunks_have_correct_function_name(tmp_path):
    repo, index_dir = _setup_index(tmp_path, _make_large_function("my_big_func"))
    chunk_repository(repo, index_dir=index_dir)

    registry = MethodRegistry(index_dir)
    methods = registry.get_all_methods()
    big = next(m for m in methods if m["name"] == "my_big_func")
    chunks = registry.get_chunks_by_function(big["id"])
    registry.close()

    assert len(chunks) >= 1
    assert all(c["function_name"] == "my_big_func" for c in chunks)


def test_stored_chunks_have_required_fields(tmp_path):
    repo, index_dir = _setup_index(tmp_path, _make_large_function())
    chunk_repository(repo, index_dir=index_dir)

    registry = MethodRegistry(index_dir)
    methods = registry.get_all_methods()
    big = next(m for m in methods if m["name"] == "big_func")
    chunks = registry.get_chunks_by_function(big["id"])
    registry.close()

    required = {
        "id", "function_id", "chunk_index", "statement_start", "statement_end",
        "statement_indices", "function_name", "file_path",
        "depends_on_chunks", "depended_on_by_chunks", "faiss_pos",
    }
    for c in chunks:
        for field in required:
            assert field in c, f"Missing field {field!r} in chunk"


def test_stored_chunks_have_correct_function_id(tmp_path):
    repo, index_dir = _setup_index(tmp_path, _make_large_function())
    chunk_repository(repo, index_dir=index_dir)

    registry = MethodRegistry(index_dir)
    methods = registry.get_all_methods()
    big = next(m for m in methods if m["name"] == "big_func")
    chunks = registry.get_chunks_by_function(big["id"])
    registry.close()

    assert all(c["function_id"] == big["id"] for c in chunks)


def test_chunk_indices_are_sequential(tmp_path):
    """chunk_index values for a function must be 0, 1, 2, …"""
    repo, index_dir = _setup_index(tmp_path, _make_large_function(n_stmts=35))
    chunk_repository(repo, index_dir=index_dir)

    registry = MethodRegistry(index_dir)
    methods = registry.get_all_methods()
    big = next(m for m in methods if m["name"] == "big_func")
    chunks = sorted(registry.get_chunks_by_function(big["id"]), key=lambda c: c["chunk_index"])
    registry.close()

    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_statement_indices_partition_all_statements(tmp_path):
    """All statement indices must appear exactly once across all chunks."""
    repo, index_dir = _setup_index(tmp_path, _make_large_function(n_stmts=35))
    chunk_repository(repo, index_dir=index_dir)

    registry = MethodRegistry(index_dir)
    methods = registry.get_all_methods()
    big = next(m for m in methods if m["name"] == "big_func")
    chunks = registry.get_chunks_by_function(big["id"])
    registry.close()

    all_indices = [idx for c in chunks for idx in c["statement_indices"]]
    assert sorted(all_indices) == list(range(len(all_indices)))


# ---------------------------------------------------------------------------
# Tests: idempotency
# ---------------------------------------------------------------------------


def test_running_twice_does_not_duplicate_chunks(tmp_path):
    """Calling chunk_repository twice should replace old chunks, not double them."""
    repo, index_dir = _setup_index(tmp_path, _make_large_function(n_stmts=35))

    data1 = json.loads(chunk_repository(repo, index_dir=index_dir))
    data2 = json.loads(chunk_repository(repo, index_dir=index_dir))

    # The chunk counts should be equal across both runs
    assert data1["chunks_created"] == data2["chunks_created"]

    # The registry should contain the same number of chunks after the second run
    registry = MethodRegistry(index_dir)
    methods = registry.get_all_methods()
    big = next(m for m in methods if m["name"] == "big_func")
    stored_count = len(registry.get_chunks_by_function(big["id"]))
    registry.close()

    assert stored_count == data1["chunks_created"]


# ---------------------------------------------------------------------------
# Tests: max_statements_per_chunk parameter
# ---------------------------------------------------------------------------


def test_smaller_max_stmts_produces_more_chunks(tmp_path):
    """Reducing max_statements_per_chunk should create more (smaller) chunks."""
    repo, index_dir = _setup_index(tmp_path, _make_large_function(n_stmts=40))

    data_large = json.loads(
        chunk_repository(repo, index_dir=index_dir, max_statements_per_chunk=20)
    )
    data_small = json.loads(
        chunk_repository(repo, index_dir=index_dir, max_statements_per_chunk=5)
    )

    assert data_small["chunks_created"] >= data_large["chunks_created"]


def test_max_stmts_respected_in_stored_chunks(tmp_path):
    """No stored chunk should contain more statement_indices than max_statements_per_chunk."""
    max_s = 8
    repo, index_dir = _setup_index(tmp_path, _make_large_function(n_stmts=40))
    chunk_repository(repo, index_dir=index_dir, max_statements_per_chunk=max_s)

    registry = MethodRegistry(index_dir)
    methods = registry.get_all_methods()
    big = next(m for m in methods if m["name"] == "big_func")
    chunks = registry.get_chunks_by_function(big["id"])
    registry.close()

    for c in chunks:
        assert len(c["statement_indices"]) <= max_s


# ---------------------------------------------------------------------------
# Tests: embeddings stored correctly
# ---------------------------------------------------------------------------


def test_stored_chunks_have_valid_faiss_pos(tmp_path):
    """Every stored chunk should have a non-negative faiss_pos."""
    repo, index_dir = _setup_index(tmp_path, _make_large_function(n_stmts=35))
    chunk_repository(repo, index_dir=index_dir)

    registry = MethodRegistry(index_dir)
    methods = registry.get_all_methods()
    big = next(m for m in methods if m["name"] == "big_func")
    chunks = registry.get_chunks_by_function(big["id"])
    registry.close()

    for c in chunks:
        assert isinstance(c["faiss_pos"], int)
        assert c["faiss_pos"] >= 0


def test_chunk_faiss_positions_are_unique(tmp_path):
    repo, index_dir = _setup_index(tmp_path, _make_large_function(n_stmts=35))
    chunk_repository(repo, index_dir=index_dir)

    registry = MethodRegistry(index_dir)
    methods = registry.get_all_methods()
    big = next(m for m in methods if m["name"] == "big_func")
    chunks = registry.get_chunks_by_function(big["id"])
    registry.close()

    positions = [c["faiss_pos"] for c in chunks]
    assert len(positions) == len(set(positions))


# ---------------------------------------------------------------------------
# Tests: multiple large functions
# ---------------------------------------------------------------------------


def test_multiple_large_functions_all_chunked(tmp_path):
    content = (
        _make_large_function("alpha", n_stmts=35)
        + "\n\n"
        + _make_large_function("beta", n_stmts=40)
    )
    repo, index_dir = _setup_index(tmp_path, content)
    data = json.loads(chunk_repository(repo, index_dir=index_dir))
    assert data["functions_chunked"] == 2
    assert data["chunks_created"] >= 4  # at least 2 chunks each


def test_chunks_stored_per_function(tmp_path):
    content = (
        _make_large_function("alpha", n_stmts=35)
        + "\n\n"
        + _make_large_function("beta", n_stmts=40)
    )
    repo, index_dir = _setup_index(tmp_path, content)
    chunk_repository(repo, index_dir=index_dir)

    registry = MethodRegistry(index_dir)
    methods = registry.get_all_methods()
    for func_name in ("alpha", "beta"):
        m = next(m for m in methods if m["name"] == func_name)
        chunks = registry.get_chunks_by_function(m["id"])
        assert len(chunks) >= 1, f"{func_name} should have stored chunks"
    registry.close()


# ---------------------------------------------------------------------------
# Tests: get_chunk_count helper
# ---------------------------------------------------------------------------


def test_get_chunk_count_zero_initially(tmp_path):
    repo, index_dir = _setup_index(tmp_path, _make_small_function())
    chunk_repository(repo, index_dir=index_dir)
    registry = MethodRegistry(index_dir)
    assert registry.get_chunk_count() == 0
    registry.close()


def test_get_chunk_count_matches_chunks_created(tmp_path):
    repo, index_dir = _setup_index(tmp_path, _make_large_function(n_stmts=35))
    data = json.loads(chunk_repository(repo, index_dir=index_dir))
    registry = MethodRegistry(index_dir)
    assert registry.get_chunk_count() == data["chunks_created"]
    registry.close()
