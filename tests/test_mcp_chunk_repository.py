"""Integration tests for the MCP chunk_repository tool (feature #25)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

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
    for key in ("files_scanned", "functions_chunked", "chunks_created", "skipped_files"):
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
        # feature #29: new metadata columns
        "normalized_code", "code_hash", "dependency_links", "created_at",
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
    """Calling chunk_repository twice should not create duplicate chunks."""
    repo, index_dir = _setup_index(tmp_path, _make_large_function(n_stmts=35))

    data1 = json.loads(chunk_repository(repo, index_dir=index_dir))
    data2 = json.loads(chunk_repository(repo, index_dir=index_dir))

    # Second run should skip unchanged functions — nothing new created
    assert data2["chunks_created"] == 0
    assert data2["functions_chunked"] == 0

    # The stored count must equal what the first run created (no duplicates)
    registry = MethodRegistry(index_dir)
    methods = registry.get_all_methods()
    big = next(m for m in methods if m["name"] == "big_func")
    stored_count = len(registry.get_chunks_by_function(big["id"]))
    registry.close()

    assert stored_count == data1["chunks_created"]


def test_second_run_unchanged_skips_all_functions(tmp_path):
    """Second run on an unchanged repo reports zero functions_chunked."""
    content = (
        _make_large_function("alpha", n_stmts=35)
        + "\n\n"
        + _make_large_function("beta", n_stmts=40)
    )
    repo, index_dir = _setup_index(tmp_path, content)

    chunk_repository(repo, index_dir=index_dir)
    data2 = json.loads(chunk_repository(repo, index_dir=index_dir))

    assert data2["functions_chunked"] == 0
    assert data2["chunks_created"] == 0


# ---------------------------------------------------------------------------
# Tests: max_statements_per_chunk parameter
# ---------------------------------------------------------------------------


def test_smaller_max_stmts_produces_more_chunks(tmp_path):
    """Reducing max_statements_per_chunk should create more (smaller) chunks."""
    repo, index_dir = _setup_index(tmp_path, _make_large_function(n_stmts=40))

    data_large = json.loads(
        chunk_repository(repo, index_dir=index_dir, max_statements_per_chunk=20)
    )
    # force_rechunk=True so the second call replaces existing chunks
    data_small = json.loads(
        chunk_repository(repo, index_dir=index_dir, max_statements_per_chunk=5, force_rechunk=True)
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


# ---------------------------------------------------------------------------
# Tests: re-chunking after file change (feature #28)
# ---------------------------------------------------------------------------


def test_force_rechunk_replaces_existing_chunks(tmp_path):
    """force_rechunk=True must replace existing chunks, not accumulate them."""
    repo, index_dir = _setup_index(tmp_path, _make_large_function(n_stmts=35))

    data1 = json.loads(chunk_repository(repo, index_dir=index_dir))
    data2 = json.loads(chunk_repository(repo, index_dir=index_dir, force_rechunk=True))

    # Both runs should have created the same number of chunks
    assert data2["chunks_created"] == data1["chunks_created"]
    assert data2["functions_chunked"] == 1

    # Stored count must equal one run's worth (no accumulation)
    registry = MethodRegistry(index_dir)
    methods = registry.get_all_methods()
    big = next(m for m in methods if m["name"] == "big_func")
    stored_count = len(registry.get_chunks_by_function(big["id"]))
    registry.close()

    assert stored_count == data1["chunks_created"]


def test_reindexed_function_gets_rechunked(tmp_path):
    """After a file is re-indexed, chunk_repository must rechunk the updated function."""
    repo = tmp_path / "repo"
    repo.mkdir()
    src = repo / "module.py"
    index_dir = str(tmp_path / "index")

    # Initial indexing and chunking
    src.write_text(_make_large_function("evolving_func", n_stmts=35), encoding="utf-8")
    index_repository(str(repo), index_dir=index_dir)
    data1 = json.loads(chunk_repository(str(repo), index_dir=index_dir))
    assert data1["functions_chunked"] == 1
    chunks_first = data1["chunks_created"]

    # Re-index with a different function (more statements)
    src.write_text(_make_large_function("evolving_func", n_stmts=50), encoding="utf-8")
    index_repository(str(repo), index_dir=index_dir, force_reindex=True)

    # chunk_repository should rechunk the updated function (new function_id, no existing chunks)
    data2 = json.loads(chunk_repository(str(repo), index_dir=index_dir))
    assert data2["functions_chunked"] == 1
    assert data2["chunks_created"] >= 1

    # Stored count must reflect only the new chunks, not old + new
    registry = MethodRegistry(index_dir)
    total_stored = registry.get_chunk_count()
    registry.close()

    assert total_stored == data2["chunks_created"]


def test_reindexing_removes_orphaned_chunks(tmp_path):
    """delete_by_file must cascade-delete chunks for the removed methods."""
    repo = tmp_path / "repo"
    repo.mkdir()
    src = repo / "module.py"
    index_dir = str(tmp_path / "index")

    # Index and chunk the file
    src.write_text(_make_large_function(n_stmts=35), encoding="utf-8")
    index_repository(str(repo), index_dir=index_dir)
    chunk_repository(str(repo), index_dir=index_dir)

    registry = MethodRegistry(index_dir)
    assert registry.get_chunk_count() > 0, "Expected chunks to be stored"
    registry.close()

    # Re-index with a small function — no large functions, so no chunks should remain
    src.write_text(_make_small_function(), encoding="utf-8")
    index_repository(str(repo), index_dir=index_dir, force_reindex=True)

    # Old orphaned chunks must have been cascade-deleted by delete_by_file
    registry = MethodRegistry(index_dir)
    orphan_count = registry.get_chunk_count()
    registry.close()

    assert orphan_count == 0, (
        f"Expected 0 orphaned chunks after re-index, got {orphan_count}"
    )


def test_only_new_function_chunked_when_stable_already_chunked(tmp_path):
    """A function already chunked is skipped; a newly indexed function is chunked."""
    repo = tmp_path / "repo"
    repo.mkdir()
    index_dir = str(tmp_path / "index")

    # Index and chunk only the stable file first
    (repo / "stable.py").write_text(_make_large_function("stable_func", n_stmts=35), encoding="utf-8")
    index_repository(str(repo), index_dir=index_dir)
    chunk_repository(str(repo), index_dir=index_dir)

    registry = MethodRegistry(index_dir)
    methods_before = {m["name"]: m for m in registry.get_all_methods()}
    stable_id_before = methods_before["stable_func"]["id"]
    stable_chunks_before = len(registry.get_chunks_by_function(stable_id_before))
    registry.close()

    # Add a brand-new large function in a second file (not yet indexed)
    (repo / "new.py").write_text(_make_large_function("new_func", n_stmts=40), encoding="utf-8")
    # index_repository with force_reindex=False skips already-indexed files,
    # so only new.py is indexed here and stable.py keeps its original method IDs.
    index_repository(str(repo), index_dir=index_dir)

    # chunk_repository: stable_func already has chunks → skipped; new_func → chunked
    data = json.loads(chunk_repository(str(repo), index_dir=index_dir))
    assert data["functions_chunked"] == 1

    registry = MethodRegistry(index_dir)
    methods_after = {m["name"]: m for m in registry.get_all_methods()}
    # stable_func must retain its original ID and the same chunks
    assert methods_after["stable_func"]["id"] == stable_id_before
    stable_chunks_after = len(registry.get_chunks_by_function(stable_id_before))
    registry.close()

    assert stable_chunks_after == stable_chunks_before


# ---------------------------------------------------------------------------
# Tests: feature #35 — no large functions → empty result, no exception
# ---------------------------------------------------------------------------


def _make_at_threshold_function(name: str = "at_threshold") -> str:
    """Generate a function with exactly 30 statements (not chunked, since threshold is >30).

    count_statements includes the 'def' header, so:
    1 (def) + 28 (assignments) + 1 (return) = 30 total.
    """
    lines = [f"def {name}():"]
    for i in range(28):
        lines.append(f"    x_{i} = {i}")
    lines.append("    return x_0")
    return "\n".join(lines) + "\n"


def test_no_large_functions_returns_valid_summary(tmp_path):
    """Step 1+2: all functions ≤30 stmts → valid summary with zero chunking counts."""
    repo, index_dir = _setup_index(tmp_path, _make_small_function())
    data = json.loads(chunk_repository(repo, index_dir=index_dir))
    assert data == {"files_scanned": 1, "functions_chunked": 0, "chunks_created": 0, "skipped_files": []}


def test_no_large_functions_no_exception_raised(tmp_path):
    """Step 3: chunk_repository must not raise when no functions exceed the threshold."""
    repo, index_dir = _setup_index(tmp_path, _make_small_function())
    # If an exception were raised, pytest would fail here automatically.
    result = chunk_repository(repo, index_dir=index_dir)
    assert isinstance(result, str)


def test_no_large_functions_no_chunk_entries_written(tmp_path):
    """Step 4: chunk table must remain empty when no functions exceed the threshold."""
    repo, index_dir = _setup_index(tmp_path, _make_small_function())
    chunk_repository(repo, index_dir=index_dir)
    registry = MethodRegistry(index_dir)
    assert registry.get_chunk_count() == 0
    registry.close()


def test_at_threshold_functions_not_chunked(tmp_path):
    """Functions with exactly 30 statements (= threshold) must not be chunked."""
    repo, index_dir = _setup_index(tmp_path, _make_at_threshold_function())
    data = json.loads(chunk_repository(repo, index_dir=index_dir))
    assert data["functions_chunked"] == 0
    assert data["chunks_created"] == 0
    registry = MethodRegistry(index_dir)
    assert registry.get_chunk_count() == 0
    registry.close()


def test_multiple_small_functions_all_skipped(tmp_path):
    """Multiple small functions across multiple files → files_scanned=N, zeros for chunking."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text(_make_small_function("f1"), encoding="utf-8")
    (repo / "b.py").write_text(_make_small_function("f2"), encoding="utf-8")
    (repo / "c.py").write_text(_make_small_function("f3"), encoding="utf-8")
    index_dir = str(tmp_path / "index")
    index_repository(str(repo), index_dir=index_dir)

    data = json.loads(chunk_repository(str(repo), index_dir=index_dir))
    assert data["files_scanned"] == 3
    assert data["functions_chunked"] == 0
    assert data["chunks_created"] == 0


def test_empty_index_returns_zero_summary(tmp_path):
    """chunk_repository on a valid dir with no indexed methods returns all zeros."""
    repo = tmp_path / "repo"
    repo.mkdir()
    index_dir = str(tmp_path / "index")
    # Do NOT call index_repository — the index is empty.
    data = json.loads(chunk_repository(str(repo), index_dir=index_dir))
    assert data == {"files_scanned": 0, "functions_chunked": 0, "chunks_created": 0, "skipped_files": []}


def test_empty_index_no_exception(tmp_path):
    """chunk_repository on an empty index must not raise."""
    repo = tmp_path / "repo"
    repo.mkdir()
    index_dir = str(tmp_path / "index")
    result = chunk_repository(str(repo), index_dir=index_dir)
    assert "error" not in json.loads(result)


# ---------------------------------------------------------------------------
# Tests: feature #36 — malformed Python file is skipped gracefully
# ---------------------------------------------------------------------------


_MALFORMED_PYTHON = "def broken(\n    x =\n"  # invalid syntax


def test_malformed_file_in_repo_does_not_raise(tmp_path):
    """A malformed .py file alongside valid ones must not cause chunk_repository to raise."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "valid.py").write_text(_make_large_function("valid_func", n_stmts=35), encoding="utf-8")
    (repo / "broken.py").write_text(_MALFORMED_PYTHON, encoding="utf-8")
    index_dir = str(tmp_path / "index")
    # index_repository already skips the malformed file silently.
    index_repository(str(repo), index_dir=index_dir)
    # chunk_repository must complete without raising even if broken.py is present.
    result = chunk_repository(str(repo), index_dir=index_dir)
    data = json.loads(result)
    assert "error" not in data


def test_malformed_file_valid_functions_still_chunked(tmp_path):
    """Valid large functions are chunked even when the repo contains a malformed file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "valid.py").write_text(_make_large_function("valid_func", n_stmts=35), encoding="utf-8")
    (repo / "broken.py").write_text(_MALFORMED_PYTHON, encoding="utf-8")
    index_dir = str(tmp_path / "index")
    index_repository(str(repo), index_dir=index_dir)
    data = json.loads(chunk_repository(str(repo), index_dir=index_dir))
    assert data["functions_chunked"] == 1
    assert data["chunks_created"] >= 1


def test_skipped_files_empty_when_all_succeed(tmp_path):
    """skipped_files must be an empty list when no function raises during chunking."""
    repo, index_dir = _setup_index(tmp_path, _make_large_function(n_stmts=35))
    data = json.loads(chunk_repository(repo, index_dir=index_dir))
    assert data["skipped_files"] == []


def test_chunking_failure_reported_in_skipped_files(tmp_path):
    """When build_dependency_graph raises, the function's file appears in skipped_files."""
    repo, index_dir = _setup_index(tmp_path, _make_large_function(n_stmts=35))
    with patch(
        "code_similarity_mcp.mcp.server.build_dependency_graph",
        side_effect=SyntaxError("simulated parse failure"),
    ):
        data = json.loads(chunk_repository(repo, index_dir=index_dir))
    assert len(data["skipped_files"]) == 1
    assert data["skipped_files"][0].endswith("module.py")


def test_chunking_failure_no_exception_raised(tmp_path):
    """chunk_repository must not propagate exceptions from build_dependency_graph."""
    repo, index_dir = _setup_index(tmp_path, _make_large_function(n_stmts=35))
    with patch(
        "code_similarity_mcp.mcp.server.build_dependency_graph",
        side_effect=RuntimeError("unexpected failure"),
    ):
        result = chunk_repository(repo, index_dir=index_dir)
    # Must return valid JSON with no top-level 'error' key.
    data = json.loads(result)
    assert "error" not in data


def test_chunking_failure_zero_functions_chunked(tmp_path):
    """When every function fails during chunking, functions_chunked is 0."""
    repo, index_dir = _setup_index(tmp_path, _make_large_function(n_stmts=35))
    with patch(
        "code_similarity_mcp.mcp.server.build_dependency_graph",
        side_effect=ValueError("bad graph"),
    ):
        data = json.loads(chunk_repository(repo, index_dir=index_dir))
    assert data["functions_chunked"] == 0
    assert data["chunks_created"] == 0


def test_partial_failure_valid_function_still_chunked(tmp_path):
    """When one function fails, other functions in different files are still chunked."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "good.py").write_text(_make_large_function("good_func", n_stmts=35), encoding="utf-8")
    (repo / "bad.py").write_text(_make_large_function("bad_func", n_stmts=35), encoding="utf-8")
    index_dir = str(tmp_path / "index")
    index_repository(str(repo), index_dir=index_dir)

    # Patch build_dependency_graph to fail only for bad_func
    original_bdg = __import__(
        "code_similarity_mcp.parser.python", fromlist=["build_dependency_graph"]
    ).build_dependency_graph

    def selective_fail(body_code: str):
        if "bad_func" in body_code or body_code.count("x_") > 30:
            # Fail for the first large function encountered by name heuristic;
            # use a simpler discriminator: fail exactly once via a counter.
            raise SyntaxError("malformed")
        return original_bdg(body_code)

    call_count = {"n": 0}

    def fail_first_call(body_code: str):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise SyntaxError("simulated failure for first function")
        return original_bdg(body_code)

    with patch("code_similarity_mcp.mcp.server.build_dependency_graph", side_effect=fail_first_call):
        data = json.loads(chunk_repository(str(repo), index_dir=index_dir))

    # One function failed, one succeeded
    assert data["functions_chunked"] == 1
    assert data["chunks_created"] >= 1
    assert len(data["skipped_files"]) == 1


def test_skipped_files_deduplicated_per_file(tmp_path):
    """If multiple functions in the same file fail, the file appears only once in skipped_files."""
    content = (
        _make_large_function("func_a", n_stmts=35)
        + "\n\n"
        + _make_large_function("func_b", n_stmts=35)
    )
    repo, index_dir = _setup_index(tmp_path, content)
    with patch(
        "code_similarity_mcp.mcp.server.build_dependency_graph",
        side_effect=SyntaxError("fail all"),
    ):
        data = json.loads(chunk_repository(repo, index_dir=index_dir))
    # Both functions are in the same file — skipped_files must have exactly one entry.
    assert len(data["skipped_files"]) == 1
