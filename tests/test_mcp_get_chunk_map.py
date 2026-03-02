"""Integration tests for the MCP get_chunk_map tool (feature #27)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from code_similarity_mcp.index.registry import MethodRegistry
from code_similarity_mcp.mcp.server import chunk_repository, get_chunk_map, index_repository


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


def _setup_index_and_chunks(
    tmp_path: Path,
    content: str,
    filename: str = "module.py",
) -> tuple[str, str]:
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


def test_neither_function_id_nor_file_path_returns_error(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    data = json.loads(get_chunk_map(index_dir=index_dir))
    assert "error" in data


def test_both_function_id_and_file_path_returns_error(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "big_func")
    registry.close()

    data = json.loads(
        get_chunk_map(function_id=method["id"], file_path=method["file_path"], index_dir=index_dir)
    )
    assert "error" in data


def test_nonexistent_function_id_returns_error(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    data = json.loads(get_chunk_map(function_id=999999, index_dir=index_dir))
    assert "error" in data
    assert "999999" in data["error"]
    assert "not found" in data["error"]


def test_function_id_with_no_chunks_returns_metadata_and_hint(tmp_path):
    """A function that was indexed but never chunked returns metadata + empty chunks + hint."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text(_make_small_function(), encoding="utf-8")
    index_dir = str(tmp_path / "index")
    index_repository(str(repo), index_dir=index_dir)
    # Do NOT chunk → no chunks stored

    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "small_func")
    registry.close()

    data = json.loads(get_chunk_map(function_id=method["id"], index_dir=index_dir))
    assert "error" not in data
    assert "functions" in data
    assert len(data["functions"]) == 1
    fn = data["functions"][0]
    assert fn["function_name"] == "small_func"
    assert fn["chunks"] == []
    assert fn["hint"] == "Run chunk_repository to generate chunks"


def test_file_path_with_no_chunks_returns_empty_functions(tmp_path):
    repo, index_dir = _setup_index_and_chunks(tmp_path, _make_small_function())
    # chunk_repository ran but small functions aren't chunked
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods())
    registry.close()

    data = json.loads(get_chunk_map(file_path=method["file_path"], index_dir=index_dir))
    assert data == {"functions": []}


# ---------------------------------------------------------------------------
# Tests: response shape — function_id query
# ---------------------------------------------------------------------------


def test_function_id_returns_valid_json(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "big_func")
    registry.close()

    raw = get_chunk_map(function_id=method["id"], index_dir=index_dir)
    data = json.loads(raw)
    assert isinstance(data, dict)


def test_function_id_has_functions_key(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "big_func")
    registry.close()

    data = json.loads(get_chunk_map(function_id=method["id"], index_dir=index_dir))
    assert "functions" in data


def test_function_id_returns_one_function_entry(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "big_func")
    registry.close()

    data = json.loads(get_chunk_map(function_id=method["id"], index_dir=index_dir))
    assert len(data["functions"]) == 1


def test_function_entry_has_required_fields(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "big_func")
    registry.close()

    data = json.loads(get_chunk_map(function_id=method["id"], index_dir=index_dir))
    entry = data["functions"][0]
    for field in ("function_id", "function_name", "file", "dag_valid", "chunks"):
        assert field in entry, f"Missing field {field!r}"


def test_function_entry_metadata_correct(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function("my_func"))
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "my_func")
    registry.close()

    data = json.loads(get_chunk_map(function_id=method["id"], index_dir=index_dir))
    entry = data["functions"][0]
    assert entry["function_id"] == method["id"]
    assert entry["function_name"] == "my_func"
    assert entry["file"] == method["file_path"]


def test_chunks_list_is_non_empty(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "big_func")
    registry.close()

    data = json.loads(get_chunk_map(function_id=method["id"], index_dir=index_dir))
    assert len(data["functions"][0]["chunks"]) >= 1


# ---------------------------------------------------------------------------
# Tests: chunk entry shape
# ---------------------------------------------------------------------------


def test_chunk_entries_have_required_fields(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "big_func")
    registry.close()

    data = json.loads(get_chunk_map(function_id=method["id"], index_dir=index_dir))
    required = {"chunk_id", "chunk_index", "statement_range", "dependencies", "normalized_code"}
    for chunk in data["functions"][0]["chunks"]:
        for field in required:
            assert field in chunk, f"Missing chunk field {field!r}"


def test_chunk_ids_are_positive_integers(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "big_func")
    registry.close()

    data = json.loads(get_chunk_map(function_id=method["id"], index_dir=index_dir))
    for chunk in data["functions"][0]["chunks"]:
        assert isinstance(chunk["chunk_id"], int)
        assert chunk["chunk_id"] > 0


def test_chunk_indices_are_sequential(tmp_path):
    """chunk_index values must be 0, 1, 2, …"""
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function(n_stmts=35))
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "big_func")
    registry.close()

    data = json.loads(get_chunk_map(function_id=method["id"], index_dir=index_dir))
    indices = [c["chunk_index"] for c in data["functions"][0]["chunks"]]
    assert indices == list(range(len(indices)))


def test_statement_range_is_two_ints(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "big_func")
    registry.close()

    data = json.loads(get_chunk_map(function_id=method["id"], index_dir=index_dir))
    for chunk in data["functions"][0]["chunks"]:
        sr = chunk["statement_range"]
        assert isinstance(sr, list) and len(sr) == 2
        assert all(isinstance(v, int) for v in sr)
        assert sr[0] <= sr[1]


def test_dependencies_is_list_of_ints(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "big_func")
    registry.close()

    data = json.loads(get_chunk_map(function_id=method["id"], index_dir=index_dir))
    for chunk in data["functions"][0]["chunks"]:
        assert isinstance(chunk["dependencies"], list)
        assert all(isinstance(d, int) for d in chunk["dependencies"])


def test_normalized_code_is_nonempty_string(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "big_func")
    registry.close()

    data = json.loads(get_chunk_map(function_id=method["id"], index_dir=index_dir))
    for chunk in data["functions"][0]["chunks"]:
        assert isinstance(chunk["normalized_code"], str)
        assert len(chunk["normalized_code"]) > 0


# ---------------------------------------------------------------------------
# Tests: DAG validity
# ---------------------------------------------------------------------------


def test_dag_valid_is_bool(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "big_func")
    registry.close()

    data = json.loads(get_chunk_map(function_id=method["id"], index_dir=index_dir))
    assert isinstance(data["functions"][0]["dag_valid"], bool)


def test_dag_valid_is_true_for_well_formed_chunks(tmp_path):
    """Chunks produced by chunk_repository must always form a valid DAG."""
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function(n_stmts=35))
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "big_func")
    registry.close()

    data = json.loads(get_chunk_map(function_id=method["id"], index_dir=index_dir))
    assert data["functions"][0]["dag_valid"] is True


def test_dependencies_reference_earlier_chunks_only(tmp_path):
    """All dependency indices must be < the current chunk_index."""
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function(n_stmts=35))
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "big_func")
    registry.close()

    data = json.loads(get_chunk_map(function_id=method["id"], index_dir=index_dir))
    for chunk in data["functions"][0]["chunks"]:
        for dep in chunk["dependencies"]:
            assert dep < chunk["chunk_index"], (
                f"Chunk {chunk['chunk_index']} depends on later chunk {dep}"
            )


# ---------------------------------------------------------------------------
# Tests: file_path query
# ---------------------------------------------------------------------------


def test_file_path_query_returns_valid_json(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "big_func")
    registry.close()

    raw = get_chunk_map(file_path=method["file_path"], index_dir=index_dir)
    data = json.loads(raw)
    assert isinstance(data, dict)


def test_file_path_query_returns_functions_key(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "big_func")
    registry.close()

    data = json.loads(get_chunk_map(file_path=method["file_path"], index_dir=index_dir))
    assert "functions" in data


def test_file_path_query_one_function(tmp_path):
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "big_func")
    registry.close()

    data = json.loads(get_chunk_map(file_path=method["file_path"], index_dir=index_dir))
    assert len(data["functions"]) == 1
    assert data["functions"][0]["function_name"] == "big_func"


def test_file_path_query_multiple_large_functions(tmp_path):
    content = (
        _make_large_function("alpha", n_stmts=35)
        + "\n\n"
        + _make_large_function("beta", n_stmts=40)
    )
    _, index_dir = _setup_index_and_chunks(tmp_path, content)
    registry = MethodRegistry(index_dir)
    # Pick the file path for the indexed functions
    methods = registry.get_all_methods()
    file_path = next(m for m in methods if m["name"] == "alpha")["file_path"]
    registry.close()

    data = json.loads(get_chunk_map(file_path=file_path, index_dir=index_dir))
    names = {f["function_name"] for f in data["functions"]}
    assert names == {"alpha", "beta"}


def test_file_path_query_skips_small_functions(tmp_path):
    """Small functions that were not chunked should not appear in the map."""
    content = _make_small_function("tiny") + "\n\n" + _make_large_function("huge")
    _, index_dir = _setup_index_and_chunks(tmp_path, content)
    registry = MethodRegistry(index_dir)
    methods = registry.get_all_methods()
    file_path = next(m for m in methods if m["name"] == "huge")["file_path"]
    registry.close()

    data = json.loads(get_chunk_map(file_path=file_path, index_dir=index_dir))
    names = {f["function_name"] for f in data["functions"]}
    assert "tiny" not in names
    assert "huge" in names


def test_file_path_functions_sorted_by_name(tmp_path):
    content = (
        _make_large_function("zebra", n_stmts=35)
        + "\n\n"
        + _make_large_function("apple", n_stmts=35)
        + "\n\n"
        + _make_large_function("mango", n_stmts=35)
    )
    _, index_dir = _setup_index_and_chunks(tmp_path, content)
    registry = MethodRegistry(index_dir)
    methods = registry.get_all_methods()
    file_path = methods[0]["file_path"]
    registry.close()

    data = json.loads(get_chunk_map(file_path=file_path, index_dir=index_dir))
    names = [f["function_name"] for f in data["functions"]]
    assert names == sorted(names)


# ---------------------------------------------------------------------------
# Tests: consistency between function_id and file_path queries
# ---------------------------------------------------------------------------


def test_function_id_and_file_path_return_same_data(tmp_path):
    """Querying by function_id or the containing file should yield identical data."""
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function("check_func"))
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "check_func")
    registry.close()

    by_id = json.loads(get_chunk_map(function_id=method["id"], index_dir=index_dir))
    by_file = json.loads(get_chunk_map(file_path=method["file_path"], index_dir=index_dir))

    assert len(by_id["functions"]) == 1
    assert len(by_file["functions"]) == 1
    assert by_id["functions"][0] == by_file["functions"][0]


# ---------------------------------------------------------------------------
# Tests: normalized_code content sanity
# ---------------------------------------------------------------------------


def test_normalized_code_contains_chunk_func(tmp_path):
    """The normalized_code of each chunk should be wrapped in _chunk_func."""
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "big_func")
    registry.close()

    data = json.loads(get_chunk_map(function_id=method["id"], index_dir=index_dir))
    for chunk in data["functions"][0]["chunks"]:
        # The normalizer renames `_chunk_func` → FUNC_NAME
        assert "FUNC_NAME" in chunk["normalized_code"] or "_chunk_func" in chunk["normalized_code"]


def test_chunk_count_matches_registry(tmp_path):
    """The number of chunks in the map must equal those stored in the registry."""
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function(n_stmts=35))
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "big_func")
    stored_count = len(registry.get_chunks_by_function(method["id"]))
    registry.close()

    data = json.loads(get_chunk_map(function_id=method["id"], index_dir=index_dir))
    assert len(data["functions"][0]["chunks"]) == stored_count


# ---------------------------------------------------------------------------
# Tests: feature #39 — error for unknown function_id
# ---------------------------------------------------------------------------


def test_unknown_function_id_error_has_error_key(tmp_path):
    """get_chunk_map with a non-existent function_id must return an error key."""
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    data = json.loads(get_chunk_map(function_id=99999, index_dir=index_dir))
    assert "error" in data


def test_unknown_function_id_error_message_contains_id(tmp_path):
    """The error message must include the unknown function_id."""
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    data = json.loads(get_chunk_map(function_id=99999, index_dir=index_dir))
    assert "99999" in data["error"]


def test_unknown_function_id_error_message_says_not_found(tmp_path):
    """The error message must say the function was not found in index."""
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    data = json.loads(get_chunk_map(function_id=99999, index_dir=index_dir))
    assert "not found" in data["error"].lower()


def test_unknown_function_id_no_functions_key(tmp_path):
    """Error response for unknown function_id must not have a 'functions' key."""
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    data = json.loads(get_chunk_map(function_id=99999, index_dir=index_dir))
    assert "functions" not in data


def test_unknown_function_id_error_and_valid_id_work_in_same_session(tmp_path):
    """Unknown function_id returns error; valid function_id still works correctly."""
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function("session_func"))

    # Unknown id should return error
    error_data = json.loads(get_chunk_map(function_id=99999, index_dir=index_dir))
    assert "error" in error_data

    # Valid id should return correct data
    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "session_func")
    registry.close()

    ok_data = json.loads(get_chunk_map(function_id=method["id"], index_dir=index_dir))
    assert "functions" in ok_data
    assert len(ok_data["functions"]) == 1
    assert ok_data["functions"][0]["function_name"] == "session_func"


def test_exact_error_message_format(tmp_path):
    """Error message must match 'Function {id} not found in index'."""
    _, index_dir = _setup_index_and_chunks(tmp_path, _make_large_function())
    data = json.loads(get_chunk_map(function_id=99999, index_dir=index_dir))
    assert data["error"] == "Function 99999 not found in index"


# ---------------------------------------------------------------------------
# Feature #40: unprocessed large function returns empty chunks + hint
# ---------------------------------------------------------------------------


def test_unprocessed_large_function_returns_empty_chunks_and_hint(tmp_path):
    """Index a repo WITHOUT chunking; get_chunk_map must return function metadata,
    empty chunks list, and a hint — no error key."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "big.py").write_text(_make_large_function("unprocessed_func"), encoding="utf-8")
    index_dir = str(tmp_path / "index")
    index_repository(str(repo), index_dir=index_dir)
    # Intentionally skip chunk_repository

    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "unprocessed_func")
    func_id = method["id"]
    registry.close()

    data = json.loads(get_chunk_map(function_id=func_id, index_dir=index_dir))

    # Must not contain an error key
    assert "error" not in data

    # Must contain exactly one function entry
    assert "functions" in data
    assert len(data["functions"]) == 1

    fn = data["functions"][0]
    assert fn["function_name"] == "unprocessed_func"
    assert "file" in fn
    assert fn["chunks"] == []
    assert fn["hint"] == "Run chunk_repository to generate chunks"


def test_unprocessed_function_distinct_from_unknown_function_id(tmp_path):
    """Error path (unknown id) must have 'error' key; unprocessed path must not."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "big.py").write_text(_make_large_function("diff_func"), encoding="utf-8")
    index_dir = str(tmp_path / "index")
    index_repository(str(repo), index_dir=index_dir)

    registry = MethodRegistry(index_dir)
    method = next(m for m in registry.get_all_methods() if m["name"] == "diff_func")
    func_id = method["id"]
    registry.close()

    # Valid id, no chunks → no error key
    ok = json.loads(get_chunk_map(function_id=func_id, index_dir=index_dir))
    assert "error" not in ok

    # Unknown id → error key present
    err = json.loads(get_chunk_map(function_id=func_id + 99999, index_dir=index_dir))
    assert "error" in err
