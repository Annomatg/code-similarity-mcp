"""Integration tests for the MCP index_repository tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from code_similarity_mcp.mcp.server import index_repository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIMPLE_PY = """\
def add(a, b):
    return a + b


def subtract(x, y):
    return x - y
"""

_ANOTHER_PY = """\
def multiply(a, b):
    result = a * b
    return result
"""

_NOT_PYTHON = """\
// This is not Python
function hello() {
    console.log("hello");
}
"""


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests: basic functionality
# ---------------------------------------------------------------------------


def test_returns_valid_json(tmp_path):
    _write(tmp_path / "src" / "math.py", _SIMPLE_PY)
    index_dir = str(tmp_path / "index")
    raw = index_repository(str(tmp_path), index_dir=index_dir)
    data = json.loads(raw)
    assert "files_processed" in data
    assert "methods_indexed" in data
    assert "index_dir" in data


def test_indexes_python_file(tmp_path):
    _write(tmp_path / "math.py", _SIMPLE_PY)
    index_dir = str(tmp_path / "index")
    data = json.loads(index_repository(str(tmp_path), index_dir=index_dir))
    assert data["files_processed"] == 1
    assert data["methods_indexed"] == 2  # add + subtract


def test_indexes_multiple_files(tmp_path):
    _write(tmp_path / "math.py", _SIMPLE_PY)
    _write(tmp_path / "ops.py", _ANOTHER_PY)
    index_dir = str(tmp_path / "index")
    data = json.loads(index_repository(str(tmp_path), index_dir=index_dir))
    assert data["files_processed"] == 2
    assert data["methods_indexed"] == 3  # add + subtract + multiply


def test_index_dir_in_result(tmp_path):
    _write(tmp_path / "math.py", _SIMPLE_PY)
    custom_index = tmp_path / "my_index"
    data = json.loads(
        index_repository(str(tmp_path), index_dir=str(custom_index))
    )
    assert data["index_dir"] == str(custom_index)


def test_empty_directory_returns_zero_counts(tmp_path):
    index_dir = str(tmp_path / "index")
    data = json.loads(index_repository(str(tmp_path), index_dir=index_dir))
    assert data["files_processed"] == 0
    assert data["methods_indexed"] == 0


def test_ignores_non_python_files(tmp_path):
    _write(tmp_path / "script.js", _NOT_PYTHON)
    index_dir = str(tmp_path / "index")
    data = json.loads(index_repository(str(tmp_path), index_dir=index_dir))
    assert data["files_processed"] == 0
    assert data["methods_indexed"] == 0


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


def test_nonexistent_root_returns_error(tmp_path):
    bad_path = str(tmp_path / "does_not_exist")
    raw = index_repository(bad_path, index_dir=str(tmp_path / "index"))
    data = json.loads(raw)
    assert "error" in data
    assert "does_not_exist" in data["error"] or "Not a directory" in data["error"]


def test_root_is_a_file_returns_error(tmp_path):
    f = tmp_path / "notadir.py"
    f.write_text("def x(): pass")
    raw = index_repository(str(f), index_dir=str(tmp_path / "index"))
    data = json.loads(raw)
    assert "error" in data


# ---------------------------------------------------------------------------
# Tests: skip already-indexed files
# ---------------------------------------------------------------------------


def test_skip_already_indexed_by_default(tmp_path):
    _write(tmp_path / "math.py", _SIMPLE_PY)
    index_dir = str(tmp_path / "index")

    first = json.loads(index_repository(str(tmp_path), index_dir=index_dir))
    assert first["methods_indexed"] == 2

    # Second call: file already indexed, should skip
    second = json.loads(index_repository(str(tmp_path), index_dir=index_dir))
    assert second["files_processed"] == 0
    assert second["methods_indexed"] == 0


def test_force_reindex_reindexes_existing_file(tmp_path):
    _write(tmp_path / "math.py", _SIMPLE_PY)
    index_dir = str(tmp_path / "index")

    json.loads(index_repository(str(tmp_path), index_dir=index_dir))

    # Force reindex
    second = json.loads(
        index_repository(str(tmp_path), index_dir=index_dir, force_reindex=True)
    )
    assert second["files_processed"] == 1
    assert second["methods_indexed"] == 2


# ---------------------------------------------------------------------------
# Tests: excluded directories
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("excluded_dir", [
    ".venv", "venv", "__pycache__", ".git", "node_modules",
    "site-packages", "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
])
def test_excluded_dirs_are_skipped(tmp_path, excluded_dir):
    _write(tmp_path / excluded_dir / "module.py", _SIMPLE_PY)
    index_dir = str(tmp_path / "index")
    data = json.loads(index_repository(str(tmp_path), index_dir=index_dir))
    assert data["files_processed"] == 0
    assert data["methods_indexed"] == 0


def test_excluded_dir_nested_skipped(tmp_path):
    """Files inside an excluded dir nested under normal dirs are also skipped."""
    _write(tmp_path / "src" / ".venv" / "helper.py", _SIMPLE_PY)
    _write(tmp_path / "src" / "real.py", _ANOTHER_PY)
    index_dir = str(tmp_path / "index")
    data = json.loads(index_repository(str(tmp_path), index_dir=index_dir))
    assert data["files_processed"] == 1  # only real.py
    assert data["methods_indexed"] == 1  # only multiply


# ---------------------------------------------------------------------------
# Tests: recursive discovery
# ---------------------------------------------------------------------------


def test_walks_subdirectories(tmp_path):
    _write(tmp_path / "a" / "b" / "deep.py", _SIMPLE_PY)
    index_dir = str(tmp_path / "index")
    data = json.loads(index_repository(str(tmp_path), index_dir=index_dir))
    assert data["files_processed"] == 1
    assert data["methods_indexed"] == 2


def test_indexes_multiple_dirs(tmp_path):
    _write(tmp_path / "module_a" / "funcs.py", _SIMPLE_PY)
    _write(tmp_path / "module_b" / "ops.py", _ANOTHER_PY)
    index_dir = str(tmp_path / "index")
    data = json.loads(index_repository(str(tmp_path), index_dir=index_dir))
    assert data["files_processed"] == 2
    assert data["methods_indexed"] == 3


# ---------------------------------------------------------------------------
# Tests: custom index_dir
# ---------------------------------------------------------------------------


def test_custom_index_dir_is_created(tmp_path):
    _write(tmp_path / "math.py", _SIMPLE_PY)
    custom_index = tmp_path / "custom" / "idx"
    json.loads(index_repository(str(tmp_path), index_dir=str(custom_index)))
    assert custom_index.exists()


def test_two_repos_use_separate_indexes(tmp_path):
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    _write(repo_a / "math.py", _SIMPLE_PY)
    _write(repo_b / "ops.py", _ANOTHER_PY)

    idx_a = str(tmp_path / "idx_a")
    idx_b = str(tmp_path / "idx_b")

    data_a = json.loads(index_repository(str(repo_a), index_dir=idx_a))
    data_b = json.loads(index_repository(str(repo_b), index_dir=idx_b))

    assert data_a["methods_indexed"] == 2
    assert data_b["methods_indexed"] == 1


# ---------------------------------------------------------------------------
# Tests: stub filtering
# ---------------------------------------------------------------------------

_ABSTRACT_PY = """\
from abc import ABC, abstractmethod


class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

    @abstractmethod
    def perimeter(self):
        ...

    def describe(self):
        return f"I am a {type(self).__name__}"
"""

_STUB_ONLY_PY = """\
def placeholder():
    pass


def also_placeholder():
    ...


def real_function(a, b):
    return a + b
"""


def test_abstract_methods_not_indexed(tmp_path):
    """Abstract methods must be skipped; concrete methods must be indexed."""
    _write(tmp_path / "shapes.py", _ABSTRACT_PY)
    index_dir = str(tmp_path / "index")
    data = json.loads(index_repository(str(tmp_path), index_dir=index_dir))
    # Only describe() is concrete; area() and perimeter() are abstract stubs
    assert data["methods_indexed"] == 1


def test_pass_and_ellipsis_stubs_not_indexed(tmp_path):
    """pass-only and ...-only methods must be skipped during indexing."""
    _write(tmp_path / "stubs.py", _STUB_ONLY_PY)
    index_dir = str(tmp_path / "index")
    data = json.loads(index_repository(str(tmp_path), index_dir=index_dir))
    # Only real_function is non-stub
    assert data["methods_indexed"] == 1


def test_mixed_stub_and_real_count(tmp_path):
    """Files with both stubs and real methods count only real methods."""
    _write(tmp_path / "module.py", _ABSTRACT_PY)
    _write(tmp_path / "stubs.py", _STUB_ONLY_PY)
    index_dir = str(tmp_path / "index")
    data = json.loads(index_repository(str(tmp_path), index_dir=index_dir))
    # describe() from shapes.py + real_function() from stubs.py = 2
    assert data["methods_indexed"] == 2
