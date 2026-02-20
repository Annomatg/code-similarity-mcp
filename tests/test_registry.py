"""Integration tests for MethodRegistry (FAISS + SQLite)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from code_similarity_mcp.index.registry import MethodRegistry
from code_similarity_mcp.parser.base import MethodInfo


def _make_method(name="test_func", file_path="test.gd", params=None):
    return MethodInfo(
        file_path=file_path,
        language="gdscript",
        name=name,
        parameters=params or ["a", "b"],
        return_type=None,
        body_code=f"func {name}(a, b):\n    return a + b",
        normalized_code=f"func {name}(a, b):\n return a + b",
        start_line=1,
        end_line=2,
        dependencies=[],
    )


def _random_embedding(seed=0):
    rng = np.random.default_rng(seed)
    vec = rng.random(384).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


@pytest.fixture
def registry(tmp_path):
    reg = MethodRegistry(tmp_path / "index")
    yield reg
    reg.close()


def test_add_and_search(registry):
    method = _make_method("my_func")
    emb = _random_embedding(42)
    db_id = registry.add_method(method, emb)
    assert db_id > 0

    results = registry.search(emb, top_k=1)
    assert len(results) == 1
    assert results[0]["name"] == "my_func"
    assert results[0]["embedding_score"] > 0.99


def test_stats(registry):
    registry.add_method(_make_method("f1", "a.gd"), _random_embedding(1))
    registry.add_method(_make_method("f2", "b.gd"), _random_embedding(2))
    s = registry.stats()
    assert s["methods"] == 2
    assert s["files"] == 2


def test_delete_by_file(registry):
    registry.add_method(_make_method("f1", "a.gd"), _random_embedding(1))
    registry.add_method(_make_method("f2", "a.gd"), _random_embedding(2))
    registry.add_method(_make_method("f3", "b.gd"), _random_embedding(3))

    removed = registry.delete_by_file("a.gd")
    assert removed == 2
    assert registry.stats()["methods"] == 1


def test_get_by_file(registry):
    registry.add_method(_make_method("f1", "target.gd"), _random_embedding(0))
    registry.add_method(_make_method("f2", "other.gd"), _random_embedding(1))

    results = registry.get_by_file("target.gd")
    assert len(results) == 1
    assert results[0]["name"] == "f1"


def test_persistence(tmp_path):
    """Index persists across MethodRegistry instances."""
    index_dir = tmp_path / "index"

    reg1 = MethodRegistry(index_dir)
    emb = _random_embedding(99)
    reg1.add_method(_make_method("persist_func"), emb)
    reg1.close()

    reg2 = MethodRegistry(index_dir)
    results = reg2.search(emb, top_k=1)
    reg2.close()

    assert len(results) == 1
    assert results[0]["name"] == "persist_func"


def test_search_empty_index(registry):
    results = registry.search(_random_embedding(), top_k=5)
    assert results == []
