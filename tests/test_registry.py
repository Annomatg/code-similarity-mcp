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


def test_delete_by_id(registry):
    db_id1 = registry.add_method(_make_method("f1"), _random_embedding(1))
    db_id2 = registry.add_method(_make_method("f2"), _random_embedding(2))

    removed = registry.delete_by_id(db_id1)
    assert removed is True
    assert registry.stats()["methods"] == 1

    # Remaining method is still searchable
    results = registry.search(_random_embedding(2), top_k=5)
    names = [r["name"] for r in results]
    assert "f2" in names
    assert "f1" not in names


def test_delete_by_id_nonexistent(registry):
    result = registry.delete_by_id(99999)
    assert result is False


def test_search_respects_top_k(registry):
    for i in range(5):
        registry.add_method(_make_method(f"func_{i}"), _random_embedding(i))

    results = registry.search(_random_embedding(0), top_k=3)
    assert len(results) == 3


def test_search_skips_deleted_faiss_positions(registry):
    """After delete_by_file, orphaned FAISS positions are excluded from results."""
    registry.add_method(_make_method("f1", "a.gd"), _random_embedding(1))
    emb2 = _random_embedding(2)
    registry.add_method(_make_method("f2", "b.gd"), emb2)

    registry.delete_by_file("a.gd")

    # FAISS index retains both vectors, but id_map no longer has the deleted one
    assert registry._faiss_index.ntotal == 2
    results = registry.search(emb2, top_k=5)
    assert len(results) == 1
    assert results[0]["name"] == "f2"


def test_delete_by_id_persists(tmp_path):
    """delete_by_id is reflected after reloading the index from disk."""
    index_dir = tmp_path / "index"
    reg1 = MethodRegistry(index_dir)
    emb = _random_embedding(7)
    db_id = reg1.add_method(_make_method("to_delete"), emb)
    reg1.delete_by_id(db_id)
    reg1.close()

    reg2 = MethodRegistry(index_dir)
    results = reg2.search(emb, top_k=5)
    reg2.close()
    assert all(r["name"] != "to_delete" for r in results)
