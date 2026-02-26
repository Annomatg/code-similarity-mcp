"""Tests for feature #30: chunk embeddings stored in dedicated FAISS index.

Covers:
- Chunk FAISS index is initialized as IndexFlatIP alongside the method index
- chunk_id_map.json is created on disk after adding a chunk
- Chunk embedding can be retrieved by similarity search (search_chunks)
- Chunk FAISS index is persisted to disk (chunks.faiss exists)
- Chunk FAISS index and chunk_id_map.json are reloaded correctly across sessions
- Chunk FAISS positions map to correct DB chunk IDs
- Deleted chunk positions are removed from chunk_id_map
- search_chunks respects allowed_chunk_ids filter
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from code_similarity_mcp.index.registry import EMBEDDING_DIM, MethodRegistry
from code_similarity_mcp.parser.base import ChunkInfo, MethodInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_embedding(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.random(EMBEDDING_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def _make_method(name: str = "func", file_path: str = "test.py") -> MethodInfo:
    return MethodInfo(
        file_path=file_path,
        language="python",
        name=name,
        parameters=["a", "b"],
        return_type=None,
        body_code=f"def {name}(a, b):\n    return a + b",
        normalized_code="def FUNC_NAME(v1, v2):\n    return v1 + v2",
        start_line=1,
        end_line=2,
        dependencies=[],
    )


def _make_chunk_info(
    function_id: int = 1,
    chunk_index: int = 0,
    depends_on: list | None = None,
    file_path: str = "test.py",
) -> ChunkInfo:
    return ChunkInfo(
        chunk_index=chunk_index,
        statement_start=0,
        statement_end=2,
        statement_indices=[0, 1, 2],
        function_name="my_func",
        file_path=file_path,
        function_id=function_id,
        depends_on_chunks=depends_on or [],
        depended_on_by_chunks=[],
    )


@pytest.fixture
def registry(tmp_path):
    reg = MethodRegistry(tmp_path / "index")
    yield reg
    reg.close()


@pytest.fixture
def index_dir(tmp_path):
    return tmp_path / "index"


# ---------------------------------------------------------------------------
# Step 1: Initialize a chunk FAISS index alongside the method index
# ---------------------------------------------------------------------------

class TestChunkFaissInitialization:
    def test_chunk_index_initialized_on_creation(self, registry):
        """_chunks_index must be an initialized FAISS IndexFlatIP."""
        import faiss
        assert isinstance(registry._chunks_index, faiss.IndexFlatIP)

    def test_chunk_index_starts_empty(self, registry):
        """Chunk FAISS index must be empty before any chunks are added."""
        assert registry._chunks_index.ntotal == 0

    def test_chunk_id_map_starts_empty(self, registry):
        """_chunk_id_map must be an empty dict before any chunks are added."""
        assert registry._chunk_id_map == {}

    def test_chunk_index_dimension_is_384(self, registry):
        """Chunk FAISS index must have the same dimension as method index (384)."""
        assert registry._chunks_index.d == EMBEDDING_DIM

    def test_chunk_faiss_path_set_correctly(self, index_dir):
        """_chunks_faiss_path must point to chunks.faiss inside the index dir."""
        reg = MethodRegistry(index_dir)
        assert reg._chunks_faiss_path == index_dir / "chunks.faiss"
        reg.close()

    def test_chunk_id_map_path_set_correctly(self, index_dir):
        """_chunk_id_map_path must point to chunk_id_map.json inside the index dir."""
        reg = MethodRegistry(index_dir)
        assert reg._chunk_id_map_path == index_dir / "chunk_id_map.json"
        reg.close()

    def test_method_index_and_chunk_index_are_separate(self, registry):
        """Method index and chunk index must be distinct FAISS objects."""
        assert registry._faiss_index is not registry._chunks_index


# ---------------------------------------------------------------------------
# Step 2: Add chunk_id_map.json file
# ---------------------------------------------------------------------------

class TestChunkIdMapFile:
    def test_chunks_faiss_file_created_after_add_chunk(self, index_dir):
        """chunks.faiss must exist on disk after the first add_chunk call."""
        reg = MethodRegistry(index_dir)
        method = _make_method()
        db_id = reg.add_method(method, _random_embedding(0))
        chunk = _make_chunk_info(function_id=db_id)
        reg.add_chunk(chunk, _random_embedding(1))
        reg.close()

        assert (index_dir / "chunks.faiss").exists()

    def test_chunk_id_map_file_created_after_add_chunk(self, index_dir):
        """chunk_id_map.json must exist on disk after the first add_chunk call."""
        reg = MethodRegistry(index_dir)
        method = _make_method()
        db_id = reg.add_method(method, _random_embedding(0))
        chunk = _make_chunk_info(function_id=db_id)
        reg.add_chunk(chunk, _random_embedding(1))
        reg.close()

        assert (index_dir / "chunk_id_map.json").exists()

    def test_chunk_id_map_json_contains_correct_mapping(self, index_dir):
        """chunk_id_map.json must map faiss_pos (str) to chunk DB id (int)."""
        reg = MethodRegistry(index_dir)
        method = _make_method()
        db_id = reg.add_method(method, _random_embedding(0))
        chunk = _make_chunk_info(function_id=db_id)
        chunk_db_id = reg.add_chunk(chunk, _random_embedding(1))
        reg.close()

        id_map = json.loads((index_dir / "chunk_id_map.json").read_text())
        # Keys are stringified ints (JSON object keys are always strings)
        assert "0" in id_map
        assert id_map["0"] == chunk_db_id

    def test_chunk_id_map_updated_for_each_chunk(self, index_dir):
        """chunk_id_map.json must have an entry for each added chunk."""
        reg = MethodRegistry(index_dir)
        method = _make_method()
        db_id = reg.add_method(method, _random_embedding(0))

        ids = []
        for i in range(3):
            chunk = _make_chunk_info(function_id=db_id, chunk_index=i)
            chunk_db_id = reg.add_chunk(chunk, _random_embedding(i + 1))
            ids.append(chunk_db_id)
        reg.close()

        id_map = json.loads((index_dir / "chunk_id_map.json").read_text())
        assert len(id_map) == 3
        assert sorted(id_map.values()) == sorted(ids)

    def test_method_index_not_affected_by_chunks(self, index_dir):
        """Adding chunks must not change the method FAISS index or id_map.json."""
        reg = MethodRegistry(index_dir)
        method = _make_method()
        db_id = reg.add_method(method, _random_embedding(0))
        method_faiss_total = reg._faiss_index.ntotal
        method_id_map_len = len(reg._id_map)

        chunk = _make_chunk_info(function_id=db_id)
        reg.add_chunk(chunk, _random_embedding(1))

        assert reg._faiss_index.ntotal == method_faiss_total
        assert len(reg._id_map) == method_id_map_len
        reg.close()


# ---------------------------------------------------------------------------
# Step 3: Insert and retrieve by similarity search
# ---------------------------------------------------------------------------

class TestChunkSearchRetrieval:
    def test_add_chunk_increments_faiss_total(self, registry):
        """Each add_chunk call must increment the chunk FAISS index total by 1."""
        method = _make_method()
        db_id = registry.add_method(method, _random_embedding(0))

        assert registry._chunks_index.ntotal == 0
        for i in range(3):
            chunk = _make_chunk_info(function_id=db_id, chunk_index=i)
            registry.add_chunk(chunk, _random_embedding(i + 1))
            assert registry._chunks_index.ntotal == i + 1

    def test_search_chunks_returns_empty_when_no_chunks(self, registry):
        """search_chunks must return [] when the index is empty."""
        result = registry.search_chunks(_random_embedding(99), top_k=5)
        assert result == []

    def test_search_chunks_finds_inserted_chunk(self, registry):
        """search_chunks must return the chunk that was just inserted."""
        method = _make_method()
        db_id = registry.add_method(method, _random_embedding(0))
        emb = _random_embedding(42)
        chunk = _make_chunk_info(function_id=db_id)
        chunk_db_id = registry.add_chunk(chunk, emb)

        results = registry.search_chunks(emb, top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == chunk_db_id

    def test_search_chunks_returns_embedding_score(self, registry):
        """search_chunks result must include an 'embedding_score' field."""
        method = _make_method()
        db_id = registry.add_method(method, _random_embedding(0))
        emb = _random_embedding(7)
        chunk = _make_chunk_info(function_id=db_id)
        registry.add_chunk(chunk, emb)

        results = registry.search_chunks(emb, top_k=1)
        assert "embedding_score" in results[0]
        assert isinstance(results[0]["embedding_score"], float)

    def test_search_chunks_self_similarity_is_near_1(self, registry):
        """Querying with the same embedding used for insertion should score ~1.0."""
        method = _make_method()
        db_id = registry.add_method(method, _random_embedding(0))
        emb = _random_embedding(5)
        chunk = _make_chunk_info(function_id=db_id)
        registry.add_chunk(chunk, emb)

        results = registry.search_chunks(emb, top_k=1)
        assert abs(results[0]["embedding_score"] - 1.0) < 1e-4

    def test_search_chunks_top_k_respected(self, registry):
        """search_chunks must return at most top_k results."""
        method = _make_method()
        db_id = registry.add_method(method, _random_embedding(0))
        for i in range(5):
            chunk = _make_chunk_info(function_id=db_id, chunk_index=i)
            registry.add_chunk(chunk, _random_embedding(i + 1))

        results = registry.search_chunks(_random_embedding(99), top_k=3)
        assert len(results) <= 3

    def test_search_chunks_most_similar_first(self, registry):
        """search_chunks must return results ordered by descending similarity."""
        method = _make_method()
        db_id = registry.add_method(method, _random_embedding(0))

        # Create a query embedding
        query = _random_embedding(100)

        # Insert two chunks with known similarity to query
        # Chunk A: same as query (score ~1.0)
        chunk_a = _make_chunk_info(function_id=db_id, chunk_index=0)
        registry.add_chunk(chunk_a, query.copy())

        # Chunk B: orthogonal-ish (lower score)
        rng = np.random.default_rng(999)
        other = rng.random(EMBEDDING_DIM).astype(np.float32)
        other /= np.linalg.norm(other)
        chunk_b = _make_chunk_info(function_id=db_id, chunk_index=1)
        registry.add_chunk(chunk_b, other)

        results = registry.search_chunks(query, top_k=2)
        assert len(results) == 2
        assert results[0]["embedding_score"] >= results[1]["embedding_score"]

    def test_search_chunks_allowed_ids_filter(self, registry):
        """search_chunks with allowed_chunk_ids must exclude chunks not in the set."""
        method = _make_method()
        db_id = registry.add_method(method, _random_embedding(0))

        emb = _random_embedding(42)
        chunk_a = _make_chunk_info(function_id=db_id, chunk_index=0)
        id_a = registry.add_chunk(chunk_a, emb.copy())

        chunk_b = _make_chunk_info(function_id=db_id, chunk_index=1)
        id_b = registry.add_chunk(chunk_b, _random_embedding(55))

        # Only allow chunk_b
        results = registry.search_chunks(emb, top_k=5, allowed_chunk_ids={id_b})
        returned_ids = {r["id"] for r in results}
        assert id_a not in returned_ids
        assert id_b in returned_ids

    def test_search_chunks_empty_allowed_ids_returns_empty(self, registry):
        """search_chunks with an empty allowed_chunk_ids set must return []."""
        method = _make_method()
        db_id = registry.add_method(method, _random_embedding(0))
        chunk = _make_chunk_info(function_id=db_id)
        registry.add_chunk(chunk, _random_embedding(1))

        results = registry.search_chunks(_random_embedding(1), top_k=5, allowed_chunk_ids=set())
        assert results == []

    def test_get_chunk_embedding_returns_stored_vector(self, registry):
        """get_chunk_embedding must reconstruct the vector that was stored."""
        method = _make_method()
        db_id = registry.add_method(method, _random_embedding(0))
        emb = _random_embedding(3)
        chunk = _make_chunk_info(function_id=db_id)
        registry.add_chunk(chunk, emb)

        # faiss_pos for first chunk is 0
        reconstructed = registry.get_chunk_embedding(0)
        assert reconstructed is not None
        assert np.allclose(reconstructed, emb, atol=1e-5)

    def test_get_chunk_embedding_returns_none_for_invalid_pos(self, registry):
        """get_chunk_embedding must return None for out-of-range positions."""
        assert registry.get_chunk_embedding(0) is None
        assert registry.get_chunk_embedding(-1) is None
        assert registry.get_chunk_embedding(999) is None


# ---------------------------------------------------------------------------
# Step 4: Verify persistence and reload across sessions
# ---------------------------------------------------------------------------

class TestChunkFaissPersistence:
    def test_chunks_faiss_persisted_to_disk(self, index_dir):
        """chunks.faiss must exist on disk after add_chunk."""
        reg = MethodRegistry(index_dir)
        method = _make_method()
        db_id = reg.add_method(method, _random_embedding(0))
        chunk = _make_chunk_info(function_id=db_id)
        reg.add_chunk(chunk, _random_embedding(1))
        reg.close()

        assert (index_dir / "chunks.faiss").exists()
        assert (index_dir / "chunk_id_map.json").exists()

    def test_chunk_index_reloaded_across_sessions(self, index_dir):
        """Chunk FAISS index and id_map must survive a close/reopen cycle."""
        # Session 1: insert a chunk
        reg1 = MethodRegistry(index_dir)
        method = _make_method()
        db_id = reg1.add_method(method, _random_embedding(0))
        emb = _random_embedding(7)
        chunk = _make_chunk_info(function_id=db_id)
        chunk_db_id = reg1.add_chunk(chunk, emb)
        reg1.close()

        # Session 2: verify the chunk FAISS index was reloaded
        reg2 = MethodRegistry(index_dir)
        assert reg2._chunks_index.ntotal == 1
        assert 0 in reg2._chunk_id_map
        assert reg2._chunk_id_map[0] == chunk_db_id
        reg2.close()

    def test_search_chunks_works_after_reload(self, index_dir):
        """search_chunks must find the original chunk after close/reopen."""
        # Session 1
        reg1 = MethodRegistry(index_dir)
        method = _make_method()
        db_id = reg1.add_method(method, _random_embedding(0))
        emb = _random_embedding(42)
        chunk = _make_chunk_info(function_id=db_id)
        chunk_db_id = reg1.add_chunk(chunk, emb)
        reg1.close()

        # Session 2
        reg2 = MethodRegistry(index_dir)
        results = reg2.search_chunks(emb, top_k=1)
        reg2.close()

        assert len(results) == 1
        assert results[0]["id"] == chunk_db_id
        assert abs(results[0]["embedding_score"] - 1.0) < 1e-4

    def test_chunk_id_map_reloaded_with_correct_types(self, index_dir):
        """After reload, _chunk_id_map keys must be int (not str)."""
        reg1 = MethodRegistry(index_dir)
        method = _make_method()
        db_id = reg1.add_method(method, _random_embedding(0))
        chunk = _make_chunk_info(function_id=db_id)
        reg1.add_chunk(chunk, _random_embedding(1))
        reg1.close()

        reg2 = MethodRegistry(index_dir)
        for k in reg2._chunk_id_map:
            assert isinstance(k, int), f"Expected int key, got {type(k)}: {k!r}"
        reg2.close()

    def test_multiple_chunks_persist_and_reload(self, index_dir):
        """All inserted chunks must be present after close/reopen."""
        reg1 = MethodRegistry(index_dir)
        method = _make_method()
        db_id = reg1.add_method(method, _random_embedding(0))

        chunk_ids = []
        embeddings = []
        for i in range(4):
            emb = _random_embedding(i + 10)
            chunk = _make_chunk_info(function_id=db_id, chunk_index=i)
            cid = reg1.add_chunk(chunk, emb)
            chunk_ids.append(cid)
            embeddings.append(emb)
        reg1.close()

        reg2 = MethodRegistry(index_dir)
        assert reg2._chunks_index.ntotal == 4
        assert len(reg2._chunk_id_map) == 4
        assert sorted(reg2._chunk_id_map.values()) == sorted(chunk_ids)
        reg2.close()

    def test_delete_chunk_removes_from_id_map_and_persists(self, index_dir):
        """After deleting chunks via delete_chunks_by_function, id_map is updated on disk."""
        reg1 = MethodRegistry(index_dir)
        method = _make_method()
        db_id = reg1.add_method(method, _random_embedding(0))
        for i in range(2):
            chunk = _make_chunk_info(function_id=db_id, chunk_index=i)
            reg1.add_chunk(chunk, _random_embedding(i + 1))

        reg1.delete_chunks_by_function(db_id)
        assert len(reg1._chunk_id_map) == 0
        reg1.close()

        # Reload and verify id_map is also empty on disk
        reg2 = MethodRegistry(index_dir)
        assert len(reg2._chunk_id_map) == 0
        reg2.close()

    def test_chunk_index_is_separate_from_method_index_on_disk(self, index_dir):
        """chunks.faiss and index.faiss must be separate files on disk."""
        reg = MethodRegistry(index_dir)
        method = _make_method()
        db_id = reg.add_method(method, _random_embedding(0))
        chunk = _make_chunk_info(function_id=db_id)
        reg.add_chunk(chunk, _random_embedding(1))
        reg.close()

        assert (index_dir / "index.faiss").exists()
        assert (index_dir / "chunks.faiss").exists()
        # They must be different files
        assert (index_dir / "index.faiss") != (index_dir / "chunks.faiss")

    def test_method_search_unaffected_after_chunks_added(self, index_dir):
        """Method search must still work correctly after chunks are added and reloaded."""
        reg1 = MethodRegistry(index_dir)
        method = _make_method()
        emb = _random_embedding(10)
        db_id = reg1.add_method(method, emb)
        for i in range(3):
            chunk = _make_chunk_info(function_id=db_id, chunk_index=i)
            reg1.add_chunk(chunk, _random_embedding(i + 20))
        reg1.close()

        reg2 = MethodRegistry(index_dir)
        results = reg2.search(emb, top_k=1)
        reg2.close()

        assert len(results) == 1
        assert results[0]["id"] == db_id
