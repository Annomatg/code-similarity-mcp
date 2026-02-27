"""Tests for feature #31: Chunk id_map updated correctly on insert and delete.

Covers:
- Insert 3 chunks and verify chunk_id_map has 3 entries with correct FAISS positions
- Delete chunk at position 1 and verify it is removed from the map
- Verify FAISS positions for remaining chunks are unchanged after a deletion
- Reload the index from disk and verify the id_map is consistent
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
    function_id: int,
    chunk_index: int = 0,
    file_path: str = "test.py",
) -> ChunkInfo:
    return ChunkInfo(
        chunk_index=chunk_index,
        statement_start=chunk_index * 3,
        statement_end=chunk_index * 3 + 2,
        statement_indices=list(range(chunk_index * 3, chunk_index * 3 + 3)),
        function_name="my_func",
        file_path=file_path,
        function_id=function_id,
        depends_on_chunks=[],
        depended_on_by_chunks=[],
    )


@pytest.fixture
def index_dir(tmp_path):
    return tmp_path / "index"


@pytest.fixture
def registry(index_dir):
    reg = MethodRegistry(index_dir)
    yield reg
    reg.close()


# ---------------------------------------------------------------------------
# Step 1: Insert 3 chunks and verify chunk_id_map has 3 entries
# ---------------------------------------------------------------------------


class TestChunkIdMapInsert:
    def test_three_chunks_produce_three_map_entries(self, registry):
        """Inserting 3 chunks must create 3 entries in _chunk_id_map."""
        method = _make_method()
        db_id = registry.add_method(method, _random_embedding(0))

        for i in range(3):
            chunk = _make_chunk_info(function_id=db_id, chunk_index=i)
            registry.add_chunk(chunk, _random_embedding(i + 1))

        assert len(registry._chunk_id_map) == 3

    def test_faiss_positions_are_sequential(self, registry):
        """FAISS positions must be 0, 1, 2 for chunks inserted in order."""
        method = _make_method()
        db_id = registry.add_method(method, _random_embedding(0))

        for i in range(3):
            chunk = _make_chunk_info(function_id=db_id, chunk_index=i)
            registry.add_chunk(chunk, _random_embedding(i + 1))

        assert set(registry._chunk_id_map.keys()) == {0, 1, 2}

    def test_map_values_are_correct_db_ids(self, registry):
        """Each map entry must map its FAISS position to the correct chunk DB id."""
        method = _make_method()
        db_id = registry.add_method(method, _random_embedding(0))

        inserted_ids = []
        for i in range(3):
            chunk = _make_chunk_info(function_id=db_id, chunk_index=i)
            chunk_db_id = registry.add_chunk(chunk, _random_embedding(i + 1))
            inserted_ids.append(chunk_db_id)

        # FAISS position i must map to the (i+1)-th inserted chunk id
        assert registry._chunk_id_map[0] == inserted_ids[0]
        assert registry._chunk_id_map[1] == inserted_ids[1]
        assert registry._chunk_id_map[2] == inserted_ids[2]

    def test_chunk_id_map_written_to_disk_after_each_insert(self, index_dir):
        """chunk_id_map.json must grow by 1 entry after each add_chunk call."""
        reg = MethodRegistry(index_dir)
        method = _make_method()
        db_id = reg.add_method(method, _random_embedding(0))

        for i in range(3):
            chunk = _make_chunk_info(function_id=db_id, chunk_index=i)
            reg.add_chunk(chunk, _random_embedding(i + 1))
            on_disk = json.loads((index_dir / "chunk_id_map.json").read_text())
            assert len(on_disk) == i + 1, f"Expected {i+1} entries after insert {i}"

        reg.close()

    def test_faiss_ntotal_matches_map_size_after_inserts(self, registry):
        """_chunks_index.ntotal must equal len(_chunk_id_map) after 3 inserts."""
        method = _make_method()
        db_id = registry.add_method(method, _random_embedding(0))

        for i in range(3):
            chunk = _make_chunk_info(function_id=db_id, chunk_index=i)
            registry.add_chunk(chunk, _random_embedding(i + 1))

        assert registry._chunks_index.ntotal == len(registry._chunk_id_map) == 3


# ---------------------------------------------------------------------------
# Step 2: Delete chunk at position 1 and verify it is removed from the map
# ---------------------------------------------------------------------------


class TestChunkIdMapDelete:
    def _insert_three_chunks(self, registry):
        """Helper: insert 3 chunks and return (function_db_id, [chunk_db_id_0, 1, 2])."""
        method = _make_method()
        db_id = registry.add_method(method, _random_embedding(0))
        chunk_ids = []
        for i in range(3):
            chunk = _make_chunk_info(function_id=db_id, chunk_index=i)
            cid = registry.add_chunk(chunk, _random_embedding(i + 1))
            chunk_ids.append(cid)
        return db_id, chunk_ids

    def test_delete_middle_chunk_removes_from_map(self, registry):
        """Deleting the chunk at FAISS position 1 must remove key 1 from the map."""
        _, chunk_ids = self._insert_three_chunks(registry)

        # chunk at FAISS position 1 is the second chunk inserted
        removed = registry.delete_chunk_by_id(chunk_ids[1])

        assert removed is True
        assert 1 not in registry._chunk_id_map

    def test_delete_middle_chunk_leaves_map_with_two_entries(self, registry):
        """After deleting position 1, _chunk_id_map must have exactly 2 entries."""
        _, chunk_ids = self._insert_three_chunks(registry)
        registry.delete_chunk_by_id(chunk_ids[1])

        assert len(registry._chunk_id_map) == 2

    def test_delete_nonexistent_chunk_returns_false(self, registry):
        """delete_chunk_by_id must return False for a non-existent chunk id."""
        result = registry.delete_chunk_by_id(99999)
        assert result is False

    def test_delete_nonexistent_chunk_does_not_change_map(self, registry):
        """Calling delete_chunk_by_id with an invalid id must leave the map unchanged."""
        method = _make_method()
        db_id = registry.add_method(method, _random_embedding(0))
        chunk = _make_chunk_info(function_id=db_id, chunk_index=0)
        registry.add_chunk(chunk, _random_embedding(1))

        before = dict(registry._chunk_id_map)
        registry.delete_chunk_by_id(99999)
        assert registry._chunk_id_map == before

    def test_delete_chunk_removes_from_sqlite(self, registry):
        """After delete_chunk_by_id, the chunk must not exist in the database."""
        _, chunk_ids = self._insert_three_chunks(registry)
        registry.delete_chunk_by_id(chunk_ids[1])

        row = registry.get_chunk_by_id(chunk_ids[1])
        assert row is None

    def test_delete_chunk_map_written_to_disk(self, index_dir):
        """chunk_id_map.json must be updated on disk after delete_chunk_by_id."""
        reg = MethodRegistry(index_dir)
        method = _make_method()
        db_id = reg.add_method(method, _random_embedding(0))
        chunk_ids = []
        for i in range(3):
            chunk = _make_chunk_info(function_id=db_id, chunk_index=i)
            cid = reg.add_chunk(chunk, _random_embedding(i + 1))
            chunk_ids.append(cid)

        reg.delete_chunk_by_id(chunk_ids[1])
        reg.close()

        on_disk = json.loads((index_dir / "chunk_id_map.json").read_text())
        assert "1" not in on_disk
        assert len(on_disk) == 2


# ---------------------------------------------------------------------------
# Step 3: Verify FAISS positions for remaining chunks are unchanged
# ---------------------------------------------------------------------------


class TestChunkIdMapPositionsStable:
    def test_remaining_positions_unchanged_after_middle_delete(self, registry):
        """FAISS positions 0 and 2 must remain in the map after deleting position 1."""
        method = _make_method()
        db_id = registry.add_method(method, _random_embedding(0))

        chunk_ids = []
        for i in range(3):
            chunk = _make_chunk_info(function_id=db_id, chunk_index=i)
            cid = registry.add_chunk(chunk, _random_embedding(i + 1))
            chunk_ids.append(cid)

        registry.delete_chunk_by_id(chunk_ids[1])

        # Positions 0 and 2 must still exist
        assert 0 in registry._chunk_id_map
        assert 2 in registry._chunk_id_map

    def test_remaining_db_ids_unchanged_after_middle_delete(self, registry):
        """After deleting position 1, positions 0 and 2 still map to the original chunk ids."""
        method = _make_method()
        db_id = registry.add_method(method, _random_embedding(0))

        chunk_ids = []
        for i in range(3):
            chunk = _make_chunk_info(function_id=db_id, chunk_index=i)
            cid = registry.add_chunk(chunk, _random_embedding(i + 1))
            chunk_ids.append(cid)

        registry.delete_chunk_by_id(chunk_ids[1])

        assert registry._chunk_id_map[0] == chunk_ids[0]
        assert registry._chunk_id_map[2] == chunk_ids[2]

    def test_embeddings_still_retrievable_after_middle_delete(self, registry):
        """Embeddings at positions 0 and 2 must still be reconstructible after deleting position 1."""
        method = _make_method()
        db_id = registry.add_method(method, _random_embedding(0))

        embeddings = []
        chunk_ids = []
        for i in range(3):
            emb = _random_embedding(i + 10)
            chunk = _make_chunk_info(function_id=db_id, chunk_index=i)
            cid = registry.add_chunk(chunk, emb)
            chunk_ids.append(cid)
            embeddings.append(emb)

        registry.delete_chunk_by_id(chunk_ids[1])

        # Positions 0 and 2 must still return correct embeddings
        recon_0 = registry.get_chunk_embedding(0)
        recon_2 = registry.get_chunk_embedding(2)
        assert recon_0 is not None
        assert recon_2 is not None
        assert np.allclose(recon_0, embeddings[0], atol=1e-5)
        assert np.allclose(recon_2, embeddings[2], atol=1e-5)

    def test_search_skips_deleted_chunk_position(self, registry):
        """After delete_chunk_by_id, search_chunks must not return the deleted chunk."""
        method = _make_method()
        db_id = registry.add_method(method, _random_embedding(0))

        chunk_ids = []
        embeddings = []
        for i in range(3):
            emb = _random_embedding(i + 20)
            chunk = _make_chunk_info(function_id=db_id, chunk_index=i)
            cid = registry.add_chunk(chunk, emb)
            chunk_ids.append(cid)
            embeddings.append(emb)

        # Delete chunk at FAISS position 1 (chunk_ids[1])
        registry.delete_chunk_by_id(chunk_ids[1])

        # Searching with the deleted chunk's embedding must not return it
        results = registry.search_chunks(embeddings[1], top_k=5)
        returned_ids = {r["id"] for r in results}
        assert chunk_ids[1] not in returned_ids

    def test_remaining_chunks_still_searchable_after_middle_delete(self, registry):
        """Chunks at positions 0 and 2 must still appear in search results."""
        method = _make_method()
        db_id = registry.add_method(method, _random_embedding(0))

        embeddings = []
        chunk_ids = []
        for i in range(3):
            emb = _random_embedding(i + 30)
            chunk = _make_chunk_info(function_id=db_id, chunk_index=i)
            cid = registry.add_chunk(chunk, emb)
            chunk_ids.append(cid)
            embeddings.append(emb)

        registry.delete_chunk_by_id(chunk_ids[1])

        # Chunk 0 still findable
        results_0 = registry.search_chunks(embeddings[0], top_k=5)
        returned_ids = {r["id"] for r in results_0}
        assert chunk_ids[0] in returned_ids

        # Chunk 2 still findable
        results_2 = registry.search_chunks(embeddings[2], top_k=5)
        returned_ids = {r["id"] for r in results_2}
        assert chunk_ids[2] in returned_ids


# ---------------------------------------------------------------------------
# Step 4: Reload the index from disk and verify the id_map is consistent
# ---------------------------------------------------------------------------


class TestChunkIdMapReloadConsistency:
    def test_map_consistent_after_three_inserts_and_reload(self, index_dir):
        """After inserting 3 chunks and reloading, id_map must have 3 correct entries."""
        reg1 = MethodRegistry(index_dir)
        method = _make_method()
        db_id = reg1.add_method(method, _random_embedding(0))

        chunk_ids = []
        for i in range(3):
            chunk = _make_chunk_info(function_id=db_id, chunk_index=i)
            cid = reg1.add_chunk(chunk, _random_embedding(i + 1))
            chunk_ids.append(cid)
        reg1.close()

        reg2 = MethodRegistry(index_dir)
        assert len(reg2._chunk_id_map) == 3
        assert reg2._chunk_id_map[0] == chunk_ids[0]
        assert reg2._chunk_id_map[1] == chunk_ids[1]
        assert reg2._chunk_id_map[2] == chunk_ids[2]
        reg2.close()

    def test_map_consistent_after_delete_and_reload(self, index_dir):
        """After deleting position 1 and reloading, id_map must contain only 0 and 2."""
        reg1 = MethodRegistry(index_dir)
        method = _make_method()
        db_id = reg1.add_method(method, _random_embedding(0))

        chunk_ids = []
        for i in range(3):
            chunk = _make_chunk_info(function_id=db_id, chunk_index=i)
            cid = reg1.add_chunk(chunk, _random_embedding(i + 1))
            chunk_ids.append(cid)

        reg1.delete_chunk_by_id(chunk_ids[1])
        reg1.close()

        reg2 = MethodRegistry(index_dir)
        assert len(reg2._chunk_id_map) == 2
        assert 0 in reg2._chunk_id_map
        assert 1 not in reg2._chunk_id_map
        assert 2 in reg2._chunk_id_map
        assert reg2._chunk_id_map[0] == chunk_ids[0]
        assert reg2._chunk_id_map[2] == chunk_ids[2]
        reg2.close()

    def test_map_keys_are_int_after_reload(self, index_dir):
        """After reload, _chunk_id_map keys must be int (not str) for all entries."""
        reg1 = MethodRegistry(index_dir)
        method = _make_method()
        db_id = reg1.add_method(method, _random_embedding(0))
        for i in range(3):
            chunk = _make_chunk_info(function_id=db_id, chunk_index=i)
            reg1.add_chunk(chunk, _random_embedding(i + 1))
        reg1.close()

        reg2 = MethodRegistry(index_dir)
        for k in reg2._chunk_id_map:
            assert isinstance(k, int), f"Expected int key, got {type(k)}: {k!r}"
        reg2.close()

    def test_search_consistent_after_delete_and_reload(self, index_dir):
        """search_chunks after reload must only return the two non-deleted chunks."""
        reg1 = MethodRegistry(index_dir)
        method = _make_method()
        db_id = reg1.add_method(method, _random_embedding(0))

        embeddings = []
        chunk_ids = []
        for i in range(3):
            emb = _random_embedding(i + 40)
            chunk = _make_chunk_info(function_id=db_id, chunk_index=i)
            cid = reg1.add_chunk(chunk, emb)
            chunk_ids.append(cid)
            embeddings.append(emb)

        reg1.delete_chunk_by_id(chunk_ids[1])
        reg1.close()

        reg2 = MethodRegistry(index_dir)
        # Deleted chunk must not appear
        results = reg2.search_chunks(embeddings[1], top_k=5)
        returned_ids = {r["id"] for r in results}
        assert chunk_ids[1] not in returned_ids

        # Remaining chunks must still be found
        results_0 = reg2.search_chunks(embeddings[0], top_k=5)
        assert chunk_ids[0] in {r["id"] for r in results_0}

        results_2 = reg2.search_chunks(embeddings[2], top_k=5)
        assert chunk_ids[2] in {r["id"] for r in results_2}
        reg2.close()

    def test_faiss_ntotal_unchanged_after_delete_and_reload(self, index_dir):
        """FAISS ntotal must remain 3 after deleting a chunk (FAISS has no removal).

        Deletion only removes the id_map entry; the FAISS vector slot stays but
        is unreachable via the map.
        """
        reg1 = MethodRegistry(index_dir)
        method = _make_method()
        db_id = reg1.add_method(method, _random_embedding(0))
        chunk_ids = []
        for i in range(3):
            chunk = _make_chunk_info(function_id=db_id, chunk_index=i)
            cid = reg1.add_chunk(chunk, _random_embedding(i + 1))
            chunk_ids.append(cid)

        reg1.delete_chunk_by_id(chunk_ids[1])
        reg1.close()

        reg2 = MethodRegistry(index_dir)
        # FAISS index still has 3 vectors (cannot delete), but map has only 2
        assert reg2._chunks_index.ntotal == 3
        assert len(reg2._chunk_id_map) == 2
        reg2.close()
