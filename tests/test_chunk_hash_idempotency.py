"""Tests for feature #34: idempotent chunk hashing.

Unchanged functions must produce identical ChunkInfo.code_hash values on every
call to embed_chunks.  A single-character change must produce a different hash
for the affected chunk.
"""

from __future__ import annotations

import hashlib

import pytest

from code_similarity_mcp.embeddings.generator import EmbeddingGenerator
from code_similarity_mcp.parser.base import annotate_chunks, embed_chunks, group_into_chunks
from code_similarity_mcp.parser.python import build_dependency_graph, get_flat_statements


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_function(source: str, generator: EmbeddingGenerator) -> list:
    """Run the full chunking pipeline and return annotated ChunkInfo list.

    embed_chunks populates chunk.code_hash on every ChunkInfo in the list.
    """
    statements = get_flat_statements(source)
    graph = build_dependency_graph(source)
    raw_chunks = group_into_chunks(graph, max_statements_per_chunk=10)
    if not raw_chunks:
        return []
    annotated = annotate_chunks(raw_chunks, graph, function_name="test_func", file_path="f.py")
    embed_chunks(annotated, source, statements, generator)
    return annotated


def _make_large_function(n_stmts: int = 35) -> str:
    """Return a Python function source with *n_stmts* independent assignments."""
    lines = ["def big_func():"]
    for i in range(n_stmts):
        lines.append(f"    x_{i} = {i}")
    lines.append("    return x_0")
    return "\n".join(lines) + "\n"


def _make_function_with_calls(n_stmts: int = 35) -> str:
    """Return a Python function source where each statement calls a uniquely named
    external function.  External function names survive normalization (only
    assignment targets are renamed), so changing one function name produces a
    different hash for the chunk that contains it.
    """
    lines = ["def big_func():"]
    # Each statement calls a distinctly named external function so that the
    # normalizer preserves the structural difference between them.
    for i in range(n_stmts):
        lines.append(f"    r_{i} = external_op_{i}()")
    lines.append("    return r_0")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def generator():
    return EmbeddingGenerator()


# ---------------------------------------------------------------------------
# Test: code_hash is populated after embed_chunks
# ---------------------------------------------------------------------------

def test_code_hash_populated_after_embed_chunks(generator):
    """embed_chunks must set a non-empty code_hash on every ChunkInfo."""
    source = _make_large_function(n_stmts=35)
    chunks = _chunk_function(source, generator)

    assert len(chunks) > 1, "Expected multiple chunks from a large function"
    for chunk in chunks:
        assert chunk.code_hash, f"Chunk {chunk.chunk_index} has empty code_hash"


def test_code_hash_is_sha256_hex(generator):
    """code_hash must be a 64-character lowercase hex string (SHA-256 digest)."""
    source = _make_large_function(n_stmts=35)
    chunks = _chunk_function(source, generator)

    for chunk in chunks:
        assert len(chunk.code_hash) == 64, (
            f"Chunk {chunk.chunk_index}: expected 64-char hex, got {len(chunk.code_hash)!r}"
        )
        assert chunk.code_hash == chunk.code_hash.lower()
        int(chunk.code_hash, 16)  # must be valid hex — raises ValueError if not


# ---------------------------------------------------------------------------
# Test: idempotency — same source → same hashes
# ---------------------------------------------------------------------------

def test_same_source_produces_same_hashes(generator):
    """Re-running embed_chunks on an unchanged function yields identical hashes."""
    source = _make_large_function(n_stmts=35)

    chunks_first = _chunk_function(source, generator)
    chunks_second = _chunk_function(source, generator)

    assert len(chunks_first) == len(chunks_second), (
        "Chunk count changed between runs"
    )
    for c1, c2 in zip(chunks_first, chunks_second):
        assert c1.code_hash == c2.code_hash, (
            f"Chunk {c1.chunk_index}: hash changed between runs "
            f"({c1.code_hash!r} != {c2.code_hash!r})"
        )


def test_idempotency_across_three_runs(generator):
    """Three successive embed_chunks calls on the same source produce equal hashes."""
    source = _make_large_function(n_stmts=25)

    runs = [_chunk_function(source, generator) for _ in range(3)]
    reference = runs[0]

    for run_idx, run in enumerate(runs[1:], start=1):
        assert len(run) == len(reference), f"Run {run_idx}: chunk count mismatch"
        for c_ref, c_run in zip(reference, run):
            assert c_ref.code_hash == c_run.code_hash, (
                f"Run {run_idx}, chunk {c_ref.chunk_index}: hash changed "
                f"({c_ref.code_hash!r} != {c_run.code_hash!r})"
            )


def test_hash_matches_sha256_of_normalized_code(generator):
    """code_hash must equal sha256(normalized_text) where normalized_text is
    what embed_chunks feeds to the encoder."""
    from code_similarity_mcp.normalizer.registry import get_normalizer

    source = _make_large_function(n_stmts=20)
    statements = get_flat_statements(source)
    graph = build_dependency_graph(source)
    raw_chunks = group_into_chunks(graph, max_statements_per_chunk=10)
    annotated = annotate_chunks(raw_chunks, graph, function_name="test_func", file_path="f.py")

    # Replicate the normalization logic from embed_chunks to compute expected hashes.
    normalizer = get_normalizer("python")
    source_lines = source.splitlines()

    for chunk in annotated:
        stmt_infos = [statements[idx] for idx in chunk.statement_indices]
        start_line = min(s.start_line for s in stmt_infos)
        end_line = max(s.end_line for s in stmt_infos)
        chunk_source = "\n".join(source_lines[start_line - 1 : end_line])
        indented = "\n".join("    " + line for line in chunk_source.splitlines())
        wrapped = "def _chunk_func():\n" + indented
        norm_text = normalizer.normalize(wrapped)
        expected_hash = hashlib.sha256(norm_text.encode()).hexdigest()

        # embed_chunks sets code_hash; call it now.
        embed_chunks([chunk], source, statements, generator)

        assert chunk.code_hash == expected_hash, (
            f"Chunk {chunk.chunk_index}: code_hash does not match sha256(normalized_text)"
        )


# ---------------------------------------------------------------------------
# Test: change detection — altered source → different hash for affected chunk
# ---------------------------------------------------------------------------

def _find_first_changed_chunk(chunks_orig: list, chunks_mod: list) -> tuple[int, str, str]:
    """Return (chunk_index, orig_hash, mod_hash) for the first differing chunk."""
    for c_orig, c_mod in zip(chunks_orig, chunks_mod):
        if c_orig.code_hash != c_mod.code_hash:
            return c_orig.chunk_index, c_orig.code_hash, c_mod.code_hash
    return -1, "", ""


def test_single_char_change_produces_different_hash(generator):
    """A structural change to a statement (external function name — not a literal,
    which is normalized away) produces a different hash for the affected chunk."""
    # Use a function whose first chunk calls 'external_op_0'.  Renaming that
    # call to 'external_op_CHANGED' changes the normalized text because external
    # function names are preserved by the normalizer (only assignment targets
    # and parameters are renamed).
    source_orig = _make_function_with_calls(n_stmts=35)
    source_mod = source_orig.replace("external_op_0()", "external_op_CHANGED()", 1)
    assert source_orig != source_mod

    chunks_orig = _chunk_function(source_orig, generator)
    chunks_mod = _chunk_function(source_mod, generator)

    assert len(chunks_orig) == len(chunks_mod), "Chunk count changed unexpectedly"

    idx, orig_hash, mod_hash = _find_first_changed_chunk(chunks_orig, chunks_mod)
    assert idx >= 0, (
        "Expected at least one chunk hash to differ after source modification"
    )
    assert orig_hash != mod_hash


def test_unchanged_chunks_keep_same_hash_after_modification(generator):
    """When only one chunk's content changes, the other chunks' hashes are stable."""
    # Rename external_op_0 only — contained in chunk 0.
    # Later chunks' hashes must be identical to the original.
    source_orig = _make_function_with_calls(n_stmts=35)
    source_mod = source_orig.replace("external_op_0()", "external_op_CHANGED()", 1)

    chunks_orig = _chunk_function(source_orig, generator)
    chunks_mod = _chunk_function(source_mod, generator)

    assert len(chunks_orig) == len(chunks_mod)

    changed_count = sum(
        1 for c_o, c_m in zip(chunks_orig, chunks_mod)
        if c_o.code_hash != c_m.code_hash
    )
    unchanged_count = len(chunks_orig) - changed_count

    # At least one chunk must be unchanged (the modification is local to chunk 0).
    assert unchanged_count > 0, (
        "Expected some chunks to have the same hash after a local modification"
    )
    # The chunk containing external_op_0 must reflect the change.
    assert changed_count >= 1


def test_code_hash_empty_before_embed_chunks():
    """ChunkInfo.code_hash must default to empty string before embed_chunks is called."""
    source = _make_large_function(n_stmts=15)
    graph = build_dependency_graph(source)
    raw_chunks = group_into_chunks(graph, max_statements_per_chunk=10)
    annotated = annotate_chunks(raw_chunks, graph, function_name="f", file_path="f.py")

    for chunk in annotated:
        assert chunk.code_hash == "", (
            f"Chunk {chunk.chunk_index}: code_hash should be empty before embed_chunks"
        )


def test_function_with_single_chunk_hash_is_stable(generator):
    """A small function that produces one chunk must have a stable hash."""
    source = "def small_func():\n    a = 1\n    b = 2\n    return a + b\n"

    chunks_first = _chunk_function(source, generator)
    chunks_second = _chunk_function(source, generator)

    assert len(chunks_first) == len(chunks_second)
    for c1, c2 in zip(chunks_first, chunks_second):
        assert c1.code_hash == c2.code_hash
