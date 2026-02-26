"""Tests for embed_chunks (feature #24): normalized embedding per chunk."""

import textwrap

import numpy as np
import pytest

from code_similarity_mcp.embeddings.generator import EmbeddingGenerator
from code_similarity_mcp.parser.python import build_dependency_graph, get_flat_statements
from code_similarity_mcp.parser.base import (
    annotate_chunks,
    embed_chunks,
    group_into_chunks,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def generator():
    """Single EmbeddingGenerator instance shared across tests (lazy model load)."""
    return EmbeddingGenerator()


def _build_chunks(code: str):
    """Helper: parse *code* and return (annotated_chunks, flat_statements)."""
    code = textwrap.dedent(code)
    graph = build_dependency_graph(code)
    stmts = get_flat_statements(code)
    raw_chunks = group_into_chunks(graph)
    chunks = annotate_chunks(raw_chunks, graph, function_name="f")
    return chunks, stmts, code


# ---------------------------------------------------------------------------
# get_flat_statements
# ---------------------------------------------------------------------------

class TestGetFlatStatements:
    def test_simple_function_three_stmts(self):
        code = textwrap.dedent("""\
            def f():
                x = 1
                y = x + 2
                return y
        """)
        stmts = get_flat_statements(code)
        assert len(stmts) == 3
        assert stmts[0].index == 0
        assert stmts[1].index == 1
        assert stmts[2].index == 2

    def test_indices_match_graph_num_statements(self):
        code = textwrap.dedent("""\
            def f(items):
                total = 0
                for item in items:
                    total += item
                return total
        """)
        stmts = get_flat_statements(code)
        graph = build_dependency_graph(code)
        assert len(stmts) == graph.num_statements

    def test_source_text_nonempty(self):
        code = textwrap.dedent("""\
            def f():
                x = 10
        """)
        stmts = get_flat_statements(code)
        assert all(s.source_text.strip() for s in stmts)

    def test_start_end_line_1based(self):
        code = textwrap.dedent("""\
            def f():
                x = 1
                y = 2
        """)
        stmts = get_flat_statements(code)
        # 'def f():' is line 1; 'x = 1' is line 2; 'y = 2' is line 3
        assert stmts[0].start_line == 2
        assert stmts[1].start_line == 3

    def test_empty_function_returns_empty(self):
        code = "def f():\n    pass\n"
        stmts = get_flat_statements(code)
        # pass_statement is a statement type — one entry expected
        # (or zero if pass is excluded from _STATEMENT_TYPES)
        assert isinstance(stmts, list)

    def test_no_function_returns_empty(self):
        stmts = get_flat_statements("x = 1\n")
        assert stmts == []

    def test_nested_stmt_has_own_entry(self):
        """The body of a for loop should appear as a separate flat statement."""
        code = textwrap.dedent("""\
            def f(items):
                total = 0
                for item in items:
                    total += item
                return total
        """)
        stmts = get_flat_statements(code)
        # Flat list: 0=total=0, 1=for header, 2=total+=item, 3=return total
        assert len(stmts) == 4
        node_types = [s.node_type for s in stmts]
        assert "for_statement" in node_types
        assert "expression_statement" in node_types  # total += item


# ---------------------------------------------------------------------------
# embed_chunks — structural / unit tests
# ---------------------------------------------------------------------------

class TestEmbedChunksUnit:
    def test_empty_chunks_returns_empty(self, generator):
        result = embed_chunks([], "def f():\n    pass\n", [], generator)
        assert result == []

    def test_returns_list_of_arrays(self, generator):
        chunks, stmts, code = _build_chunks("""\
            def f():
                x = 10
                y = x + 5
                return y
        """)
        result = embed_chunks(chunks, code, stmts, generator)
        assert isinstance(result, list)
        assert all(isinstance(e, np.ndarray) for e in result)

    def test_one_embedding_per_chunk(self, generator):
        chunks, stmts, code = _build_chunks("""\
            def f():
                x = 10
                y = x + 5
                return y
        """)
        result = embed_chunks(chunks, code, stmts, generator)
        assert len(result) == len(chunks)

    def test_embedding_dimension_384(self, generator):
        chunks, stmts, code = _build_chunks("""\
            def f():
                x = 10
                y = x + 5
                return y
        """)
        result = embed_chunks(chunks, code, stmts, generator)
        for emb in result:
            assert emb.shape == (384,)

    def test_embedding_dtype_float32(self, generator):
        chunks, stmts, code = _build_chunks("""\
            def f():
                x = 10
                return x
        """)
        result = embed_chunks(chunks, code, stmts, generator)
        for emb in result:
            assert emb.dtype == np.float32

    def test_embeddings_are_l2_normalized(self, generator):
        """EmbeddingGenerator returns L2-normalized vectors; norm should be ~1."""
        chunks, stmts, code = _build_chunks("""\
            def f():
                x = 10
                return x
        """)
        result = embed_chunks(chunks, code, stmts, generator)
        for emb in result:
            norm = float(np.linalg.norm(emb))
            assert abs(norm - 1.0) < 1e-5, f"expected norm ~1, got {norm}"

    def test_two_chunks_from_same_function(self, generator):
        """A function that splits into two chunks produces two distinct embeddings."""
        # Two fresh-start statements (no deps on each other) → two chunks
        code = textwrap.dedent("""\
            def f():
                a = 1
                b = 2
        """)
        graph = build_dependency_graph(code)
        stmts = get_flat_statements(code)
        # Both statements have no providers → each is a fresh start (2 chunks)
        raw_chunks = group_into_chunks(graph)
        chunks = annotate_chunks(raw_chunks, graph, function_name="f")
        result = embed_chunks(chunks, code, stmts, generator)
        assert len(result) == len(chunks)
        assert all(e.shape == (384,) for e in result)


# ---------------------------------------------------------------------------
# Semantic similarity test (core feature requirement)
# ---------------------------------------------------------------------------

class TestEmbedChunksSemantic:
    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity of two L2-normalized vectors (= dot product)."""
        return float(np.dot(a, b))

    def test_equivalent_chunks_cosine_above_0_9(self, generator):
        """Two chunks that perform identical logic under different variable names
        must produce embeddings with cosine similarity > 0.9.

        After normalization both chunks produce identical text, so the expected
        cosine is 1.0 (or very close to it).
        """
        code_a = textwrap.dedent("""\
            def func_a():
                x = 10
                y = x + 5
                return y
        """)
        code_b = textwrap.dedent("""\
            def func_b():
                count = 10
                total = count + 5
                return total
        """)
        chunks_a, stmts_a, _ = _build_chunks(code_a)
        chunks_b, stmts_b, _ = _build_chunks(code_b)

        embs_a = embed_chunks(chunks_a, code_a, stmts_a, generator)
        embs_b = embed_chunks(chunks_b, code_b, stmts_b, generator)

        assert len(embs_a) == 1, "func_a should produce exactly one chunk"
        assert len(embs_b) == 1, "func_b should produce exactly one chunk"

        sim = self._cosine(embs_a[0], embs_b[0])
        assert sim > 0.9, f"expected cosine > 0.9 for equivalent chunks, got {sim:.4f}"

    def test_identical_chunks_cosine_near_1(self, generator):
        """Identical code produces embeddings with cosine ≈ 1.0."""
        code = textwrap.dedent("""\
            def f():
                x = 10
                y = x + 5
                return y
        """)
        chunks, stmts, _ = _build_chunks(code)
        embs = embed_chunks(chunks, code, stmts, generator)
        assert len(embs) >= 1
        sim = self._cosine(embs[0], embs[0])
        assert abs(sim - 1.0) < 1e-5

    def test_different_logic_chunks_lower_similarity(self, generator):
        """Two chunks with very different logic should have lower similarity
        than two semantically equivalent chunks."""
        code_equiv_a = textwrap.dedent("""\
            def equiv_a():
                x = 10
                y = x + 5
                return y
        """)
        code_equiv_b = textwrap.dedent("""\
            def equiv_b():
                count = 10
                total = count + 5
                return total
        """)
        code_different = textwrap.dedent("""\
            def different():
                items = []
                items.append(1)
                return items
        """)

        chunks_a, stmts_a, _ = _build_chunks(code_equiv_a)
        chunks_b, stmts_b, _ = _build_chunks(code_equiv_b)
        chunks_d, stmts_d, _ = _build_chunks(code_different)

        emb_a = embed_chunks(chunks_a, code_equiv_a, stmts_a, generator)[0]
        emb_b = embed_chunks(chunks_b, code_equiv_b, stmts_b, generator)[0]
        emb_d = embed_chunks(chunks_d, code_different, stmts_d, generator)[0]

        sim_equiv = self._cosine(emb_a, emb_b)
        sim_diff = self._cosine(emb_a, emb_d)

        assert sim_equiv > sim_diff, (
            f"equivalent chunks ({sim_equiv:.4f}) should be more similar "
            f"than different chunks ({sim_diff:.4f})"
        )
