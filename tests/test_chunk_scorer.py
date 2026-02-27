"""Unit tests for ChunkSimilarityScorer (feature #32)."""

from __future__ import annotations

import pytest

from code_similarity_mcp.similarity.chunk_scorer import ChunkSimilarityScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    stmt_count: int = 5,
    depends_on: list[int] | None = None,
    depended_on_by: list[int] | None = None,
) -> dict:
    """Build a minimal candidate chunk dict."""
    indices = list(range(stmt_count))
    return {
        "statement_indices": indices,
        "statement_start": 0,
        "statement_end": stmt_count - 1,
        "depends_on_chunks": depends_on or [],
        "depended_on_by_chunks": depended_on_by or [],
    }


def _make_query(
    stmt_count: int = 5,
    depends_on: list[int] | None = None,
    depended_on_by: list[int] | None = None,
    include_topology: bool = True,
) -> dict:
    """Build a minimal query info dict."""
    q: dict = {"stmt_count": stmt_count}
    if include_topology:
        q["depends_on_chunks"] = depends_on if depends_on is not None else []
        q["depended_on_by_chunks"] = depended_on_by if depended_on_by is not None else []
    return q


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCORER = ChunkSimilarityScorer()

W_E = ChunkSimilarityScorer.W_EMBEDDING
W_S = ChunkSimilarityScorer.W_STRUCTURAL


# ---------------------------------------------------------------------------
# Tests: identical chunks score 1.0
# ---------------------------------------------------------------------------

class TestIdenticalChunks:
    def test_identical_chunks_score_1(self):
        """Two identical chunks (same embed + same structure) should score 1.0."""
        query = _make_query(stmt_count=5)
        cand = _make_chunk(stmt_count=5)
        result = SCORER.score(embedding_score=1.0, query=query, candidate=cand)
        assert result == pytest.approx(1.0)

    def test_identical_with_deps_score_1(self):
        """Identical chunks with matching dependency topology score 1.0."""
        query = _make_query(stmt_count=4, depends_on=[0], depended_on_by=[2])
        cand = _make_chunk(stmt_count=4, depends_on=[0], depended_on_by=[2])
        result = SCORER.score(embedding_score=1.0, query=query, candidate=cand)
        assert result == pytest.approx(1.0)

    def test_snippet_query_identical_score_1(self):
        """Snippet query (no topology) with matching stmt count and embed=1 scores 1.0."""
        query = _make_query(stmt_count=6, include_topology=False)
        cand = _make_chunk(stmt_count=6)
        result = SCORER.score(embedding_score=1.0, query=query, candidate=cand)
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests: combined score is always in [0, 1]
# ---------------------------------------------------------------------------

class TestScoreRange:
    @pytest.mark.parametrize("emb,q_stmts,c_stmts", [
        (0.0, 1, 1),
        (0.5, 5, 10),
        (1.0, 10, 1),
        (0.99, 8, 8),
        (0.0, 100, 1),
    ])
    def test_score_in_0_1(self, emb, q_stmts, c_stmts):
        query = _make_query(stmt_count=q_stmts, include_topology=False)
        cand = _make_chunk(stmt_count=c_stmts)
        result = SCORER.score(embedding_score=emb, query=query, candidate=cand)
        assert 0.0 <= result <= 1.0

    def test_score_with_topology_in_0_1(self):
        query = _make_query(stmt_count=3, depends_on=[0, 1], depended_on_by=[])
        cand = _make_chunk(stmt_count=7, depends_on=[], depended_on_by=[3])
        result = SCORER.score(embedding_score=0.6, query=query, candidate=cand)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Tests: weighted combination formula
# ---------------------------------------------------------------------------

class TestWeightedFormula:
    def test_weights_sum_to_1(self):
        assert W_E + W_S == pytest.approx(1.0)

    def test_weights_are_07_and_03(self):
        assert W_E == pytest.approx(0.7)
        assert W_S == pytest.approx(0.3)

    def test_formula_with_known_values(self):
        """Manually verify: embed=0.8, structural=1.0 → 0.7*0.8 + 0.3*1.0 = 0.86."""
        query = _make_query(stmt_count=5, include_topology=False)
        cand = _make_chunk(stmt_count=5)
        result = SCORER.score(embedding_score=0.8, query=query, candidate=cand)
        assert result == pytest.approx(0.7 * 0.8 + 0.3 * 1.0)

    def test_formula_with_stmt_mismatch(self):
        """embed=1.0, stmt ratio=5/10=0.5 (no topology) → 0.7*1.0 + 0.3*0.5 = 0.85."""
        query = _make_query(stmt_count=5, include_topology=False)
        cand = _make_chunk(stmt_count=10)
        result = SCORER.score(embedding_score=1.0, query=query, candidate=cand)
        assert result == pytest.approx(0.7 * 1.0 + 0.3 * 0.5)


# ---------------------------------------------------------------------------
# Tests: structural similarity component
# ---------------------------------------------------------------------------

class TestStructuralSimilarity:
    def test_same_stmt_count_scores_1(self):
        query = _make_query(stmt_count=7, include_topology=False)
        cand = _make_chunk(stmt_count=7)
        assert SCORER.structural_similarity(query, cand) == pytest.approx(1.0)

    def test_stmt_count_ratio(self):
        query = _make_query(stmt_count=4, include_topology=False)
        cand = _make_chunk(stmt_count=8)
        assert SCORER.structural_similarity(query, cand) == pytest.approx(0.5)

    def test_no_deps_both_sides_scores_1(self):
        """When both query and candidate have no deps, dep score = 1.0."""
        query = _make_query(stmt_count=5, depends_on=[], depended_on_by=[])
        cand = _make_chunk(stmt_count=5, depends_on=[], depended_on_by=[])
        assert SCORER.structural_similarity(query, cand) == pytest.approx(1.0)

    def test_matching_dep_counts(self):
        """Same dep counts → dep components = 1.0."""
        query = _make_query(stmt_count=5, depends_on=[0, 1], depended_on_by=[3])
        cand = _make_chunk(stmt_count=5, depends_on=[0, 2], depended_on_by=[4])
        # stmt: 1.0, depends ratio: 2/2=1.0, dep_by ratio: 1/1=1.0 → avg=1.0
        assert SCORER.structural_similarity(query, cand) == pytest.approx(1.0)

    def test_dep_count_mismatch(self):
        """Different dep counts → dep score < 1.0."""
        query = _make_query(stmt_count=5, depends_on=[0], depended_on_by=[])
        cand = _make_chunk(stmt_count=5, depends_on=[0, 1, 2], depended_on_by=[])
        # stmt: 1.0, depends: min(1,3)/max(1,3)=1/3, dep_by: both 0→1.0
        expected = (1.0 + 1.0 / 3.0 + 1.0) / 3.0
        assert SCORER.structural_similarity(query, cand) == pytest.approx(expected)

    def test_snippet_query_no_topology(self):
        """Snippet query (no topology) only uses stmt count."""
        query = _make_query(stmt_count=5, include_topology=False)
        cand = _make_chunk(stmt_count=5, depends_on=[0, 1, 2], depended_on_by=[3, 4])
        # Only stmt count: 5/5=1.0
        assert SCORER.structural_similarity(query, cand) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests: structural score separates structurally similar from dissimilar
# ---------------------------------------------------------------------------

class TestStructuralDisambiguation:
    def test_structurally_similar_scores_higher(self):
        """Structurally similar candidate scores higher than a structurally different one."""
        query = _make_query(stmt_count=5, depends_on=[0], depended_on_by=[2])
        cand_similar = _make_chunk(stmt_count=5, depends_on=[0], depended_on_by=[2])
        cand_different = _make_chunk(stmt_count=1, depends_on=[], depended_on_by=[2, 3, 4])

        emb = 0.7  # same embedding for both candidates
        score_similar = SCORER.score(emb, query, cand_similar)
        score_different = SCORER.score(emb, query, cand_different)
        assert score_similar > score_different

    def test_structural_component_distinguishes_dep_topology(self):
        """When embedding is fixed, higher structural similarity → higher combined score."""
        query = _make_query(stmt_count=6, depends_on=[0, 1], depended_on_by=[])
        cand_good = _make_chunk(stmt_count=6, depends_on=[0, 1], depended_on_by=[])
        cand_bad = _make_chunk(stmt_count=6, depends_on=[], depended_on_by=[2, 3])

        emb = 0.8
        score_good = SCORER.score(emb, query, cand_good)
        score_bad = SCORER.score(emb, query, cand_bad)
        assert score_good > score_bad

    def test_structural_vs_embedding_tradeoff(self):
        """Higher structural score can compensate for lower embedding score."""
        query = _make_query(stmt_count=8, include_topology=False)
        cand = _make_chunk(stmt_count=8)

        # Low embedding but perfect structural
        score_low_emb = SCORER.score(embedding_score=0.5, query=query, candidate=cand)
        # Structural: 1.0, combined: 0.7*0.5 + 0.3*1.0 = 0.65

        cand_mismatch = _make_chunk(stmt_count=1)
        # Same low embedding but poor structural
        score_bad_struct = SCORER.score(embedding_score=0.5, query=query, candidate=cand_mismatch)
        # Structural: 1/8=0.125, combined: 0.7*0.5 + 0.3*0.125 = 0.3875

        assert score_low_emb > score_bad_struct


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_embedding_score(self):
        query = _make_query(stmt_count=5, include_topology=False)
        cand = _make_chunk(stmt_count=5)
        result = SCORER.score(embedding_score=0.0, query=query, candidate=cand)
        # 0.7*0 + 0.3*1.0 = 0.3
        assert result == pytest.approx(0.3)

    def test_unknown_stmt_count_defaults_to_equal(self):
        """When query stmt_count=0, falls back to neutral (treats as equal to candidate)."""
        query = {"stmt_count": 0}
        cand = _make_chunk(stmt_count=7)
        # stmt ratio: 7/7 = 1.0 (unknown treated as equal)
        struct = SCORER.structural_similarity(query, cand)
        assert struct == pytest.approx(1.0)

    def test_empty_statement_indices_fallback(self):
        """Candidate with empty statement_indices falls back to statement range size."""
        query = _make_query(stmt_count=3, include_topology=False)
        cand = {
            "statement_indices": [],
            "statement_start": 0,
            "statement_end": 2,
            "depends_on_chunks": [],
            "depended_on_by_chunks": [],
        }
        # c_stmts = 2 - 0 + 1 = 3, so ratio = 3/3 = 1.0
        struct = SCORER.structural_similarity(query, cand)
        assert struct == pytest.approx(1.0)
