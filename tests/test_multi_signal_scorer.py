"""Feature #9 — Multi-signal similarity scorer.

Covers:
  - AST fingerprint extraction via PythonParser
  - Fingerprint persisted in MethodRegistry
  - Fingerprint-based AST similarity in SimilarityScorer (primary path)
  - Fallback to normalized-code text similarity (no fingerprint present)
  - Exact hash match (score == 1.0, bypasses embedding/AST)
  - Configurable weights and threshold
  - SimilarityResult breakdown fields
  - End-to-end: parser → registry → scorer uses fingerprints
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from code_similarity_mcp.parser.python import PythonParser
from code_similarity_mcp.index.registry import MethodRegistry
from code_similarity_mcp.similarity.scorer import SimilarityScorer, SimilarityResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PARSER = PythonParser()


def parse_one(code: str):
    methods = _PARSER.parse_snippet(textwrap.dedent(code))
    assert len(methods) == 1, f"Expected 1 method, got {len(methods)}"
    return methods[0]


def _random_embedding(seed: int = 0, dim: int = 384) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.random(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def _method_dict(**kwargs) -> dict:
    """Minimal method dict for SimilarityScorer tests."""
    defaults = {
        "id": 1,
        "file_path": "test.py",
        "language": "python",
        "name": "func",
        "parameters": ["a", "b"],
        "return_type": None,
        "normalized_code": "def FUNC_NAME(v1, v2):\n return v1 + v2",
        "code_hash": "hash_default",
        "start_line": 1,
        "end_line": 4,
        "dependencies": [],
        "embedding_score": 0.9,
        "ast_fingerprint": [],
    }
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# 1. AST fingerprint extraction
# ---------------------------------------------------------------------------


class TestAstFingerprintExtraction:
    def test_fingerprint_is_non_empty(self):
        m = parse_one("""\
            def add(a, b):
                return a + b
        """)
        assert len(m.ast_fingerprint) > 0

    def test_fingerprint_contains_function_definition(self):
        m = parse_one("""\
            def foo():
                pass
        """)
        assert "function_definition" in m.ast_fingerprint

    def test_fingerprint_contains_return_statement(self):
        m = parse_one("""\
            def compute(x):
                return x * 2
        """)
        assert "return_statement" in m.ast_fingerprint

    def test_identical_structure_identical_fingerprint(self):
        """Two structurally identical functions produce the same fingerprint."""
        m1 = parse_one("""\
            def add(a, b):
                return a + b
        """)
        m2 = parse_one("""\
            def multiply(x, y):
                return x * y
        """)
        # Structure is the same (function_def → params → return binary_op)
        assert m1.ast_fingerprint == m2.ast_fingerprint

    def test_different_structure_different_fingerprint(self):
        """Functions with different structure produce different fingerprints."""
        m_simple = parse_one("""\
            def simple(a):
                return a
        """)
        m_complex = parse_one("""\
            def complex_func(a, b):
                if a > 0:
                    return a + b
                return b
        """)
        assert m_simple.ast_fingerprint != m_complex.ast_fingerprint

    def test_fingerprint_is_list_of_strings(self):
        m = parse_one("""\
            def greet(name):
                return name
        """)
        assert isinstance(m.ast_fingerprint, list)
        assert all(isinstance(t, str) for t in m.ast_fingerprint)

    def test_nested_function_fingerprint_includes_inner(self):
        # parse_snippet returns both outer and inner as separate methods;
        # the outer function's subtree fingerprint contains both
        # function_definition nodes (outer wraps inner in the AST).
        methods = _PARSER.parse_snippet(textwrap.dedent("""\
            def outer(x):
                def inner(y):
                    return y
                return inner(x)
        """))
        outer = methods[0]
        # outer's DFS fingerprint spans the whole subtree including inner
        assert outer.ast_fingerprint.count("function_definition") >= 2


# ---------------------------------------------------------------------------
# 2. Fingerprint persisted in MethodRegistry
# ---------------------------------------------------------------------------


class TestFingerprintInRegistry:
    @pytest.fixture
    def registry(self, tmp_path):
        reg = MethodRegistry(tmp_path / "index")
        yield reg
        reg.close()

    def test_fingerprint_stored_and_retrieved(self, registry):
        m = parse_one("""\
            def add(a, b):
                return a + b
        """)
        registry.add_method(m, _random_embedding(1))
        rows = registry.get_by_file("<snippet>")
        assert len(rows) == 1
        assert rows[0]["ast_fingerprint"] == m.ast_fingerprint

    def test_fingerprint_is_list_after_retrieval(self, registry):
        m = parse_one("""\
            def foo(x):
                return x
        """)
        registry.add_method(m, _random_embedding(2))
        rows = registry.get_by_file("<snippet>")
        assert isinstance(rows[0]["ast_fingerprint"], list)

    def test_empty_fingerprint_stored_as_empty_list(self, registry):
        """MethodInfo with no fingerprint stores and returns [] gracefully."""
        from code_similarity_mcp.parser.base import MethodInfo
        m = MethodInfo(
            file_path="<test>",
            language="python",
            name="no_fp",
            parameters=[],
            return_type=None,
            body_code="def no_fp(): pass",
            normalized_code="def FUNC_NAME(): pass",
            start_line=1,
            end_line=1,
            dependencies=[],
            ast_fingerprint=[],
        )
        registry.add_method(m, _random_embedding(3))
        rows = registry.get_by_file("<test>")
        assert rows[0]["ast_fingerprint"] == []

    def test_fingerprint_survives_search(self, registry):
        """Fingerprint is included in rows returned by registry.search()."""
        m = parse_one("""\
            def sum_list(items):
                return sum(items)
        """)
        emb = _random_embedding(10)
        registry.add_method(m, emb)
        results = registry.search(emb, top_k=1)
        assert len(results) == 1
        assert results[0]["ast_fingerprint"] == m.ast_fingerprint


# ---------------------------------------------------------------------------
# 3. Fingerprint-based AST similarity in scorer
# ---------------------------------------------------------------------------


class TestFingerprintAstSimilarity:
    @pytest.fixture
    def scorer(self):
        return SimilarityScorer(threshold=0.0)  # accept all to test scores

    def test_identical_fingerprints_give_ast_score_one(self, scorer):
        fp = ["function_definition", "parameters", "return_statement", "binary_operator"]
        query = _method_dict(code_hash="Q", ast_fingerprint=fp, embedding_score=0.0)
        cand = _method_dict(id=2, code_hash="C", ast_fingerprint=fp, embedding_score=0.0)
        results = scorer.score_candidates(query, [cand])
        assert len(results) == 1
        assert results[0].ast_score == pytest.approx(1.0)

    def test_disjoint_fingerprints_give_low_ast_score(self, scorer):
        fp_a = ["function_definition", "return_statement"]
        fp_b = ["if_statement", "for_statement", "while_statement"]
        query = _method_dict(code_hash="Q", ast_fingerprint=fp_a, embedding_score=0.0)
        cand = _method_dict(id=2, code_hash="C", ast_fingerprint=fp_b, embedding_score=0.0)
        results = scorer.score_candidates(query, [cand])
        assert len(results) == 1
        assert results[0].ast_score < 0.5

    def test_partial_overlap_gives_intermediate_ast_score(self, scorer):
        fp_a = ["function_definition", "parameters", "return_statement", "binary_operator"]
        fp_b = ["function_definition", "parameters", "return_statement", "call"]
        query = _method_dict(code_hash="Q", ast_fingerprint=fp_a, embedding_score=0.0)
        cand = _method_dict(id=2, code_hash="C", ast_fingerprint=fp_b, embedding_score=0.0)
        results = scorer.score_candidates(query, [cand])
        assert len(results) == 1
        assert 0.0 < results[0].ast_score < 1.0

    def test_fallback_to_text_when_no_fingerprints(self, scorer):
        """When both fingerprints are empty, text-based similarity is used."""
        same_code = "def FUNC_NAME(v1, v2):\n return v1 + v2"
        query = _method_dict(code_hash="Q", normalized_code=same_code, ast_fingerprint=[],
                             embedding_score=0.0)
        cand = _method_dict(id=2, code_hash="C", normalized_code=same_code,
                            ast_fingerprint=[], embedding_score=0.0)
        results = scorer.score_candidates(query, [cand])
        assert len(results) == 1
        assert results[0].ast_score == pytest.approx(1.0)

    def test_fallback_when_only_query_has_no_fingerprint(self, scorer):
        """Missing query fingerprint → fallback to text diff."""
        code = "def FUNC_NAME(v1):\n return v1"
        fp = ["function_definition", "parameters", "return_statement"]
        query = _method_dict(code_hash="Q", normalized_code=code, ast_fingerprint=[],
                             embedding_score=0.0)
        cand = _method_dict(id=2, code_hash="C", normalized_code=code,
                            ast_fingerprint=fp, embedding_score=0.0)
        results = scorer.score_candidates(query, [cand])
        assert len(results) == 1
        # Should not crash and should produce a valid score
        assert 0.0 <= results[0].ast_score <= 1.0


# ---------------------------------------------------------------------------
# 4. Exact hash match
# ---------------------------------------------------------------------------


class TestExactHashMatch:
    @pytest.fixture
    def scorer(self):
        return SimilarityScorer(threshold=0.85)

    def test_exact_match_score_is_one(self, scorer):
        query = _method_dict(code_hash="SAME_HASH")
        cand = _method_dict(id=2, code_hash="SAME_HASH")
        results = scorer.score_candidates(query, [cand])
        assert len(results) == 1
        assert results[0].exact_match is True
        assert results[0].score == pytest.approx(1.0)

    def test_exact_match_embedding_and_ast_are_one(self, scorer):
        """Exact hash → both sub-scores forced to 1.0."""
        query = _method_dict(code_hash="X")
        cand = _method_dict(id=2, code_hash="X", embedding_score=0.3)
        results = scorer.score_candidates(query, [cand])
        assert results[0].embedding_score == pytest.approx(1.0)
        assert results[0].ast_score == pytest.approx(1.0)

    def test_exact_match_bypasses_threshold(self, scorer):
        """Exact hash match always included regardless of weights."""
        scorer2 = SimilarityScorer(threshold=0.99)
        query = _method_dict(code_hash="Y")
        cand = _method_dict(id=2, code_hash="Y")
        results = scorer2.score_candidates(query, [cand])
        assert len(results) == 1
        assert results[0].exact_match is True

    def test_different_hash_not_exact_match(self, scorer):
        query = _method_dict(code_hash="A")
        cand = _method_dict(id=2, code_hash="B", embedding_score=0.95)
        results = scorer.score_candidates(query, [cand])
        if results:
            assert results[0].exact_match is False


# ---------------------------------------------------------------------------
# 5. Configurable weights
# ---------------------------------------------------------------------------


class TestConfigurableWeights:
    def test_custom_weights_affect_score(self):
        """Different weight configs produce different combined scores."""
        fp = ["function_definition", "return_statement"]
        query = _method_dict(code_hash="Q", ast_fingerprint=fp)
        cand_base = dict(
            id=2,
            code_hash="C",
            ast_fingerprint=fp,
            embedding_score=0.7,
            normalized_code="def FUNC_NAME(v1, v2):\n return v1 + v2",
            language="python",
            parameters=["a", "b"],
            return_type=None,
            start_line=1,
            end_line=4,
            dependencies=[],
            file_path="test.py",
            name="func",
        )

        scorer_emb_heavy = SimilarityScorer(
            threshold=0.0, w_embedding=0.9, w_ast=0.05, w_structural=0.05
        )
        scorer_ast_heavy = SimilarityScorer(
            threshold=0.0, w_embedding=0.05, w_ast=0.9, w_structural=0.05
        )

        r_emb = scorer_emb_heavy.score_candidates(query, [cand_base])
        r_ast = scorer_ast_heavy.score_candidates(query, [cand_base])

        # Both should produce results
        assert len(r_emb) == 1
        assert len(r_ast) == 1
        # The scores should differ when weights differ
        assert r_emb[0].score != r_ast[0].score

    def test_default_weights_sum_to_one(self):
        s = SimilarityScorer()
        assert s.w_embedding + s.w_ast + s.w_structural == pytest.approx(1.0)

    def test_custom_weights_accepted(self):
        s = SimilarityScorer(w_embedding=0.6, w_ast=0.2, w_structural=0.2)
        assert s.w_embedding == 0.6
        assert s.w_ast == 0.2
        assert s.w_structural == 0.2


# ---------------------------------------------------------------------------
# 6. Configurable threshold
# ---------------------------------------------------------------------------


class TestConfigurableThreshold:
    def _make_pair(self, embedding_score: float, ast_fp=None):
        """Return (query, cand) with controllable embedding score."""
        fp = ast_fp or ["function_definition", "return_statement"]
        query = _method_dict(code_hash="Q", ast_fingerprint=fp)
        cand = _method_dict(
            id=2, code_hash="C", embedding_score=embedding_score, ast_fingerprint=fp
        )
        return query, cand

    def test_candidate_above_threshold_included(self):
        scorer = SimilarityScorer(threshold=0.5)
        query, cand = self._make_pair(embedding_score=0.9)
        results = scorer.score_candidates(query, [cand])
        assert len(results) == 1

    def test_candidate_below_threshold_excluded(self):
        scorer = SimilarityScorer(threshold=0.95)
        query, cand = self._make_pair(embedding_score=0.1)
        results = scorer.score_candidates(query, [cand])
        assert len(results) == 0

    def test_default_threshold_is_0_85(self):
        assert SimilarityScorer().threshold == pytest.approx(0.85)

    def test_zero_threshold_includes_everything(self):
        scorer = SimilarityScorer(threshold=0.0)
        query = _method_dict(code_hash="Q")
        cand = _method_dict(id=2, code_hash="C", embedding_score=0.01)
        results = scorer.score_candidates(query, [cand])
        assert len(results) == 1

    def test_one_threshold_only_exact_matches(self):
        scorer = SimilarityScorer(threshold=1.0)
        query = _method_dict(code_hash="SAME")
        exact = _method_dict(id=2, code_hash="SAME")
        not_exact = _method_dict(id=3, code_hash="OTHER", embedding_score=0.99)
        results = scorer.score_candidates(query, [exact, not_exact])
        assert len(results) == 1
        assert results[0].exact_match is True


# ---------------------------------------------------------------------------
# 7. SimilarityResult breakdown fields
# ---------------------------------------------------------------------------


class TestSimilarityResultBreakdown:
    @pytest.fixture
    def scorer(self):
        return SimilarityScorer(threshold=0.0)

    def test_result_has_all_breakdown_fields(self, scorer):
        query = _method_dict(code_hash="Q", embedding_score=0.0)
        cand = _method_dict(id=2, code_hash="C", embedding_score=0.8)
        results = scorer.score_candidates(query, [cand])
        r = results[0]
        assert hasattr(r, "score")
        assert hasattr(r, "embedding_score")
        assert hasattr(r, "ast_score")
        assert hasattr(r, "exact_match")
        assert hasattr(r, "differences")
        assert hasattr(r, "refactoring_hints")

    def test_result_scores_are_rounded(self, scorer):
        query = _method_dict(code_hash="Q")
        cand = _method_dict(id=2, code_hash="C", embedding_score=0.123456789)
        results = scorer.score_candidates(query, [cand])
        # Scores are rounded to 4 decimal places
        r = results[0]
        assert r.score == round(r.score, 4)
        assert r.embedding_score == round(r.embedding_score, 4)
        assert r.ast_score == round(r.ast_score, 4)

    def test_result_metadata_present(self, scorer):
        # query LOC = 4 (start=1, end=4); candidate must be within ±30%
        # LOC range: [floor(4*0.7), ceil(4/0.7)+1] = [2, 6]
        # Use start=10, end=13 → LOC=4 (ratio 4/4=1.0, passes)
        query = _method_dict(code_hash="Q")
        cand = _method_dict(id=2, code_hash="C", file_path="src/foo.py",
                            name="my_func", start_line=10, end_line=13,
                            embedding_score=0.9)
        results = scorer.score_candidates(query, [cand])
        assert len(results) == 1, "Candidate should pass filter and threshold"
        r = results[0]
        assert r.db_id == 2
        assert r.file_path == "src/foo.py"
        assert r.name == "my_func"
        assert r.start_line == 10
        assert r.end_line == 13

    def test_results_sorted_descending(self, scorer):
        query = _method_dict(code_hash="Q")
        c1 = _method_dict(id=2, code_hash="C1", embedding_score=0.6)
        c2 = _method_dict(id=3, code_hash="C2", embedding_score=0.95)
        c3 = _method_dict(id=4, code_hash="C3", embedding_score=0.8)
        results = scorer.score_candidates(query, [c1, c2, c3])
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# 8. Integration: parser → registry → scorer uses fingerprints
# ---------------------------------------------------------------------------


class TestEndToEndFingerprintFlow:
    @pytest.fixture
    def registry(self, tmp_path):
        reg = MethodRegistry(tmp_path / "index")
        yield reg
        reg.close()

    def test_fingerprint_flows_from_parser_to_scorer(self, registry):
        """Fingerprints extracted by parser survive round-trip through registry."""
        code_add = textwrap.dedent("""\
            def add(a, b):
                return a + b
        """)
        code_mul = textwrap.dedent("""\
            def multiply(x, y):
                return x * y
        """)
        # Both have same structure → same fingerprint
        m_add = _PARSER.parse_snippet(code_add)[0]
        m_mul = _PARSER.parse_snippet(code_mul)[0]
        assert m_add.ast_fingerprint == m_mul.ast_fingerprint

        emb = _random_embedding(42)
        registry.add_method(m_add, emb)

        # Query with m_mul (same structure, different names)
        candidates = registry.search(emb, top_k=5)
        assert len(candidates) == 1
        # The stored fingerprint should match what the parser produced
        assert candidates[0]["ast_fingerprint"] == m_add.ast_fingerprint

        # Now score using the fingerprints
        scorer = SimilarityScorer(threshold=0.0)
        query_dict = {
            "id": -1,
            "file_path": "<snippet>",
            "language": "python",
            "name": m_mul.name,
            "parameters": m_mul.parameters,
            "return_type": m_mul.return_type,
            "normalized_code": m_mul.normalized_code or "def FUNC_NAME(v1, v2):\n return v1 * v2",
            "code_hash": "DIFFERENT",
            "start_line": m_mul.start_line,
            "end_line": m_mul.end_line,
            "dependencies": m_mul.dependencies,
            "ast_fingerprint": m_mul.ast_fingerprint,
            "embedding_score": 0.95,
        }
        candidates[0]["embedding_score"] = 0.95

        results = scorer.score_candidates(query_dict, candidates)
        assert len(results) == 1
        # Same structure → ast_score should be 1.0
        assert results[0].ast_score == pytest.approx(1.0)

    def test_structurally_different_functions_get_lower_ast_score(self, registry):
        """A structurally complex function scores lower against a simple one.

        We test _ast_similarity directly to isolate AST score logic from the
        LOC pre-filter (which would reject candidates with very different LOC).
        """
        simple = _PARSER.parse_snippet(textwrap.dedent("""\
            def simple(a):
                return a
        """))[0]
        complex_fn = _PARSER.parse_snippet(textwrap.dedent("""\
            def complex_func(a, b):
                if a > 0:
                    for i in range(b):
                        a += i
                return a
        """))[0]

        assert simple.ast_fingerprint != complex_fn.ast_fingerprint

        scorer = SimilarityScorer(threshold=0.0)
        q = {"normalized_code": "...", "ast_fingerprint": complex_fn.ast_fingerprint}
        c = {"normalized_code": "...", "ast_fingerprint": simple.ast_fingerprint}
        score = scorer._ast_similarity(q, c)
        # Structurally different → score < 1.0
        assert score < 1.0
