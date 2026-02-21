"""Multi-signal similarity scoring with fast pre-filtering."""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from .filter import FilterPipeline


@dataclass
class SimilarityResult:
    db_id: int
    file_path: str
    name: str
    start_line: int
    end_line: int
    score: float
    embedding_score: float
    ast_score: float
    exact_match: bool
    differences: list[str]
    refactoring_hints: list[str]


class SimilarityScorer:
    """Combines multiple similarity signals into a final score."""

    def __init__(
        self,
        threshold: float = 0.85,
        w_embedding: float = 0.5,
        w_ast: float = 0.3,
        w_structural: float = 0.2,
        filter_pipeline: FilterPipeline | None = None,
    ) -> None:
        self.threshold = threshold
        self.w_embedding = w_embedding
        self.w_ast = w_ast
        self.w_structural = w_structural
        self._filter = filter_pipeline if filter_pipeline is not None else FilterPipeline()

    def score_candidates(
        self,
        query: dict,
        candidates: list[dict],
    ) -> list[SimilarityResult]:
        """
        Score and filter candidates against query method.
        query and candidates are method dicts from MethodRegistry.
        """
        results: list[SimilarityResult] = []

        for cand in candidates:
            if not self._filter.passes(query, cand):
                continue

            exact = query["code_hash"] == cand["code_hash"]
            if exact:
                embedding_score = 1.0
                ast_score = 1.0
            else:
                embedding_score = cand.get("embedding_score", 0.0)
                ast_score = self._ast_similarity(
                    query["normalized_code"], cand["normalized_code"]
                )

            structural = self._structural_score(query, cand)
            final = (
                self.w_embedding * embedding_score
                + self.w_ast * ast_score
                + self.w_structural * structural
            )

            if final < self.threshold and not exact:
                continue

            diffs = self._compute_differences(query, cand)
            hints = self._refactoring_hints(diffs)

            results.append(
                SimilarityResult(
                    db_id=cand["id"],
                    file_path=cand["file_path"],
                    name=cand["name"],
                    start_line=cand["start_line"],
                    end_line=cand["end_line"],
                    score=round(final, 4),
                    embedding_score=round(embedding_score, 4),
                    ast_score=round(ast_score, 4),
                    exact_match=exact,
                    differences=diffs,
                    refactoring_hints=hints,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _ast_similarity(self, a: str, b: str) -> float:
        """SequenceMatcher ratio on normalized code lines."""
        return difflib.SequenceMatcher(None, a, b).ratio()

    def _structural_score(self, query: dict, cand: dict) -> float:
        """Simple structural feature similarity."""
        scores = []
        # Parameter count similarity
        qp = len(query["parameters"])
        cp = len(cand["parameters"])
        scores.append(1.0 if qp == cp else 0.5)
        # Return type similarity
        qr = query.get("return_type")
        cr = cand.get("return_type")
        scores.append(1.0 if qr == cr else 0.3)
        # Dependency overlap
        qdeps = set(query.get("dependencies") or [])
        cdeps = set(cand.get("dependencies") or [])
        if qdeps or cdeps:
            overlap = len(qdeps & cdeps) / max(len(qdeps | cdeps), 1)
            scores.append(overlap)
        return sum(scores) / len(scores)

    # ------------------------------------------------------------------
    # Diff & hints
    # ------------------------------------------------------------------

    def _compute_differences(self, query: dict, cand: dict) -> list[str]:
        diffs = []
        qp, cp = query["parameters"], cand["parameters"]
        if len(qp) != len(cp):
            diffs.append(f"parameter count differs ({len(qp)} vs {len(cp)})")
        qr, cr = query.get("return_type"), cand.get("return_type")
        if qr != cr:
            diffs.append(f"return type differs ({qr!r} vs {cr!r})")
        qdeps = set(query.get("dependencies") or [])
        cdeps = set(cand.get("dependencies") or [])
        only_q = qdeps - cdeps
        only_c = cdeps - qdeps
        if only_q:
            diffs.append(f"query calls not in candidate: {sorted(only_q)}")
        if only_c:
            diffs.append(f"candidate calls not in query: {sorted(only_c)}")
        return diffs

    def _refactoring_hints(self, differences: list[str]) -> list[str]:
        hints = []
        for diff in differences:
            if "parameter count" in diff:
                hints.append("consider unifying parameter signatures")
            if "return type" in diff:
                hints.append("align return types or extract shared interface")
            if "calls not in" in diff:
                hints.append("extract shared dependency logic into utility")
        return list(dict.fromkeys(hints))
