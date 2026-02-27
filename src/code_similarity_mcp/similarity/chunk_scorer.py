"""Chunk-specific similarity scorer combining embedding and structural signals."""

from __future__ import annotations


class ChunkSimilarityScorer:
    """Combines embedding cosine similarity and structural similarity for chunks.

    Weighted score: ``W_EMBEDDING * embedding_cosine + W_STRUCTURAL * structural``.

    Structural similarity is computed from:

    * **Statement count ratio** — ``min(q, c) / max(q, c)`` where *q* and *c*
      are the statement counts of the query and candidate chunks respectively.
      Always included.

    * **Dependency topology** — ratio of ``depends_on_chunks`` counts and
      ``depended_on_by_chunks`` counts.  Only included when the *query* dict
      contains ``depends_on_chunks`` (i.e. when the query is a stored chunk
      rather than a raw code snippet).
    """

    W_EMBEDDING: float = 0.7
    W_STRUCTURAL: float = 0.3

    def score(
        self,
        embedding_score: float,
        query: dict,
        candidate: dict,
    ) -> float:
        """Return combined similarity score in ``[0.0, 1.0]``.

        Args:
            embedding_score: Cosine similarity from FAISS (0.0–1.0).
            query: Query chunk info dict.  Expected keys:

                * ``stmt_count`` (*int*): number of statements in the query chunk.
                * ``depends_on_chunks`` (*list*, optional): chunk indices this
                  chunk depends on.  When absent, dependency topology is not
                  included in the structural component.
                * ``depended_on_by_chunks`` (*list*, optional): chunk indices
                  that depend on this chunk.
            candidate: Full chunk dict from :class:`~MethodRegistry` (as
                returned by :meth:`~MethodRegistry.search_chunks`).

        Returns:
            Combined score clamped to ``[0.0, 1.0]``.
        """
        structural = self.structural_similarity(query, candidate)
        combined = self.W_EMBEDDING * embedding_score + self.W_STRUCTURAL * structural
        return max(0.0, min(1.0, combined))

    def structural_similarity(self, query: dict, candidate: dict) -> float:
        """Compute structural similarity between a query and a candidate chunk.

        Args:
            query: Query chunk info dict (see :meth:`score` for expected keys).
            candidate: Full candidate chunk dict from :class:`~MethodRegistry`.

        Returns:
            Structural similarity in ``[0.0, 1.0]``.
        """
        scores: list[float] = []

        # --- 1. Statement count similarity -----------------------------------
        q_stmts = query.get("stmt_count") or len(query.get("statement_indices", []))
        c_stmts = len(candidate.get("statement_indices", []))
        if not c_stmts:
            # Fallback when statement_indices is empty (shouldn't happen in practice)
            c_stmts = max(
                1,
                candidate.get("statement_end", 1) - candidate.get("statement_start", 0) + 1,
            )
        if not q_stmts:
            q_stmts = c_stmts  # Unknown → treat as equal (neutral score)

        scores.append(min(q_stmts, c_stmts) / max(q_stmts, c_stmts))

        # --- 2. Dependency topology (only when query has this info) ----------
        if "depends_on_chunks" in query:
            q_deps = len(query["depends_on_chunks"])
            c_deps = len(candidate.get("depends_on_chunks", []))
            if max(q_deps, c_deps) == 0:
                scores.append(1.0)
            else:
                scores.append(min(q_deps, c_deps) / max(q_deps, c_deps))

            q_by = len(query.get("depended_on_by_chunks", []))
            c_by = len(candidate.get("depended_on_by_chunks", []))
            if max(q_by, c_by) == 0:
                scores.append(1.0)
            else:
                scores.append(min(q_by, c_by) / max(q_by, c_by))

        return sum(scores) / len(scores)
