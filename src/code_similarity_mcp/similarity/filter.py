"""Fast pre-filter pipeline for candidate methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from code_similarity_mcp.index.registry import MethodRegistry

_LOC_RATIO_MIN = 0.7  # candidates must have LOC within ±30% of query


class FilterPipeline:
    """
    Pre-filters candidate methods using cheap structural checks.

    Filters applied (in order):
    - Language: exact match required
    - Parameter count: difference ≤ 1
    - LOC: min/max ratio ≥ 0.7  (i.e. within ±30%)
    """

    LOC_RATIO_MIN: float = _LOC_RATIO_MIN

    # ------------------------------------------------------------------
    # Individual predicate filters
    # ------------------------------------------------------------------

    def language_matches(self, query_language: str, candidate: dict) -> bool:
        """Return True if candidate's language exactly matches the query."""
        return query_language == candidate["language"]

    def param_count_within_one(self, query_param_count: int, candidate: dict) -> bool:
        """Return True if candidate's parameter count is within ±1 of query."""
        return abs(len(candidate["parameters"]) - query_param_count) <= 1

    def loc_within_range(self, query_loc: int, candidate: dict) -> bool:
        """Return True if candidate's LOC is within ±30% of query (ratio ≥ 0.7)."""
        if query_loc == 0:
            return True
        c_loc = candidate["end_line"] - candidate["start_line"] + 1
        if c_loc == 0:
            return True
        return min(query_loc, c_loc) / max(query_loc, c_loc) >= self.LOC_RATIO_MIN

    # ------------------------------------------------------------------
    # Combined filter
    # ------------------------------------------------------------------

    def passes(self, query: dict, candidate: dict) -> bool:
        """Return True if candidate passes all fast filters."""
        q_loc = query["end_line"] - query["start_line"] + 1
        return (
            self.language_matches(query["language"], candidate)
            and self.param_count_within_one(len(query["parameters"]), candidate)
            and self.loc_within_range(q_loc, candidate)
        )

    def filter_candidates(self, query: dict, candidates: list[dict]) -> list[dict]:
        """Return only the candidates that pass all fast filters."""
        return [c for c in candidates if self.passes(query, c)]

    # ------------------------------------------------------------------
    # DB-level pre-filter — returns candidate DB IDs
    # ------------------------------------------------------------------

    def get_candidate_ids(self, registry: "MethodRegistry", query: dict) -> set[int]:
        """
        Query the index for DB IDs that pass fast filter criteria.

        Delegates to registry.filter_by_criteria, which queries SQLite
        directly for language + LOC match and then checks param count
        in Python (JSON column).

        Returns an empty set when no candidates pass or the index is empty.
        """
        q_loc = query["end_line"] - query["start_line"] + 1
        return registry.filter_by_criteria(
            language=query["language"],
            param_count=len(query["parameters"]),
            loc=q_loc,
        )
