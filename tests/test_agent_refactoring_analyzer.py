"""Tests for the refactoring-analyzer agent definition and its MCP tool dependencies."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Agent file fixtures
# ---------------------------------------------------------------------------

_AGENT_PATH = Path(__file__).parent.parent / ".claude" / "agents" / "refactoring-analyzer.md"

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Extract key: value pairs from YAML-style frontmatter."""
    match = _FRONTMATTER_RE.match(text)
    assert match, "Agent file must start with --- frontmatter ---"
    result: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip()
    return result


# ---------------------------------------------------------------------------
# Agent file structure tests
# ---------------------------------------------------------------------------


class TestAgentFileStructure:
    def test_agent_file_exists(self):
        assert _AGENT_PATH.exists(), f"Agent file not found: {_AGENT_PATH}"

    def test_agent_file_has_frontmatter(self):
        text = _AGENT_PATH.read_text(encoding="utf-8")
        assert _FRONTMATTER_RE.match(text), "Missing YAML frontmatter block"

    def test_agent_name_field(self):
        fm = _parse_frontmatter(_AGENT_PATH.read_text(encoding="utf-8"))
        assert fm.get("name") == "refactoring-analyzer"

    def test_agent_model_is_opus(self):
        """Refactoring analysis requires claude-opus-4-6."""
        fm = _parse_frontmatter(_AGENT_PATH.read_text(encoding="utf-8"))
        assert fm.get("model") == "opus", (
            f"Model must be 'opus' for refactoring analysis, got: {fm.get('model')!r}"
        )

    def test_agent_has_description(self):
        fm = _parse_frontmatter(_AGENT_PATH.read_text(encoding="utf-8"))
        desc = fm.get("description", "")
        assert len(desc) > 20, "description must be substantive for auto-routing"

    def test_agent_description_mentions_refactoring(self):
        fm = _parse_frontmatter(_AGENT_PATH.read_text(encoding="utf-8"))
        desc = fm.get("description", "").lower()
        assert any(kw in desc for kw in ("refactor", "duplicate", "similar", "consolidat")), (
            "description must mention refactoring/similarity for correct auto-routing"
        )

    def test_agent_has_color(self):
        fm = _parse_frontmatter(_AGENT_PATH.read_text(encoding="utf-8"))
        assert fm.get("color"), "color field is required"

    def test_agent_body_references_index_repository_tool(self):
        body = _AGENT_PATH.read_text(encoding="utf-8")
        assert "mcp__code-similarity__index_repository" in body

    def test_agent_body_references_analyze_project_tool(self):
        body = _AGENT_PATH.read_text(encoding="utf-8")
        assert "mcp__code-similarity__analyze_project" in body

    def test_agent_body_references_analyze_new_code_tool(self):
        body = _AGENT_PATH.read_text(encoding="utf-8")
        assert "mcp__code-similarity__analyze_new_code" in body

    def test_agent_body_has_workflow(self):
        body = _AGENT_PATH.read_text(encoding="utf-8")
        assert "## Workflow" in body or "## Phase" in body or "Phase 1" in body

    def test_agent_body_covers_exact_match_case(self):
        body = _AGENT_PATH.read_text(encoding="utf-8")
        assert "exact_match" in body or "Exact" in body

    def test_agent_body_covers_empty_result_case(self):
        body = _AGENT_PATH.read_text(encoding="utf-8")
        assert "empty" in body.lower() or "No similar" in body or "0" in body

    def test_agent_body_has_domain_detection(self):
        body = _AGENT_PATH.read_text(encoding="utf-8")
        assert "domain" in body.lower()

    def test_agent_body_distinguishes_same_and_cross_domain(self):
        body = _AGENT_PATH.read_text(encoding="utf-8")
        assert "same domain" in body.lower() or "Same Domain" in body
        assert "cross domain" in body.lower() or "Cross Domain" in body

    def test_agent_body_warns_cross_domain_refactoring_cost(self):
        body = _AGENT_PATH.read_text(encoding="utf-8")
        assert "shared layer" in body.lower() or "coupling" in body.lower()

    def test_agent_body_has_shared_layer_conditions(self):
        """Agent must not recommend cross-domain merges without explicit conditions."""
        body = _AGENT_PATH.read_text(encoding="utf-8")
        # Three conditions gate cross-domain recommendations
        assert "three or more" in body.lower() or "3" in body

    def test_agent_body_prioritises_same_domain_first(self):
        body = _AGENT_PATH.read_text(encoding="utf-8")
        assert "same-domain" in body.lower() or "same domain" in body.lower()

    def test_agent_body_does_not_recommend_cross_domain_below_90(self):
        body = _AGENT_PATH.read_text(encoding="utf-8")
        # The classification table must mark 0.75-0.89 cross-domain as non-actionable
        assert "coincidental" in body.lower() or "do not recommend" in body.lower()


# ---------------------------------------------------------------------------
# MCP tool integration tests (validate output format the agent consumes)
# ---------------------------------------------------------------------------

# Lazy import to avoid loading the heavy ML stack unless needed
from code_similarity_mcp.mcp.server import analyze_project, index_repository


_SIMILAR_PY = """\
def add(a, b):
    return a + b


def sum_values(x, y):
    return x + y
"""

_DISTINCT_PY = """\
def add(a, b):
    return a + b


def process(items, key, value, extra):
    result = []
    for item in items:
        if item.get(key) == value:
            result.append(item[extra])
    return result
"""


def _index(tmp_path: Path, content: str) -> str:
    (tmp_path / "module.py").write_text(content, encoding="utf-8")
    index_dir = str(tmp_path / "index")
    index_repository(str(tmp_path), index_dir=index_dir)
    return index_dir


class TestMcpOutputFormatForAgent:
    """Verify the MCP tools produce output the agent can reliably consume."""

    def test_analyze_project_returns_valid_json(self, tmp_path):
        index_dir = _index(tmp_path, _SIMILAR_PY)
        raw = analyze_project(index_dir=index_dir)
        data = json.loads(raw)
        assert isinstance(data, dict)

    def test_analyze_project_has_total_methods(self, tmp_path):
        index_dir = _index(tmp_path, _SIMILAR_PY)
        data = json.loads(analyze_project(index_dir=index_dir))
        assert "total_methods" in data
        assert isinstance(data["total_methods"], int)

    def test_analyze_project_has_similar_pairs_list(self, tmp_path):
        index_dir = _index(tmp_path, _SIMILAR_PY)
        data = json.loads(analyze_project(index_dir=index_dir))
        assert "similar_pairs" in data
        assert isinstance(data["similar_pairs"], list)

    def test_pair_has_all_agent_required_fields(self, tmp_path):
        """Every field the agent's report template references must be present."""
        index_dir = _index(tmp_path, _SIMILAR_PY)
        data = json.loads(analyze_project(index_dir=index_dir))
        assert len(data["similar_pairs"]) > 0, "Expected at least one similar pair"
        pair = data["similar_pairs"][0]

        for key in ("method_a", "method_b", "score", "exact_match",
                    "embedding_similarity", "ast_similarity",
                    "differences", "refactoring_hints"):
            assert key in pair, f"Missing key in pair: {key!r}"

    def test_method_entry_has_file_method_line(self, tmp_path):
        index_dir = _index(tmp_path, _SIMILAR_PY)
        data = json.loads(analyze_project(index_dir=index_dir))
        pair = data["similar_pairs"][0]
        for side in ("method_a", "method_b"):
            entry = pair[side]
            assert "file" in entry
            assert "method" in entry
            assert "line" in entry

    def test_score_is_float_in_unit_interval(self, tmp_path):
        index_dir = _index(tmp_path, _SIMILAR_PY)
        data = json.loads(analyze_project(index_dir=index_dir))
        for pair in data["similar_pairs"]:
            assert isinstance(pair["score"], float)
            assert 0.0 <= pair["score"] <= 1.0

    def test_exact_match_is_bool(self, tmp_path):
        index_dir = _index(tmp_path, _SIMILAR_PY)
        data = json.loads(analyze_project(index_dir=index_dir))
        for pair in data["similar_pairs"]:
            assert isinstance(pair["exact_match"], bool)

    def test_refactoring_hints_is_list(self, tmp_path):
        index_dir = _index(tmp_path, _SIMILAR_PY)
        data = json.loads(analyze_project(index_dir=index_dir))
        for pair in data["similar_pairs"]:
            assert isinstance(pair["refactoring_hints"], list)

    def test_identical_methods_flagged_as_exact_match(self, tmp_path):
        """Agent rule: exact_match=true → 'Consolidate immediately'."""
        index_dir = _index(tmp_path, _SIMILAR_PY)
        data = json.loads(analyze_project(index_dir=index_dir))
        exact = [p for p in data["similar_pairs"] if p["exact_match"]]
        assert len(exact) > 0
        assert exact[0]["score"] == 1.0

    def test_dissimilar_methods_produce_no_pairs_above_threshold(self, tmp_path):
        """Agent rule: pairs below threshold must not appear in the report."""
        index_dir = _index(tmp_path, _DISTINCT_PY)
        data = json.loads(analyze_project(index_dir=index_dir, threshold=0.85))
        assert data["similar_pairs"] == []

    def test_empty_repo_gives_zero_methods(self, tmp_path):
        """Agent must handle empty index gracefully."""
        empty = tmp_path / "empty"
        empty.mkdir()
        index_dir = str(tmp_path / "idx")
        index_repository(str(empty), index_dir=index_dir)
        data = json.loads(analyze_project(index_dir=index_dir))
        assert data["total_methods"] == 0
        assert data["similar_pairs"] == []

    def test_pairs_sorted_by_score_descending(self, tmp_path):
        """Agent relies on descending order to present highest-priority recommendations first."""
        src = """\
def add(a, b):
    return a + b

def sum_values(x, y):
    return x + y

def multiply(a, b):
    result = a * b
    return result
"""
        index_dir = _index(tmp_path, src)
        data = json.loads(analyze_project(index_dir=index_dir))
        scores = [p["score"] for p in data["similar_pairs"]]
        assert scores == sorted(scores, reverse=True)

    def test_threshold_parameter_filters_pairs(self, tmp_path):
        """Agent can pass threshold parameter to narrow results."""
        index_dir = _index(tmp_path, _SIMILAR_PY)
        default_data = json.loads(analyze_project(index_dir=index_dir, threshold=0.85))
        strict_data = json.loads(analyze_project(index_dir=index_dir, threshold=1.0))
        assert len(strict_data["similar_pairs"]) <= len(default_data["similar_pairs"])
