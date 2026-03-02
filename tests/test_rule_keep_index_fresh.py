"""Tests for .claude/rules/keep-index-fresh.md.

Verifies that the rule file exists and contains all required sections,
mandatory language, and correct configuration for the index freshness
enforcement workflow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_RULE_PATH = Path(__file__).parent.parent / ".claude" / "rules" / "keep-index-fresh.md"


@pytest.fixture(scope="module")
def rule_text() -> str:
    assert _RULE_PATH.exists(), f"Rule file not found: {_RULE_PATH}"
    return _RULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# File existence and basic structure
# ---------------------------------------------------------------------------


class TestRuleFileStructure:
    def test_rule_file_exists(self):
        assert _RULE_PATH.exists(), f"Rule file not found: {_RULE_PATH}"

    def test_rule_file_not_empty(self, rule_text):
        assert len(rule_text.strip()) > 100

    def test_rule_has_yaml_frontmatter(self, rule_text):
        assert rule_text.startswith("---"), "Rule must have YAML frontmatter"

    def test_rule_frontmatter_has_paths(self, rule_text):
        assert "paths:" in rule_text, "Frontmatter must include paths filter"

    def test_rule_applies_to_python_source_files(self, rule_text):
        assert "src/**/*.py" in rule_text or "**/*.py" in rule_text


# ---------------------------------------------------------------------------
# Mandatory language (MUST, not SHOULD)
# ---------------------------------------------------------------------------


class TestMandatoryLanguage:
    def test_rule_uses_must_keyword(self, rule_text):
        """Rule must use MUST (not just SHOULD) to indicate mandatory action."""
        assert "MUST" in rule_text

    def test_rule_requires_precondition_before_similarity_tools(self, rule_text):
        lower = rule_text.lower()
        assert "before" in lower and (
            "calling" in lower or "call" in lower
        )


# ---------------------------------------------------------------------------
# Trigger specification
# ---------------------------------------------------------------------------


class TestTriggerSpecification:
    def test_rule_mentions_analyze_new_code(self, rule_text):
        assert "analyze_new_code" in rule_text

    def test_rule_mentions_analyze_project(self, rule_text):
        assert "analyze_project" in rule_text

    def test_rule_mentions_index_repository(self, rule_text):
        assert "index_repository" in rule_text


# ---------------------------------------------------------------------------
# Precondition / session tracking
# ---------------------------------------------------------------------------


class TestPreconditionSpecification:
    def test_rule_mentions_session(self, rule_text):
        lower = rule_text.lower()
        assert "session" in lower

    def test_rule_specifies_per_project_root(self, rule_text):
        lower = rule_text.lower()
        assert "project root" in lower or "repository_root" in lower

    def test_rule_specifies_call_once_per_session(self, rule_text):
        lower = rule_text.lower()
        assert "once" in lower or "not yet" in lower or "already" in lower

    def test_rule_specifies_cwd_as_project_root(self, rule_text):
        lower = rule_text.lower()
        assert (
            "cwd" in lower
            or "current working directory" in lower
            or "project root" in lower
        )


# ---------------------------------------------------------------------------
# Idempotency note
# ---------------------------------------------------------------------------


class TestIdempotencyNote:
    def test_rule_notes_idempotent(self, rule_text):
        lower = rule_text.lower()
        assert "idempotent" in lower

    def test_rule_notes_safe_to_rerun(self, rule_text):
        lower = rule_text.lower()
        assert "safe" in lower or "multiple times" in lower


# ---------------------------------------------------------------------------
# Integration with duplication-check workflow
# ---------------------------------------------------------------------------


class TestWorkflowIntegration:
    def test_rule_mentions_duplication_check(self, rule_text):
        lower = rule_text.lower()
        assert "duplication" in lower or "duplication-check" in lower

    def test_rule_shows_canonical_order(self, rule_text):
        """In the workflow diagram, index_repository must appear before analyze_new_code."""
        # Find the workflow/canonical-order section
        workflow_start = rule_text.find("## Full Workflow")
        if workflow_start == -1:
            workflow_start = rule_text.find("Canonical Order")
        if workflow_start == -1:
            workflow_start = rule_text.find("index_repository")
        workflow_section = rule_text[workflow_start:]
        idx_repo = workflow_section.find("index_repository")
        idx_analyze = workflow_section.find("analyze_new_code")
        assert idx_repo != -1 and idx_analyze != -1
        assert idx_repo < idx_analyze, (
            "In the workflow section, index_repository should appear before analyze_new_code"
        )


# ---------------------------------------------------------------------------
# False-negative rationale
# ---------------------------------------------------------------------------


class TestRationale:
    def test_rule_explains_false_negatives(self, rule_text):
        lower = rule_text.lower()
        assert "false negative" in lower or "stale" in lower or "missing" in lower


# ---------------------------------------------------------------------------
# Suppression / escape hatch
# ---------------------------------------------------------------------------


class TestSuppression:
    def test_rule_allows_explicit_skip(self, rule_text):
        """Rule must allow user to explicitly skip the indexing step."""
        lower = rule_text.lower()
        assert "skip" in lower or "suppress" in lower or "opt-out" in lower

    def test_rule_allows_skip_if_already_indexed(self, rule_text):
        lower = rule_text.lower()
        assert "already" in lower or "current session" in lower
