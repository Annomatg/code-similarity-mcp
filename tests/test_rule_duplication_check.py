"""Tests for .claude/rules/duplication-check.md.

Verifies that the rule file exists and contains all required sections,
mandatory language, and correct configuration for the post-coding
duplication check workflow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_RULE_PATH = Path(__file__).parent.parent / ".claude" / "rules" / "duplication-check.md"


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

    def test_rule_not_only_should(self, rule_text):
        """MUST should dominate; rule should not rely only on SHOULD."""
        # Having MUST is required; SHOULD can also appear but MUST is needed
        assert "MUST" in rule_text

    def test_rule_requires_action_before_completing_task(self, rule_text):
        lower = rule_text.lower()
        assert "before" in lower and (
            "completing" in lower or "marking" in lower or "done" in lower
        )


# ---------------------------------------------------------------------------
# Trigger specification
# ---------------------------------------------------------------------------


class TestTriggerSpecification:
    def test_rule_mentions_edit_tool(self, rule_text):
        assert "Edit" in rule_text

    def test_rule_mentions_write_tool(self, rule_text):
        assert "Write" in rule_text

    def test_rule_mentions_python_function(self, rule_text):
        lower = rule_text.lower()
        assert "function" in lower or "def" in lower

    def test_rule_mentions_creates_or_modifies(self, rule_text):
        lower = rule_text.lower()
        assert "creat" in lower or "modif" in lower


# ---------------------------------------------------------------------------
# Action specification
# ---------------------------------------------------------------------------


class TestActionSpecification:
    def test_rule_specifies_analyze_new_code(self, rule_text):
        assert "analyze_new_code" in rule_text

    def test_rule_specifies_code_snippet_param(self, rule_text):
        assert "code_snippet" in rule_text

    def test_rule_specifies_repository_root_param(self, rule_text):
        assert "repository_root" in rule_text

    def test_rule_uses_cwd_as_repository_root(self, rule_text):
        lower = rule_text.lower()
        assert "cwd" in lower or "current working directory" in lower or "project root" in lower


# ---------------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------------


class TestThresholdConfiguration:
    def test_rule_specifies_085_threshold(self, rule_text):
        assert "0.85" in rule_text

    def test_rule_reports_matches_above_threshold(self, rule_text):
        lower = rule_text.lower()
        assert "warning" in lower or "report" in lower or "warn" in lower


# ---------------------------------------------------------------------------
# Report format
# ---------------------------------------------------------------------------


class TestReportFormat:
    def test_rule_includes_method_name_in_report(self, rule_text):
        lower = rule_text.lower()
        assert "method" in lower or "function" in lower

    def test_rule_includes_file_path_in_report(self, rule_text):
        lower = rule_text.lower()
        assert "file" in lower

    def test_rule_includes_score_in_report(self, rule_text):
        lower = rule_text.lower()
        assert "score" in lower

    def test_rule_includes_suggestion_in_report(self, rule_text):
        lower = rule_text.lower()
        assert "suggestion" in lower or "suggest" in lower


# ---------------------------------------------------------------------------
# Suppression / escape hatch
# ---------------------------------------------------------------------------


class TestSuppression:
    def test_rule_allows_explicit_skip(self, rule_text):
        """Rule must allow user to explicitly skip the check."""
        lower = rule_text.lower()
        assert "skip" in lower or "suppress" in lower or "opt-out" in lower
