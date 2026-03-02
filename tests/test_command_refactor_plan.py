"""Tests for the /refactor-plan command prompt file.

Verifies that the prompt contains correct fallback instructions for when
the feature-mcp server (feature_create tool) is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_COMMAND_PATH = Path(__file__).parent.parent / ".claude" / "commands" / "refactor-plan.md"


@pytest.fixture(scope="module")
def command_text() -> str:
    assert _COMMAND_PATH.exists(), f"Command file not found: {_COMMAND_PATH}"
    return _COMMAND_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# File existence and core structure
# ---------------------------------------------------------------------------


class TestCommandFileStructure:
    def test_command_file_exists(self):
        assert _COMMAND_PATH.exists(), f"Command file not found: {_COMMAND_PATH}"

    def test_command_file_not_empty(self, command_text):
        assert len(command_text.strip()) > 100

    def test_command_describes_analyze_project_step(self, command_text):
        assert "analyze_project" in command_text

    def test_command_describes_feature_create_step(self, command_text):
        assert "feature_create" in command_text

    def test_command_has_rules_section(self, command_text):
        assert "## Rules" in command_text


# ---------------------------------------------------------------------------
# Fallback section existence
# ---------------------------------------------------------------------------


class TestFallbackSectionExists:
    def test_fallback_section_heading(self, command_text):
        """Must have a dedicated fallback section."""
        assert "## Fallback" in command_text or "fallback" in command_text.lower()

    def test_fallback_mentions_feature_create_unavailable(self, command_text):
        """Fallback must trigger when feature_create is not available."""
        lower = command_text.lower()
        assert "feature_create" in command_text
        assert "unavailable" in lower or "not available" in lower or "does not exist" in lower

    def test_fallback_mentions_feature_mcp(self, command_text):
        """Fallback must name the feature-mcp server so users know what to install."""
        assert "feature-mcp" in command_text

    def test_fallback_does_not_abort_silently(self, command_text):
        """Must explicitly instruct the agent not to abort silently."""
        lower = command_text.lower()
        assert "not abort" in lower or "do not abort" in lower or "don't abort" in lower or "silently" in lower


# ---------------------------------------------------------------------------
# Fallback note content
# ---------------------------------------------------------------------------


class TestFallbackNote:
    def test_fallback_note_instructs_how_to_configure(self, command_text):
        """Note must explain how to configure feature-mcp."""
        lower = command_text.lower()
        assert "configure" in lower or "configuration" in lower or "install" in lower

    def test_fallback_note_references_global_config(self, command_text):
        """Note must point to global MCP configuration."""
        lower = command_text.lower()
        assert "global" in lower or "claude_desktop_config" in lower or "~/.claude" in lower

    def test_fallback_note_provides_setup_link_or_reference(self, command_text):
        """Note must include a link or reference to setup instructions."""
        # Accept either a markdown link, a file path, or a documentation reference
        assert (
            "http" in command_text
            or "docs" in command_text.lower()
            or "README" in command_text
            or "setup" in command_text.lower()
        )


# ---------------------------------------------------------------------------
# Fallback markdown table output
# ---------------------------------------------------------------------------


class TestFallbackMarkdownTable:
    def test_fallback_instructs_markdown_table_output(self, command_text):
        """Fallback must produce a markdown table (|---|)."""
        assert "|---|" in command_text or "| --- |" in command_text or "markdown table" in command_text.lower()

    def test_fallback_table_includes_score_column(self, command_text):
        """Fallback table must include similarity score."""
        assert "Score" in command_text or "score" in command_text

    def test_fallback_table_includes_method_names(self, command_text):
        """Fallback table must include method/function names."""
        assert "Method A" in command_text or "method_a" in command_text.lower() or "function name" in command_text.lower()

    def test_fallback_table_includes_file_paths(self, command_text):
        """Fallback table must include file paths for both methods."""
        assert "File A" in command_text or "file_a" in command_text.lower() or "file path" in command_text.lower()

    def test_fallback_table_includes_suggested_action(self, command_text):
        """Fallback table must include a recommended refactoring action."""
        assert "Suggested Action" in command_text or "action" in command_text.lower()

    def test_fallback_table_covers_exact_match(self, command_text):
        """Fallback must use exact_match to determine urgency."""
        assert "exact_match" in command_text

    def test_fallback_covers_top_10_pairs(self, command_text):
        """Fallback must output the top 10 similar pairs."""
        assert "10" in command_text

    def test_fallback_consolidate_immediately_action(self, command_text):
        """Exact-match pairs must be labelled 'Consolidate immediately'."""
        assert "Consolidate immediately" in command_text or "consolidate immediately" in command_text.lower()

    def test_fallback_extract_shared_function_action(self, command_text):
        """High-score pairs must suggest extracting a shared function."""
        lower = command_text.lower()
        assert "extract" in lower and "function" in lower

    def test_fallback_review_manually_action(self, command_text):
        """Lower-score pairs must suggest manual review."""
        lower = command_text.lower()
        assert "review" in lower


# ---------------------------------------------------------------------------
# Fallback summary
# ---------------------------------------------------------------------------


class TestFallbackSummary:
    def test_fallback_includes_summary_after_table(self, command_text):
        """Fallback must end with a summary of total pairs and score range."""
        lower = command_text.lower()
        assert "summary" in lower or "total pairs" in lower or "score range" in lower or "no backlog" in lower

    def test_fallback_reminds_no_entries_created(self, command_text):
        """Fallback must remind the user that no backlog entries were created."""
        lower = command_text.lower()
        assert "no backlog" in lower or "not created" in lower or "were not created" in lower or "could not be added" in lower
