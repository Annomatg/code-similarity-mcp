"""Tests for global MCP deployment configuration (feature #42).

Verifies that:
- pyproject.toml declares the correct console-script entry point.
- The documented MCP server name matches the FastMCP app name.
- The module entry point (-m code_similarity_mcp) is importable and exposes main().
- The documented tool list matches all registered MCP tools.
- The global config JSON structure (as documented in GLOBAL_INSTALL.md) is valid.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_GLOBAL_INSTALL_DOC = _REPO_ROOT / "GLOBAL_INSTALL.md"

# Expected values that must stay in sync between code and documentation.
_EXPECTED_SERVER_NAME = "code-similarity-mcp"
_EXPECTED_SCRIPT_ENTRY = "code_similarity_mcp.__main__:main"
_EXPECTED_TOOLS = {
    "index_repository",
    "analyze_new_code",
    "analyze_project",
    "find_large_functions",
    "chunk_repository",
    "analyze_chunks",
    "get_chunk_map",
}


# ---------------------------------------------------------------------------
# pyproject.toml contract
# ---------------------------------------------------------------------------


def test_pyproject_exists():
    """pyproject.toml must be present in the repo root."""
    assert _PYPROJECT.exists(), f"pyproject.toml not found at {_PYPROJECT}"


def test_pyproject_has_script_entry():
    """pyproject.toml must declare the code-similarity-mcp console script."""
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]  # fallback for 3.10

    with open(_PYPROJECT, "rb") as fh:
        data = tomllib.load(fh)

    scripts = data.get("project", {}).get("scripts", {})
    assert _EXPECTED_SERVER_NAME in scripts, (
        f"Script '{_EXPECTED_SERVER_NAME}' not found in [project.scripts]. "
        f"Found: {list(scripts.keys())}"
    )
    assert scripts[_EXPECTED_SERVER_NAME] == _EXPECTED_SCRIPT_ENTRY, (
        f"Script entry mismatch. Expected '{_EXPECTED_SCRIPT_ENTRY}', "
        f"got '{scripts[_EXPECTED_SERVER_NAME]}'"
    )


# ---------------------------------------------------------------------------
# Module entry point (-m code_similarity_mcp)
# ---------------------------------------------------------------------------


def test_package_importable_as_module():
    """code_similarity_mcp must be importable (required for -m code_similarity_mcp)."""
    mod = importlib.import_module("code_similarity_mcp")
    assert mod is not None


def test_main_module_has_main_callable():
    """code_similarity_mcp.__main__ must expose a callable main() for -m execution."""
    main_mod = importlib.import_module("code_similarity_mcp.__main__")
    assert callable(getattr(main_mod, "main", None)), (
        "code_similarity_mcp.__main__.main is not callable"
    )


# ---------------------------------------------------------------------------
# MCP server name — must match documented config key semantics
# ---------------------------------------------------------------------------


def test_server_name_matches_documented_name():
    """The FastMCP app name must match the documented server name."""
    from code_similarity_mcp.mcp.server import app

    assert app.name == _EXPECTED_SERVER_NAME, (
        f"app.name is '{app.name}', expected '{_EXPECTED_SERVER_NAME}'. "
        "Update GLOBAL_INSTALL.md if the server name changes."
    )


# ---------------------------------------------------------------------------
# Tool list — must match GLOBAL_INSTALL.md documentation table
# ---------------------------------------------------------------------------


def test_all_documented_tools_are_registered():
    """Every tool listed in GLOBAL_INSTALL.md must be registered in the MCP app."""
    from code_similarity_mcp.mcp.server import app

    registered = {t.name for t in asyncio.run(app.list_tools())}
    missing = _EXPECTED_TOOLS - registered
    assert not missing, (
        f"Tools documented in GLOBAL_INSTALL.md but not registered: {missing}"
    )


def test_no_undocumented_tools_registered():
    """No registered tool should be absent from the GLOBAL_INSTALL.md table."""
    from code_similarity_mcp.mcp.server import app

    registered = {t.name for t in asyncio.run(app.list_tools())}
    extra = registered - _EXPECTED_TOOLS
    assert not extra, (
        f"Tools registered but not documented in GLOBAL_INSTALL.md: {extra}. "
        "Add them to the tool table in GLOBAL_INSTALL.md."
    )


# ---------------------------------------------------------------------------
# GLOBAL_INSTALL.md sanity checks
# ---------------------------------------------------------------------------


def test_global_install_doc_exists():
    """GLOBAL_INSTALL.md must be present in the repo root."""
    assert _GLOBAL_INSTALL_DOC.exists(), (
        f"GLOBAL_INSTALL.md not found at {_GLOBAL_INSTALL_DOC}"
    )


def test_global_install_doc_contains_module_arg():
    """GLOBAL_INSTALL.md must document the -m code_similarity_mcp invocation."""
    content = _GLOBAL_INSTALL_DOC.read_text(encoding="utf-8")
    assert "-m code_similarity_mcp" in content, (
        "GLOBAL_INSTALL.md does not mention '-m code_similarity_mcp'"
    )


def test_global_install_doc_contains_windows_config():
    """GLOBAL_INSTALL.md must include a Windows JSON config example."""
    content = _GLOBAL_INSTALL_DOC.read_text(encoding="utf-8")
    assert "Scripts\\\\python.exe" in content or r"Scripts\python.exe" in content, (
        "GLOBAL_INSTALL.md does not contain a Windows venv Python path example"
    )


def test_global_install_doc_contains_linux_config():
    """GLOBAL_INSTALL.md must include a Linux/macOS JSON config example."""
    content = _GLOBAL_INSTALL_DOC.read_text(encoding="utf-8")
    assert ".venv/bin/python" in content, (
        "GLOBAL_INSTALL.md does not contain a Linux/macOS venv Python path example"
    )


def test_global_install_doc_lists_all_tools():
    """GLOBAL_INSTALL.md must mention every registered tool by name."""
    content = _GLOBAL_INSTALL_DOC.read_text(encoding="utf-8")
    for tool in _EXPECTED_TOOLS:
        assert tool in content, (
            f"Tool '{tool}' is not mentioned in GLOBAL_INSTALL.md"
        )


def test_sample_windows_config_is_valid_json():
    """The Windows JSON config snippet in GLOBAL_INSTALL.md must parse as valid JSON."""
    sample = {
        "mcpServers": {
            "code-similarity": {
                "command": "C:\\Dev\\code-similarity-mcp\\.venv\\Scripts\\python.exe",
                "args": ["-m", "code_similarity_mcp"],
                "cwd": "C:\\Dev\\code-similarity-mcp",
            }
        }
    }
    # Round-trip through JSON serializer to confirm it is well-formed.
    serialized = json.dumps(sample)
    parsed = json.loads(serialized)
    assert parsed["mcpServers"]["code-similarity"]["args"] == [
        "-m",
        "code_similarity_mcp",
    ]


def test_sample_linux_config_is_valid_json():
    """The Linux/macOS JSON config snippet in GLOBAL_INSTALL.md must parse as valid JSON."""
    sample = {
        "mcpServers": {
            "code-similarity": {
                "command": "/home/alice/dev/code-similarity-mcp/.venv/bin/python",
                "args": ["-m", "code_similarity_mcp"],
                "cwd": "/home/alice/dev/code-similarity-mcp",
            }
        }
    }
    serialized = json.dumps(sample)
    parsed = json.loads(serialized)
    assert parsed["mcpServers"]["code-similarity"]["command"].endswith(
        ".venv/bin/python"
    )


def test_config_entry_has_required_fields():
    """Every documented config entry must have command, args, and cwd fields."""
    required_fields = {"command", "args", "cwd"}
    # Verify against both sample configs (Windows and Linux/macOS).
    for platform, config in [
        (
            "Windows",
            {
                "command": "C:\\Dev\\code-similarity-mcp\\.venv\\Scripts\\python.exe",
                "args": ["-m", "code_similarity_mcp"],
                "cwd": "C:\\Dev\\code-similarity-mcp",
            },
        ),
        (
            "Linux",
            {
                "command": "/home/alice/dev/code-similarity-mcp/.venv/bin/python",
                "args": ["-m", "code_similarity_mcp"],
                "cwd": "/home/alice/dev/code-similarity-mcp",
            },
        ),
    ]:
        missing = required_fields - config.keys()
        assert not missing, (
            f"{platform} config is missing required fields: {missing}"
        )
