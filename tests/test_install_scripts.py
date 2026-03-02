"""Tests for the global install helper scripts (feature #43).

Verifies that:
- scripts/install-global.ps1 exists and contains required patterns.
- scripts/install-global.sh exists, has a shebang, and contains required patterns.
- scripts/merge_mcp_config.py exists and the merge logic is correct.
- GLOBAL_INSTALL.md references both scripts.

The Python merge logic (merge_mcp_config.merge_mcp_entry) is tested with a
temporary directory so the real ~/.claude.json is never touched.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_PS1_SCRIPT = _SCRIPTS_DIR / "install-global.ps1"
_SH_SCRIPT = _SCRIPTS_DIR / "install-global.sh"
_MERGE_HELPER = _SCRIPTS_DIR / "merge_mcp_config.py"
_GLOBAL_INSTALL_DOC = _REPO_ROOT / "GLOBAL_INSTALL.md"

# Make the scripts/ directory importable so we can import merge_mcp_config.
sys.path.insert(0, str(_SCRIPTS_DIR))
import merge_mcp_config  # noqa: E402  (must come after sys.path insert)

# ---------------------------------------------------------------------------
# Script file existence
# ---------------------------------------------------------------------------


def test_ps1_script_exists():
    """install-global.ps1 must be present in scripts/."""
    assert _PS1_SCRIPT.exists(), f"install-global.ps1 not found at {_PS1_SCRIPT}"


def test_sh_script_exists():
    """install-global.sh must be present in scripts/."""
    assert _SH_SCRIPT.exists(), f"install-global.sh not found at {_SH_SCRIPT}"


def test_merge_helper_exists():
    """merge_mcp_config.py must be present in scripts/."""
    assert _MERGE_HELPER.exists(), f"merge_mcp_config.py not found at {_MERGE_HELPER}"


# ---------------------------------------------------------------------------
# PS1 script content checks
# ---------------------------------------------------------------------------


def test_ps1_uses_psscriptroot():
    """install-global.ps1 must use $PSScriptRoot to detect the repo location."""
    content = _PS1_SCRIPT.read_text(encoding="utf-8")
    assert "PSScriptRoot" in content, "$PSScriptRoot not found in install-global.ps1"


def test_ps1_references_userprofile():
    """install-global.ps1 must reference %USERPROFILE% for the config file path."""
    content = _PS1_SCRIPT.read_text(encoding="utf-8")
    assert "USERPROFILE" in content, "$env:USERPROFILE not found in install-global.ps1"


def test_ps1_references_claude_json():
    """install-global.ps1 must reference .claude.json."""
    content = _PS1_SCRIPT.read_text(encoding="utf-8")
    assert ".claude.json" in content, ".claude.json not found in install-global.ps1"


def test_ps1_references_code_similarity_key():
    """install-global.ps1 must mention the 'code-similarity' server key."""
    content = _PS1_SCRIPT.read_text(encoding="utf-8")
    assert "code-similarity" in content, "'code-similarity' not found in install-global.ps1"


def test_ps1_references_merge_script():
    """install-global.ps1 must delegate to merge_mcp_config.py."""
    content = _PS1_SCRIPT.read_text(encoding="utf-8")
    assert "merge_mcp_config.py" in content, (
        "merge_mcp_config.py not referenced in install-global.ps1"
    )


def test_ps1_references_venv_python():
    """install-global.ps1 must locate the venv Python interpreter."""
    content = _PS1_SCRIPT.read_text(encoding="utf-8")
    assert ".venv" in content, ".venv not referenced in install-global.ps1"


def test_ps1_prints_next_steps():
    """install-global.ps1 must print next-step guidance after success."""
    content = _PS1_SCRIPT.read_text(encoding="utf-8")
    assert "Next steps" in content or "next steps" in content, (
        "install-global.ps1 does not print next steps"
    )


# ---------------------------------------------------------------------------
# SH script content checks
# ---------------------------------------------------------------------------


def test_sh_has_bash_shebang():
    """install-global.sh must start with a bash shebang."""
    content = _SH_SCRIPT.read_text(encoding="utf-8")
    first_line = content.splitlines()[0]
    assert first_line in ("#!/usr/bin/env bash", "#!/bin/bash"), (
        f"install-global.sh has unexpected shebang: {first_line!r}"
    )


def test_sh_uses_bash_source():
    """install-global.sh must use BASH_SOURCE to detect the script directory."""
    content = _SH_SCRIPT.read_text(encoding="utf-8")
    assert "BASH_SOURCE" in content, "BASH_SOURCE not found in install-global.sh"


def test_sh_references_home_claude_json():
    """install-global.sh must reference HOME/.claude.json for the config path."""
    content = _SH_SCRIPT.read_text(encoding="utf-8")
    assert ".claude.json" in content, ".claude.json not found in install-global.sh"


def test_sh_references_code_similarity_key():
    """install-global.sh must pass 'code-similarity' context or reference it."""
    content = _SH_SCRIPT.read_text(encoding="utf-8")
    assert "code-similarity" in content or "merge_mcp_config" in content, (
        "install-global.sh does not reference code-similarity or merge helper"
    )


def test_sh_references_merge_script():
    """install-global.sh must call merge_mcp_config.py."""
    content = _SH_SCRIPT.read_text(encoding="utf-8")
    assert "merge_mcp_config.py" in content, (
        "merge_mcp_config.py not referenced in install-global.sh"
    )


def test_sh_references_venv_python():
    """install-global.sh must locate the venv Python interpreter."""
    content = _SH_SCRIPT.read_text(encoding="utf-8")
    assert ".venv" in content, ".venv not referenced in install-global.sh"


def test_sh_prints_next_steps():
    """install-global.sh must print next-step guidance after success."""
    content = _SH_SCRIPT.read_text(encoding="utf-8")
    assert "Next steps" in content or "next steps" in content, (
        "install-global.sh does not print next steps"
    )


def test_sh_uses_set_euo_pipefail():
    """install-global.sh must use 'set -euo pipefail' for safety."""
    content = _SH_SCRIPT.read_text(encoding="utf-8")
    assert "set -euo pipefail" in content, (
        "'set -euo pipefail' not found in install-global.sh"
    )


# ---------------------------------------------------------------------------
# merge_mcp_config.py — unit tests for the merge logic
# ---------------------------------------------------------------------------


def test_merge_creates_config_when_missing(tmp_path: pathlib.Path):
    """merge_mcp_entry creates claude.json when the file does not exist."""
    config_file = tmp_path / ".claude.json"
    assert not config_file.exists()

    merge_mcp_config.merge_mcp_entry(config_file, "/repo", "/repo/.venv/bin/python")

    assert config_file.exists()
    data = json.loads(config_file.read_text(encoding="utf-8"))
    assert "mcpServers" in data
    assert "code-similarity" in data["mcpServers"]


def test_merge_entry_has_required_fields(tmp_path: pathlib.Path):
    """The written entry must contain command, args, and cwd."""
    config_file = tmp_path / ".claude.json"
    merge_mcp_config.merge_mcp_entry(config_file, "/repo", "/repo/.venv/bin/python")

    data = json.loads(config_file.read_text(encoding="utf-8"))
    entry = data["mcpServers"]["code-similarity"]
    assert "command" in entry
    assert "args" in entry
    assert "cwd" in entry


def test_merge_sets_correct_args(tmp_path: pathlib.Path):
    """args must be ['-m', 'code_similarity_mcp']."""
    config_file = tmp_path / ".claude.json"
    merge_mcp_config.merge_mcp_entry(config_file, "/repo", "/repo/.venv/bin/python")

    data = json.loads(config_file.read_text(encoding="utf-8"))
    assert data["mcpServers"]["code-similarity"]["args"] == ["-m", "code_similarity_mcp"]


def test_merge_sets_repo_root_as_cwd(tmp_path: pathlib.Path):
    """cwd must equal the repo_root argument passed to merge_mcp_entry."""
    config_file = tmp_path / ".claude.json"
    merge_mcp_config.merge_mcp_entry(config_file, "/my/repo", "/my/repo/.venv/bin/python")

    data = json.loads(config_file.read_text(encoding="utf-8"))
    assert data["mcpServers"]["code-similarity"]["cwd"] == "/my/repo"


def test_merge_sets_python_as_command(tmp_path: pathlib.Path):
    """command must equal the python_exe argument passed to merge_mcp_entry."""
    config_file = tmp_path / ".claude.json"
    merge_mcp_config.merge_mcp_entry(config_file, "/repo", "/repo/.venv/bin/python")

    data = json.loads(config_file.read_text(encoding="utf-8"))
    assert data["mcpServers"]["code-similarity"]["command"] == "/repo/.venv/bin/python"


def test_merge_preserves_existing_servers(tmp_path: pathlib.Path):
    """Merging must not clobber other mcpServers entries."""
    config_file = tmp_path / ".claude.json"
    existing = {
        "mcpServers": {
            "some-other-server": {"command": "/bin/other", "args": [], "cwd": "/other"}
        }
    }
    config_file.write_text(json.dumps(existing), encoding="utf-8")

    merge_mcp_config.merge_mcp_entry(config_file, "/repo", "/repo/.venv/bin/python")

    data = json.loads(config_file.read_text(encoding="utf-8"))
    assert "some-other-server" in data["mcpServers"], "existing server entry was clobbered"
    assert "code-similarity" in data["mcpServers"]


def test_merge_preserves_non_mcp_top_level_keys(tmp_path: pathlib.Path):
    """Other top-level keys in claude.json must survive the merge."""
    config_file = tmp_path / ".claude.json"
    existing = {"theme": "dark", "version": 2, "mcpServers": {}}
    config_file.write_text(json.dumps(existing), encoding="utf-8")

    merge_mcp_config.merge_mcp_entry(config_file, "/repo", "/repo/.venv/bin/python")

    data = json.loads(config_file.read_text(encoding="utf-8"))
    assert data["theme"] == "dark"
    assert data["version"] == 2


def test_merge_creates_mcp_servers_key_when_absent(tmp_path: pathlib.Path):
    """If mcpServers is absent, merge must create it."""
    config_file = tmp_path / ".claude.json"
    existing = {"theme": "dark"}
    config_file.write_text(json.dumps(existing), encoding="utf-8")

    merge_mcp_config.merge_mcp_entry(config_file, "/repo", "/repo/.venv/bin/python")

    data = json.loads(config_file.read_text(encoding="utf-8"))
    assert "mcpServers" in data
    assert "code-similarity" in data["mcpServers"]


def test_merge_upserts_existing_entry(tmp_path: pathlib.Path):
    """Re-running merge must update the entry (upsert), not duplicate it."""
    config_file = tmp_path / ".claude.json"

    merge_mcp_config.merge_mcp_entry(config_file, "/old/repo", "/old/.venv/bin/python")
    merge_mcp_config.merge_mcp_entry(config_file, "/new/repo", "/new/.venv/bin/python")

    data = json.loads(config_file.read_text(encoding="utf-8"))
    servers = data["mcpServers"]
    keys = list(servers.keys())
    assert keys.count("code-similarity") == 1, "code-similarity entry was duplicated"
    assert servers["code-similarity"]["cwd"] == "/new/repo", "cwd was not updated"


def test_merge_raises_on_invalid_json(tmp_path: pathlib.Path):
    """merge_mcp_entry must raise ValueError when the existing file is not valid JSON."""
    config_file = tmp_path / ".claude.json"
    config_file.write_text("{ not valid json }", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not parse"):
        merge_mcp_config.merge_mcp_entry(config_file, "/repo", "/repo/.venv/bin/python")


def test_merge_output_is_valid_json(tmp_path: pathlib.Path):
    """The file written by merge_mcp_entry must be parseable JSON."""
    config_file = tmp_path / ".claude.json"
    merge_mcp_config.merge_mcp_entry(config_file, "/repo", "/repo/.venv/bin/python")

    data = json.loads(config_file.read_text(encoding="utf-8"))
    assert isinstance(data, dict)


def test_merge_output_ends_with_newline(tmp_path: pathlib.Path):
    """The written file should end with a newline (unix convention)."""
    config_file = tmp_path / ".claude.json"
    merge_mcp_config.merge_mcp_entry(config_file, "/repo", "/repo/.venv/bin/python")

    raw = config_file.read_bytes()
    assert raw.endswith(b"\n"), "merge_mcp_config.py output does not end with newline"


def test_merge_from_empty_json_object(tmp_path: pathlib.Path):
    """Merge must succeed when the existing file contains just {}."""
    config_file = tmp_path / ".claude.json"
    config_file.write_text("{}", encoding="utf-8")

    merge_mcp_config.merge_mcp_entry(config_file, "/repo", "/repo/.venv/bin/python")

    data = json.loads(config_file.read_text(encoding="utf-8"))
    assert "code-similarity" in data["mcpServers"]


# ---------------------------------------------------------------------------
# GLOBAL_INSTALL.md must reference the install scripts
# ---------------------------------------------------------------------------


def test_global_install_doc_references_ps1_script():
    """GLOBAL_INSTALL.md must mention install-global.ps1."""
    content = _GLOBAL_INSTALL_DOC.read_text(encoding="utf-8")
    assert "install-global.ps1" in content, (
        "install-global.ps1 not referenced in GLOBAL_INSTALL.md"
    )


def test_global_install_doc_references_sh_script():
    """GLOBAL_INSTALL.md must mention install-global.sh."""
    content = _GLOBAL_INSTALL_DOC.read_text(encoding="utf-8")
    assert "install-global.sh" in content, (
        "install-global.sh not referenced in GLOBAL_INSTALL.md"
    )
