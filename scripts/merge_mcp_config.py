"""Merge the code-similarity-mcp entry into a claude.json config file.

Used by install-global.sh (and callable from install-global.ps1) so that both
scripts share a single, tested implementation of the JSON merge logic.

CLI usage (called by the install scripts):
    python merge_mcp_config.py <config_path> <repo_root> <python_exe>

The script exits 0 on success and 1 on error (invalid JSON in the existing
file).  It does NOT print a success message — the caller is responsible for
user-facing output.
"""

from __future__ import annotations

import json
import pathlib
import sys


def merge_mcp_entry(
    config_path: pathlib.Path,
    repo_root: str,
    python_exe: str,
) -> None:
    """Read *config_path* (or create it), upsert the code-similarity entry, write back.

    Args:
        config_path: Path to ~/.claude.json (or %USERPROFILE%\\.claude.json).
        repo_root:   Absolute path to the code-similarity-mcp repository root.
        python_exe:  Absolute path to the venv Python interpreter.

    Raises:
        ValueError: If *config_path* exists but is not valid JSON.
    """
    entry: dict = {
        "type": "stdio",
        "command": python_exe,
        "args": ["-m", "code_similarity_mcp"],
        "cwd": repo_root,
    }

    if config_path.exists():
        try:
            config: dict = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse {config_path}: {exc}") from exc
    else:
        config = {}

    config.setdefault("mcpServers", {})
    config["mcpServers"]["code-similarity"] = entry

    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: merge_mcp_config.py <config_path> <repo_root> <python_exe>",
            file=sys.stderr,
        )
        sys.exit(1)

    _config_path, _repo_root, _python_exe = sys.argv[1], sys.argv[2], sys.argv[3]
    try:
        merge_mcp_entry(pathlib.Path(_config_path), _repo_root, _python_exe)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
