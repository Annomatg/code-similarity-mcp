#!/usr/bin/env bash
# Register code-similarity-mcp as a global MCP server in ~/.claude.json.
#
# Reads (or creates) ~/.claude.json, merges the code-similarity MCP server
# entry without clobbering any existing entries, then writes the file back.
# JSON manipulation is performed by the Python interpreter from the venv.
#
# Usage:
#   bash scripts/install-global.sh
#   # or, after chmod +x:
#   ./scripts/install-global.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

# BASH_SOURCE[0] is this script file; its parent is the scripts/ directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"
MERGE_SCRIPT="$SCRIPT_DIR/merge_mcp_config.py"
CONFIG_FILE="${HOME}/.claude.json"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python venv not found at: $PYTHON" >&2
    echo "       Run '.venv/bin/pip install -e .' from the repo root first." >&2
    exit 1
fi

if [ ! -f "$MERGE_SCRIPT" ]; then
    echo "ERROR: merge_mcp_config.py not found at: $MERGE_SCRIPT" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Merge the MCP entry via the shared Python helper
# ---------------------------------------------------------------------------

"$PYTHON" "$MERGE_SCRIPT" "$CONFIG_FILE" "$REPO_ROOT" "$PYTHON"

# ---------------------------------------------------------------------------
# Success message
# ---------------------------------------------------------------------------

echo ""
echo "Success! code-similarity-mcp registered in: $CONFIG_FILE"
echo ""
echo "Next steps:"
echo "  1. Restart Claude Code."
echo "  2. The following MCP tools will be available globally:"
echo "       index_repository, analyze_new_code, analyze_project,"
echo "       find_large_functions, chunk_repository, analyze_chunks, get_chunk_map"
echo ""
