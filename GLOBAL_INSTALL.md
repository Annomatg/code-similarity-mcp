# Global MCP Installation Guide

Register `code-similarity-mcp` as a global MCP server so it is available in every Claude Code session without per-project configuration.

---

## Prerequisites

1. Clone the repository to a permanent location on your machine.
2. Create the virtual environment and install the package in editable mode:

```bash
cd /path/to/code-similarity-mcp   # your clone location
python -m venv .venv
# Windows
.venv\Scripts\pip install -e .
# Linux / macOS
.venv/bin/pip install -e .
```

---

## The MCP Server Entry

The server runs via the Python module entry point:

```
python -m code_similarity_mcp
```

It communicates over **stdio** (no HTTP port needed).

---

## Global Configuration

Edit `~/.claude.json` (Linux/macOS) or `%USERPROFILE%\.claude.json` (Windows) and add an `mcpServers` entry. If the file does not exist, create it.

### Linux / macOS

```json
{
  "mcpServers": {
    "code-similarity": {
      "command": "/path/to/code-similarity-mcp/.venv/bin/python",
      "args": ["-m", "code_similarity_mcp"],
      "cwd": "/path/to/code-similarity-mcp"
    }
  }
}
```

**Example** (if you cloned to `~/dev/code-similarity-mcp`):

```json
{
  "mcpServers": {
    "code-similarity": {
      "command": "/home/alice/dev/code-similarity-mcp/.venv/bin/python",
      "args": ["-m", "code_similarity_mcp"],
      "cwd": "/home/alice/dev/code-similarity-mcp"
    }
  }
}
```

### Windows

```json
{
  "mcpServers": {
    "code-similarity": {
      "command": "C:\\path\\to\\code-similarity-mcp\\.venv\\Scripts\\python.exe",
      "args": ["-m", "code_similarity_mcp"],
      "cwd": "C:\\path\\to\\code-similarity-mcp"
    }
  }
}
```

**Example** (if you cloned to `C:\Dev\code-similarity-mcp`):

```json
{
  "mcpServers": {
    "code-similarity": {
      "command": "C:\\Dev\\code-similarity-mcp\\.venv\\Scripts\\python.exe",
      "args": ["-m", "code_similarity_mcp"],
      "cwd": "C:\\Dev\\code-similarity-mcp"
    }
  }
}
```

### Field reference

| Field | Purpose |
|-------|---------|
| `command` | Absolute path to the venv Python interpreter |
| `args` | `["-m", "code_similarity_mcp"]` — runs the package as a module |
| `cwd` | Repo root; used as the base for any relative paths the server resolves |

---

## How to Find Your Paths

**Repo root** — the directory containing `pyproject.toml`:

```bash
# from inside the repo
pwd            # Linux/macOS
cd             # Windows (prints current directory)
```

**Venv Python** — after creating the venv:

```bash
# Linux/macOS
which .venv/bin/python     # or: realpath .venv/bin/python

# Windows (PowerShell)
(Resolve-Path .venv\Scripts\python.exe).Path
```

---

## Merging With an Existing ~/.claude.json

If `~/.claude.json` already has content, add the new server inside the existing `mcpServers` object:

```json
{
  "mcpServers": {
    "some-other-server": { "..." : "..." },
    "code-similarity": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "code_similarity_mcp"],
      "cwd": "/path/to/code-similarity-mcp"
    }
  }
}
```

---

## Verifying the Installation

Restart Claude Code after editing `~/.claude.json`. The following MCP tools should now appear in every session:

| Tool | Description |
|------|-------------|
| `index_repository` | Index all Python source files in a repository |
| `analyze_new_code` | Find similar methods for a code snippet |
| `analyze_project` | Report duplicate/similar method pairs across the repo |
| `find_large_functions` | List functions exceeding a statement-count threshold |
| `chunk_repository` | Split large functions into dependency-aware chunks |
| `analyze_chunks` | Search for similar stored chunks |
| `get_chunk_map` | Return the full chunk DAG for a function or file |

You can confirm the server is running by checking the log file written to:

- **Linux/macOS**: `~/.code-similarity-mcp/server.log`
- **Windows**: `%USERPROFILE%\.code-similarity-mcp\server.log`

---

## Troubleshooting

**Server does not appear in Claude Code**
- Ensure the JSON in `~/.claude.json` is valid (no trailing commas, correct backslash escaping on Windows).
- Check that `command` points to an existing file: `ls /path/to/.venv/bin/python` (or `Test-Path` on PowerShell).

**ImportError on startup**
- Confirm the package is installed: `.venv/bin/pip show code-similarity-mcp` should return package info.
- If not, re-run `pip install -e .` from the repo root using the venv pip.

**Server starts but no tools appear**
- Look in `~/.code-similarity-mcp/server.log` for error messages.
- Verify you are running Claude Code ≥ 1.0 which supports the stdio MCP transport.
