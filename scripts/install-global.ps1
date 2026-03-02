<#
.SYNOPSIS
    Register code-similarity-mcp as a global MCP server in %USERPROFILE%\.claude.json.

.DESCRIPTION
    Reads (or creates) %USERPROFILE%\.claude.json, merges the code-similarity
    MCP server entry without clobbering any existing entries, then writes the
    file back.  JSON manipulation is done with native PowerShell ConvertFrom-Json /
    ConvertTo-Json (compatible with PowerShell 5.1 and PowerShell 7+).

.EXAMPLE
    .\scripts\install-global.ps1
    # Run from the repo root or any location — $PSScriptRoot is used to locate the repo.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

# $PSScriptRoot is the directory that contains this script (scripts/).
# The repo root is one level up.
$RepoRoot   = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonExe  = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$MergeScript = Join-Path $PSScriptRoot "merge_mcp_config.py"
$ConfigFile = Join-Path $env:USERPROFILE ".claude.json"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

if (-not (Test-Path $PythonExe)) {
    Write-Error @"
Python venv not found at: $PythonExe

Run the following from the repo root first:
    .venv\Scripts\pip install -e .
"@
    exit 1
}

if (-not (Test-Path $MergeScript)) {
    Write-Error "merge_mcp_config.py not found at: $MergeScript"
    exit 1
}

# ---------------------------------------------------------------------------
# Merge the MCP entry via the shared Python helper
# ---------------------------------------------------------------------------

& $PythonExe $MergeScript $ConfigFile $RepoRoot $PythonExe
if ($LASTEXITCODE -ne 0) {
    Write-Error "merge_mcp_config.py exited with code $LASTEXITCODE"
    exit $LASTEXITCODE
}

# ---------------------------------------------------------------------------
# Success message
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "Success! code-similarity-mcp registered in: $ConfigFile"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Restart Claude Code."
Write-Host "  2. The following MCP tools will be available globally:"
Write-Host "       index_repository, analyze_new_code, analyze_project,"
Write-Host "       find_large_functions, chunk_repository, analyze_chunks, get_chunk_map"
Write-Host ""
