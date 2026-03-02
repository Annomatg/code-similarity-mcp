Analyze this repository for code duplication and create Feature MCP entries for the top refactoring candidates.

## Sequence of MCP Tool Calls

### Step 1 — Index the repository

Call `index_repository` with the current project root as `repository_root`. Use the primary working directory from the environment (the repo root where this command was invoked).

### Step 2 — Analyze for similar pairs

Call `analyze_project` with `threshold=0.85`. Collect the returned `similar_pairs` list.

If `similar_pairs` is empty, report: "No similar methods found above threshold 0.85. No refactoring tasks created." and stop — do **not** call `feature_create`.

### Step 3 — Select top 10 offenders

Sort `similar_pairs` by `score` descending. Take the first 10 (or all if fewer than 10).

### Step 4 — Create a Feature entry for each pair

For each pair call `feature_create` with these exact field values:

| Field | Value |
|-------|-------|
| `category` | `"Refactoring"` |
| `name` | `"<method_a> in <file_a> duplicates <method_b> in <file_b>"` |
| `description` | See template below |
| `steps` | See list below |

**`name` template** — use short relative paths (strip the repo root prefix):
```
<method_a_name> in <relative_file_a> duplicates <method_b_name> in <relative_file_b>
```

**`description` template:**
```
Similarity score: <score> (exact_match: <true|false>)

Method A: <method_a_name> in <relative_file_a> (line <method_a_line>)
Method B: <method_b_name> in <relative_file_b> (line <method_b_line>)

These two methods have been identified as highly similar by the code-similarity-mcp
analyzer. Refactoring them to share a common implementation will reduce duplication
and improve maintainability.

Refactoring hints: <refactoring_hints if present, otherwise "None">
```

**`steps` list** (always these 5 entries, substituting method/file names):
1. `"Identify the shared logic between <method_a_name> and <method_b_name>"`
2. `"Extract the shared logic into a new, well-named reusable function"`
3. `"Update <method_a_name> in <relative_file_a> to delegate to the new function"`
4. `"Update <method_b_name> in <relative_file_b> to delegate to the new function"`
5. `"Run the test suite to verify no regressions"`

## Rules

- Do **not** read, grep, or modify any source files — use only MCP tool calls.
- Use file paths relative to the repository root in all names, descriptions, and steps.
- Process pairs in score-descending order so the most critical duplicates appear first in the backlog.
- Report a brief summary when finished: number of features created, score range covered.

## Fallback: feature-mcp unavailable

If `feature_create` is not available (tool call returns an error, the tool does not exist, or the feature MCP server is not configured), do **not** abort silently. Instead:

1. Output the following note at the top of your response:

   > **⚠ feature-mcp not found** — The `feature_create` tool is unavailable, so refactoring tasks could not be added to the backlog automatically.
   > To enable automatic backlog creation, configure the feature MCP server globally:
   > see [global MCP configuration instructions](https://github.com/anthropics/claude-code/blob/main/docs/mcp.md) and add `feature-mcp` to your `~/.claude/claude_desktop_config.json`.

2. Then output the top 10 similar pairs (or all if fewer) as a formatted markdown table with these columns:

   | # | Score | Exact | Method A | File A | Method B | File B | Suggested Action |
   |---|-------|-------|----------|--------|----------|--------|-----------------|

   - **#**: rank (1 = highest score)
   - **Score**: similarity score rounded to 3 decimal places
   - **Exact**: ✓ if `exact_match` is true, otherwise blank
   - **Method A / Method B**: function name only
   - **File A / File B**: relative file path (strip repo root prefix)
   - **Suggested Action**: use `exact_match` and score to pick one of:
     - `exact_match=true` → `"Consolidate immediately"`
     - `score >= 0.95` → `"Extract shared function"`
     - `score >= 0.85` → `"Review and extract if domain matches"`
     - otherwise → `"Review manually"`

3. After the table add a plain-text summary: total pairs found, score range, and a reminder that the analysis is complete but no backlog entries were created.
