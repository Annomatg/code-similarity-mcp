---
paths:
  - "src/**/*.py"
  - "tests/**/*.py"
---

# Post-Coding Duplication Check (REQUIRED)

## Trigger

After any **Edit** or **Write** tool call that creates or modifies a Python function (`def` statement), you MUST run a duplication check before marking the task or feature as done.

## Action

Call `analyze_new_code` with:
- `code_snippet`: the full source of the new or modified function
- `repository_root`: the current working directory (project root)

## Threshold

If any result has `similarity_score >= 0.85`, you MUST report it as a warning to the user **before** completing the task.

## Report Format

For each match above threshold, output:

```
DUPLICATION WARNING:
  Match: <method_name> in <file_path>
  Score: <score>
  Suggestion: <one-line suggestion, e.g. "Consider reusing or extending <method_name> instead of duplicating it.">
```

## Suppression

Only skip this check if:
- The project index is empty (nothing has been indexed yet)
- The function is a test helper that intentionally mirrors production code
- The user explicitly says "skip duplication check"
