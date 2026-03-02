---
paths:
  - "src/**/*.py"
  - "tests/**/*.py"
---

# Index Freshness Enforcement (REQUIRED)

## Trigger

Before calling `analyze_new_code`, `analyze_project`, or `analyze_chunks` in any session, you MUST ensure that `index_repository` has been called for the current project root in this session.

## Precondition Check

At the start of each session, track whether `index_repository` has been called for each project root. If you are about to call a similarity tool and `index_repository` has **not yet been called** for that project root:

1. Call `index_repository` first with `repository_root` set to the current working directory (project root).
2. Then proceed with the similarity check.

## Full Workflow (Canonical Order)

```
index_repository(repository_root=<project root>)   # once per session, if not done yet
  ↓
analyze_new_code / analyze_project / analyze_chunks
  ↓
report results
```

Combined with `duplication-check.md`, the full post-coding flow is:

```
[Edit or Write a Python function]
  ↓
index_repository(repository_root=<cwd>)             # if not indexed this session
  ↓
analyze_new_code(code_snippet=<new function>, repository_root=<cwd>)
  ↓
report duplication warnings (score >= 0.85)
```

## Why This Matters

Calling similarity tools against a stale or empty index produces false negatives — real duplicates are missed. `index_repository` is **idempotent**: it only re-indexes files that have changed since the last run, so calling it multiple times per session is safe and fast.

## Suppression

Only skip the pre-index step if:

- The user explicitly says "skip indexing" or "skip index_repository"
- You have already called `index_repository` for this project root in the current session
