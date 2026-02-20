MCP Tool: Cooperative Code Similarity & Refactoring Assistant
1. Purpose

An MCP tool that assists autonomous agents during code generation by:

Detecting potential duplicate methods

Suggesting semantically similar existing methods

Providing structured refactoring hints

Supporting architecture-aware consolidation

The tool does not enforce decisions.
It provides structured signals for agent reasoning.

2. Design Philosophy

Not a quality gate.
Not a static analyzer replacement.

It is a:

Cooperative architectural feedback system for code-generating agents.

Agents remain responsible for final decisions.

3. High-Level Architecture
Core Components

Parser & Normalizer

Structural Similarity Engine

Embedding Generator

Local Vector Index

MCP Interface Layer

Optional Refactoring Planner Agent

4. Technology Stack (Recommended)
Language: Python 3.11+

Reason:

Strong AST tooling

Mature ML ecosystem

Excellent vector DB integrations

Easy orchestration layer for MCP

Parsing & Normalization

Depends on target language.

For Python repos

ast (built-in)

libcst (better for round-tripping & canonicalization)

For TypeScript / JavaScript

tree-sitter (Python bindings)

or tree_sitter_languages

Tree-sitter is ideal if you want multi-language support.

Structural Analysis

Custom AST canonicalizer

Hashing via hashlib

Optional graph modeling via networkx

Embeddings

Options:

Lightweight & Local

sentence-transformers

Model: all-MiniLM-L6-v2

Code-specialized

microsoft/codebert-base

Salesforce/codet5-base

Start simple. Upgrade later.

Vector Store

For local repo use:

Option A (Simplest)

faiss (in-memory or disk-backed)

Option B (Persistent)

chromadb

If this is strictly local and per-repo:
? FAISS is enough.

Diff & Structural Comparison

difflib

redbaron (Python-specific)

Custom AST tree distance

MCP Layer

FastAPI

Pydantic (strict JSON schema validation)

Expose as:

CLI

HTTP endpoint

Direct tool binding for agent runtime

5. Data Model

Unit of analysis: Method / Function

{
  "id": "hash",
  "file_path": "src/module/file.ts",
  "language": "typescript",
  "signature": {
    "name": "calculateTotal",
    "parameters": 2,
    "return_type": "number"
  },
  "normalized_code": "...",
  "ast_fingerprint": "...",
  "embedding": [ ... ],
  "dependencies": ["taxUtil", "round"]
}

6. Normalization Strategy

Goal: remove superficial variation.

Minimum:

Rename local variables to placeholders (v1, v2, …)

Normalize literals (NUM_LITERAL, STR_LITERAL)

Remove comments

Canonical whitespace

Stronger version:

Canonical AST serialization

Deterministic ordering of independent statements

Inline trivial aliases

Without this step embeddings will produce noisy similarity.

7. Similarity Pipeline
Phase A — Fast Filtering

Reduce candidate set:

Same language

Parameter count ±1

LOC ±30%

Similar dependency footprint

Similar return structure

This avoids embedding comparisons across entire repo.

Phase B — Multi-Signal Similarity

Combine:

Exact hash match

AST fingerprint similarity

Embedding cosine similarity

Example scoring:

final_score =
0.5 * embedding_similarity +
0.3 * ast_similarity +
0.2 * structural_features


Threshold configurable (start around 0.88–0.92).

8. MCP Interface
Tool: analyze_new_code
Input
{
  "repository_root": "...",
  "changed_files": ["src/a.ts"],
  "code_snippet": "...",
  "language": "typescript"
}

Output
{
  "new_methods": [
    {
      "name": "calculateTotal",
      "candidates": [
        {
          "file": "src/billing/utils.ts",
          "method": "computeInvoiceTotal",
          "score": 0.93,
          "similarity_type": "semantic",
          "differences": [
            "tax rate hardcoded vs parameter",
            "rounding strategy differs"
          ],
          "refactoring_hints": [
            "extract shared tax calculation",
            "introduce rounding strategy parameter"
          ]
        }
      ]
    }
  ]
}


Important: provide structured differences, not just scores.

9. Refactoring Scan Mode

Tool: scan_refactoring_candidates

Purpose:

Full-repo clustering

Identify high-density similarity regions

Suggest refactoring strategy

Output:

{
  "clusters": [
    {
      "cluster_id": "c1",
      "methods": [...],
      "avg_similarity": 0.91,
      "suggested_strategy": "extract shared utility"
    }
  ]
}

10. Agent Interaction Pattern
During Code Generation

Agent generates method

MCP tool is called

Similar candidates returned

Agent:

Reuses existing method

Generalizes shared logic

Or explicitly keeps separation

Optional:
If score > threshold, require justification.

Scheduled Refactoring

Run clustering periodically

Pass clusters to refactoring agent

Execute candidate transformations

Run test suite

Accept only fully validated changes

Agents can safely iterate faster than humans.

11. Guardrails

Prevent architecture degradation:

Respect module boundaries

Avoid merging across defined bounded contexts

Reject refactorings that increase coupling score

Limit maximum utility complexity

Similarity ? merge.

12. Metrics

Track:

Duplicate reduction rate

Suggestion acceptance rate

False positive rate

Refactoring success rate

Coupling change over time

Otherwise you’ll optimize blind.

13. Realistic MVP

Don’t overengineer first.

MVP:

Tree-sitter parsing

Basic normalization

Embedding per method

FAISS index

Top-3 similarity output

No clustering. No automatic refactoring.