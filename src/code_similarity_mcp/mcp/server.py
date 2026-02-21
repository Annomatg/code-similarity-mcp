"""MCP server exposing code similarity tools."""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from code_similarity_mcp.embeddings.generator import EmbeddingGenerator
from code_similarity_mcp.index.registry import MethodRegistry
from code_similarity_mcp.normalizer import normalize_code
from code_similarity_mcp.parser.registry import SUPPORTED_EXTENSIONS, get_parser
from code_similarity_mcp.similarity.filter import FilterPipeline
from code_similarity_mcp.similarity.scorer import SimilarityScorer

# ---------------------------------------------------------------------------
# Logging — file only (stdout would corrupt the stdio MCP transport)
# ---------------------------------------------------------------------------

_LOG_DIR = Path.home() / ".code-similarity-mcp"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[logging.FileHandler(_LOG_DIR / "server.log", encoding="utf-8")],
)
log = logging.getLogger("code-similarity-mcp")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastMCP("code-similarity-mcp")
_generator = EmbeddingGenerator()
_scorer = SimilarityScorer()
_filter = FilterPipeline()

_DEFAULT_INDEX_DIR = Path.home() / ".code-similarity-mcp" / "index"

_EXCLUDED_DIRS = {
    ".venv", "venv", ".env",
    "__pycache__", ".git", ".svn",
    "node_modules", "site-packages",
    "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
}

log.info("Server module loaded. Default index dir: %s", _DEFAULT_INDEX_DIR)


def _get_registry(index_dir: str | None = None) -> MethodRegistry:
    path = Path(index_dir) if index_dir else _DEFAULT_INDEX_DIR
    log.debug("Opening registry at %s", path)
    return MethodRegistry(path)


# ---------------------------------------------------------------------------
# Tool: index_repository
# ---------------------------------------------------------------------------

@app.tool()
def index_repository(
    repository_root: str,
    index_dir: str | None = None,
    force_reindex: bool = False,
) -> str:
    """
    Index all supported source files in a repository.

    Args:
        repository_root: Absolute path to the repository root.
        index_dir: Where to store the index (default: ~/.code-similarity-mcp/index).
        force_reindex: If true, re-index files even if already indexed.

    Returns:
        JSON summary with files processed and methods indexed.
    """
    log.info("index_repository called: root=%s index_dir=%s force=%s",
             repository_root, index_dir, force_reindex)

    root = Path(repository_root)
    if not root.is_dir():
        log.error("Not a directory: %s", repository_root)
        return json.dumps({"error": f"Not a directory: {repository_root}"})

    registry = _get_registry(index_dir)
    files_processed = 0
    methods_indexed = 0

    for ext, lang in SUPPORTED_EXTENSIONS.items():
        found = [
            p for p in root.rglob(f"*{ext}")
            if not any(part in _EXCLUDED_DIRS for part in p.parts)
        ]
        log.debug("Found %d %s files (after exclusions)", len(found), lang)
        for file_path in found:
            str_path = str(file_path)
            if not force_reindex and registry.get_by_file(str_path):
                log.debug("Skipping already-indexed file: %s", str_path)
                continue
            registry.delete_by_file(str_path)
            try:
                parser = get_parser(lang)
                methods = parser.parse_file(str_path)
            except Exception:
                log.warning("Failed to parse %s:\n%s", str_path, traceback.format_exc())
                continue

            log.debug("  %s: %d methods", file_path.name, len(methods))
            for method in methods:
                method.normalized_code = normalize_code(method.body_code, lang)
                embedding = _generator.encode_one(method.normalized_code)
                registry.add_method(method, embedding)
                methods_indexed += 1

            files_processed += 1

    registry.close()
    result = {
        "files_processed": files_processed,
        "methods_indexed": methods_indexed,
        "index_dir": str(_DEFAULT_INDEX_DIR if not index_dir else index_dir),
    }
    log.info("index_repository done: %s", result)
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool: analyze_new_code
# ---------------------------------------------------------------------------

@app.tool()
def analyze_new_code(
    code_snippet: str,
    language: str = "python",
    top_k: int = 3,
    index_dir: str | None = None,
) -> str:
    """
    Analyze a code snippet and return similar methods from the index.

    Args:
        code_snippet: Source code to analyze.
        language: Language of the snippet (default: python).
        top_k: Number of candidates to return per method (default: 3).
        index_dir: Index directory to query.

    Returns:
        JSON with new_methods list, each containing candidates with scores and hints.
    """
    log.info("analyze_new_code called: language=%s top_k=%d snippet_len=%d",
             language, top_k, len(code_snippet))

    registry = _get_registry(index_dir)

    try:
        parser = get_parser(language)
    except ValueError as e:
        log.error("Unsupported language: %s", language)
        return json.dumps({"error": str(e)})

    methods = parser.parse_snippet(code_snippet, language)
    log.debug("Parsed %d methods from snippet", len(methods))
    if not methods:
        return json.dumps({"new_methods": [], "note": "No methods found in snippet"})

    output = {"new_methods": []}

    for method in methods:
        method.normalized_code = normalize_code(method.body_code, language)
        embedding = _generator.encode_one(method.normalized_code)

        query = {
            "id": -1,
            "file_path": "<snippet>",
            "language": language,
            "name": method.name,
            "parameters": method.parameters,
            "return_type": method.return_type,
            "normalized_code": method.normalized_code,
            "code_hash": method.code_hash,
            "start_line": method.start_line,
            "end_line": method.end_line,
            "dependencies": method.dependencies,
            "ast_fingerprint": method.ast_fingerprint,
        }

        valid_ids = _filter.get_candidate_ids(registry, query)
        log.debug("Method '%s': %d candidate IDs from fast filter", method.name, len(valid_ids))
        raw_candidates = registry.search(embedding, top_k=top_k * 3, allowed_ids=valid_ids)
        scored = _scorer.score_candidates(query, raw_candidates)[:top_k]
        log.debug("Method '%s': %d raw candidates, %d scored", method.name, len(raw_candidates), len(scored))

        output["new_methods"].append({
            "name": method.name,
            "parameters": method.parameters,
            "candidates": [
                {
                    "file": c.file_path,
                    "method": c.name,
                    "line": c.start_line,
                    "score": c.score,
                    "exact_match": c.exact_match,
                    "embedding_similarity": c.embedding_score,
                    "ast_similarity": c.ast_score,
                    "differences": c.differences,
                    "refactoring_hints": c.refactoring_hints,
                }
                for c in scored
            ],
        })

    registry.close()
    log.info("analyze_new_code done: %d methods analyzed", len(output["new_methods"]))
    return json.dumps(output, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import asyncio
    # Pre-load the embedding model in the main thread before the async loop
    # starts. SentenceTransformer hangs when first loaded inside a thread
    # executor on Windows (PyTorch DLL / multiprocessing init issue).
    log.info("Pre-loading embedding model...")
    _ = _generator.model
    log.info("Model loaded. Starting MCP server (stdio)")
    asyncio.run(app.run_stdio_async())
