"""MCP server exposing code similarity tools."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from code_similarity_mcp.embeddings.generator import EmbeddingGenerator
from code_similarity_mcp.index.registry import MethodRegistry
from code_similarity_mcp.normalizer import normalize_code
from code_similarity_mcp.parser.registry import SUPPORTED_EXTENSIONS, get_parser
from code_similarity_mcp.similarity.scorer import SimilarityScorer

app = Server("code-similarity-mcp")
_generator = EmbeddingGenerator()
_scorer = SimilarityScorer()

# Default index location (can be overridden per call)
_DEFAULT_INDEX_DIR = Path.home() / ".code-similarity-mcp" / "index"


def _get_registry(index_dir: str | None = None) -> MethodRegistry:
    path = Path(index_dir) if index_dir else _DEFAULT_INDEX_DIR
    return MethodRegistry(path)


# ---------------------------------------------------------------------------
# Tool: index_repository
# ---------------------------------------------------------------------------

@app.tool()
async def index_repository(
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
    root = Path(repository_root)
    if not root.is_dir():
        return json.dumps({"error": f"Not a directory: {repository_root}"})

    registry = _get_registry(index_dir)
    files_processed = 0
    methods_indexed = 0

    for ext, lang in SUPPORTED_EXTENSIONS.items():
        for file_path in root.rglob(f"*{ext}"):
            str_path = str(file_path)
            if not force_reindex and registry.get_by_file(str_path):
                continue
            registry.delete_by_file(str_path)
            try:
                parser = get_parser(lang)
                methods = parser.parse_file(str_path)
            except Exception as exc:
                continue  # skip unparseable files silently

            for method in methods:
                method.normalized_code = normalize_code(method.body_code, lang)
                embedding = _generator.encode_one(method.normalized_code)
                registry.add_method(method, embedding)
                methods_indexed += 1

            files_processed += 1

    registry.close()
    return json.dumps({
        "files_processed": files_processed,
        "methods_indexed": methods_indexed,
        "index_dir": str(_DEFAULT_INDEX_DIR if not index_dir else index_dir),
    })


# ---------------------------------------------------------------------------
# Tool: analyze_new_code
# ---------------------------------------------------------------------------

@app.tool()
async def analyze_new_code(
    code_snippet: str,
    language: str = "gdscript",
    top_k: int = 3,
    index_dir: str | None = None,
) -> str:
    """
    Analyze a code snippet and return similar methods from the index.

    Args:
        code_snippet: Source code to analyze.
        language: Language of the snippet (default: gdscript).
        top_k: Number of candidates to return per method (default: 3).
        index_dir: Index directory to query.

    Returns:
        JSON with new_methods list, each containing candidates with scores and hints.
    """
    registry = _get_registry(index_dir)

    try:
        parser = get_parser(language)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    methods = parser.parse_snippet(code_snippet, language)
    if not methods:
        return json.dumps({"new_methods": [], "note": "No methods found in snippet"})

    output = {"new_methods": []}

    for method in methods:
        method.normalized_code = normalize_code(method.body_code, language)
        embedding = _generator.encode_one(method.normalized_code)

        # Build query dict compatible with scorer
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
        }

        raw_candidates = registry.search(embedding, top_k=top_k * 3)
        scored = _scorer.score_candidates(query, raw_candidates)[:top_k]

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
    return json.dumps(output, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    async def _run():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())

    asyncio.run(_run())
