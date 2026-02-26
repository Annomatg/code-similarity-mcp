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
from code_similarity_mcp.parser.base import annotate_chunks, embed_chunks, group_into_chunks
from code_similarity_mcp.parser.python import (
    build_dependency_graph,
    count_statements,
    get_flat_statements,
)
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
                if method.is_stub:
                    log.debug("  Skipping stub: %s", method.name)
                    continue
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
# Tool: analyze_project
# ---------------------------------------------------------------------------

@app.tool()
def analyze_project(
    index_dir: str | None = None,
    threshold: float = 0.85,
    top_k: int = 5,
    min_lines: int = 4,
) -> str:
    """
    Compare all indexed methods against each other and report similar pairs.

    Useful for finding duplicate or highly similar functions within a project
    that may be candidates for refactoring or consolidation.

    Args:
        index_dir: Index directory to query (default: ~/.code-similarity-mcp/index).
        threshold: Minimum similarity score to include a pair (0.0–1.0, default: 0.85).
        top_k: Maximum number of similar matches to find per method (default: 5).
        min_lines: Minimum number of lines a method must have to participate in
            comparisons (default: 4). Methods shorter than this threshold are
            skipped as both queries and candidates — they remain in the index for
            use with analyze_new_code. This prevents trivial getters/setters from
            generating low-value similarity pairs.

    Returns:
        JSON with total_methods count and similar_pairs list. Each pair contains
        method_a, method_b (with file/method/line), score, exact_match,
        embedding_similarity, ast_similarity, differences, and refactoring_hints.
    """
    log.info("analyze_project called: index_dir=%s threshold=%f top_k=%d min_lines=%d",
             index_dir, threshold, top_k, min_lines)

    registry = _get_registry(index_dir)
    all_methods = registry.get_all_methods()

    if not all_methods:
        registry.close()
        log.info("analyze_project: empty index, returning 0 pairs")
        return json.dumps({"total_methods": 0, "similar_pairs": []})

    # Filter out methods below the line threshold — they remain in the index
    # but are excluded from both query and candidate roles in this run.
    eligible_methods = [
        m for m in all_methods
        if m["end_line"] - m["start_line"] + 1 >= min_lines
    ]
    eligible_ids = {m["id"] for m in eligible_methods}

    log.debug("analyze_project: %d total methods, %d eligible (min_lines=%d)",
              len(all_methods), len(eligible_methods), min_lines)

    seen_pairs: set[tuple[int, int]] = set()
    similar_pairs: list[dict] = []

    for method in eligible_methods:
        faiss_pos = method.get("faiss_pos")
        embedding = registry.get_embedding(faiss_pos)
        if embedding is None:
            continue

        # Get candidate IDs via the fast filter; exclude self and short methods
        valid_ids = _filter.get_candidate_ids(registry, method)
        valid_ids.discard(method["id"])
        valid_ids &= eligible_ids
        if not valid_ids:
            continue

        raw_candidates = registry.search(embedding, top_k=top_k * 3, allowed_ids=valid_ids)
        scored = _scorer.score_candidates(method, raw_candidates)[:top_k]

        for result in scored:
            if result.score < threshold:
                continue
            pair_key = (min(method["id"], result.db_id), max(method["id"], result.db_id))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            similar_pairs.append({
                "method_a": {
                    "file": method["file_path"],
                    "method": method["name"],
                    "line": method["start_line"],
                },
                "method_b": {
                    "file": result.file_path,
                    "method": result.name,
                    "line": result.start_line,
                },
                "score": result.score,
                "exact_match": result.exact_match,
                "embedding_similarity": result.embedding_score,
                "ast_similarity": result.ast_score,
                "differences": result.differences,
                "refactoring_hints": result.refactoring_hints,
            })

    similar_pairs.sort(key=lambda p: p["score"], reverse=True)

    registry.close()
    log.info("analyze_project done: %d methods (%d eligible), %d similar pairs found",
             len(all_methods), len(eligible_methods), len(similar_pairs))
    return json.dumps({"total_methods": len(all_methods), "similar_pairs": similar_pairs}, indent=2)


# ---------------------------------------------------------------------------
# Tool: find_large_functions
# ---------------------------------------------------------------------------

@app.tool()
def find_large_functions(
    index_dir: str | None = None,
    min_statements: int = 30,
) -> str:
    """
    Scan an indexed repository and return all functions with more than
    min_statements statements.  These are candidates for dependency-aware
    chunking or refactoring.

    Args:
        index_dir: Index directory to query (default: ~/.code-similarity-mcp/index).
        min_statements: Minimum statement count threshold, exclusive (default: 30).

    Returns:
        JSON with large_functions list, each entry containing id, name, file,
        start_line, end_line, and statement_count.
    """
    log.info("find_large_functions called: index_dir=%s min_statements=%d",
             index_dir, min_statements)

    registry = _get_registry(index_dir)
    all_methods = registry.get_all_methods()
    registry.close()

    large: list[dict] = []
    for method in all_methods:
        stmt_count = count_statements(method["body_code"])
        if stmt_count > min_statements:
            large.append({
                "id": method["id"],
                "name": method["name"],
                "file": method["file_path"],
                "start_line": method["start_line"],
                "end_line": method["end_line"],
                "statement_count": stmt_count,
            })

    large.sort(key=lambda e: e["statement_count"], reverse=True)

    log.info("find_large_functions done: %d/%d methods exceed %d statements",
             len(large), len(all_methods), min_statements)
    return json.dumps({"large_functions": large}, indent=2)


# ---------------------------------------------------------------------------
# Tool: chunk_repository
# ---------------------------------------------------------------------------

_CHUNK_MIN_STATEMENTS = 30  # functions with more than this many statements are chunked


@app.tool()
def chunk_repository(
    repository_root: str,
    index_dir: str | None = None,
    max_statements_per_chunk: int = 10,
    force_rechunk: bool = False,
) -> str:
    """
    Run the full chunking pipeline on large functions in a repository.

    Scans all indexed functions from files under repository_root, identifies
    those with more than 30 statements, and runs the dependency-graph chunking
    pipeline on each.  Chunk metadata and embeddings are stored in the index
    for later retrieval and similarity search.

    Re-running on an already-chunked repository is safe: functions whose
    chunks are already stored are skipped so no duplicate records are created.
    Set force_rechunk=True to replace existing chunks unconditionally.

    Args:
        repository_root: Root directory of the repository used to filter
            which indexed functions to process.
        index_dir: Index directory (default: ~/.code-similarity-mcp/index).
        max_statements_per_chunk: Hard cap on the number of statements per
            chunk (default: 10).
        force_rechunk: If true, re-chunk functions even if chunks are already
            stored (replaces existing chunks). Default: false.

    Returns:
        JSON summary: { files_scanned, functions_chunked, chunks_created }
    """
    log.info(
        "chunk_repository called: root=%s index_dir=%s max_stmts=%d",
        repository_root,
        index_dir,
        max_statements_per_chunk,
    )

    root = Path(repository_root)
    if not root.is_dir():
        log.error("Not a directory: %s", repository_root)
        return json.dumps({"error": f"Not a directory: {repository_root}"})

    registry = _get_registry(index_dir)
    all_methods = registry.get_all_methods()

    # Filter to methods from files within repository_root.
    repo_methods = [
        m for m in all_methods
        if Path(m["file_path"]).is_relative_to(root)
    ]

    files_scanned = len({m["file_path"] for m in repo_methods})
    functions_chunked = 0
    chunks_created = 0

    for method in repo_methods:
        if count_statements(method["body_code"]) <= _CHUNK_MIN_STATEMENTS:
            continue

        # Skip functions that are already chunked unless a re-chunk is forced.
        # A function whose code hasn't changed retains its DB id, so existing
        # chunks (if any) are always current — no need to recompute them.
        if not force_rechunk and registry.get_chunks_by_function(method["id"]):
            log.debug(
                "Skipping already-chunked function %s (%s)",
                method["name"],
                method["file_path"],
            )
            continue

        try:
            graph = build_dependency_graph(method["body_code"])
            if graph.num_statements == 0:
                continue

            raw_chunks = group_into_chunks(graph, max_statements_per_chunk)
            if not raw_chunks:
                continue

            statements = get_flat_statements(method["body_code"])
            annotated = annotate_chunks(
                raw_chunks,
                graph,
                function_name=method["name"],
                file_path=method["file_path"],
                function_id=method["id"],
            )
            embeddings, norm_texts = embed_chunks(
                annotated, method["body_code"], statements, _generator,
                return_texts=True,
            )

            # Replace any previously stored chunks for this function.
            registry.delete_chunks_by_function(method["id"])
            for chunk_info, emb, norm_code in zip(annotated, embeddings, norm_texts):
                registry.add_chunk(chunk_info, emb, normalized_code=norm_code)
                chunks_created += 1

            functions_chunked += 1
            log.debug(
                "Chunked %s (%s): %d chunks",
                method["name"],
                method["file_path"],
                len(annotated),
            )
        except Exception:
            log.warning(
                "Failed to chunk %s in %s:\n%s",
                method["name"],
                method["file_path"],
                traceback.format_exc(),
            )

    registry.close()
    result = {
        "files_scanned": files_scanned,
        "functions_chunked": functions_chunked,
        "chunks_created": chunks_created,
    }
    log.info("chunk_repository done: %s", result)
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool: analyze_chunks
# ---------------------------------------------------------------------------

def _refactoring_hint(score: float) -> str:
    if score >= 0.99:
        return "Exact duplicate chunk. Consider extracting to a shared helper function."
    if score >= 0.85:
        return "Very similar chunk. Consider consolidating this logic."
    if score >= 0.70:
        return "Related chunk. Review for potential refactoring opportunities."
    return "Loosely similar chunk. Low refactoring priority."


@app.tool()
def analyze_chunks(
    code_snippet: str | None = None,
    chunk_id: int | None = None,
    top_k: int = 3,
    index_dir: str | None = None,
) -> str:
    """
    Search for similar stored chunks.

    Given a raw code snippet or the id of a stored chunk, return the top-k
    most similar chunks with scores and refactoring hints.

    Args:
        code_snippet: Code fragment to search for (mutually exclusive with chunk_id).
        chunk_id: DB id of a stored chunk to use as the query (mutually exclusive
            with code_snippet).  The query chunk itself is excluded from results.
        top_k: Number of results to return (default: 3).
        index_dir: Index directory to query.

    Returns:
        JSON with a 'results' list, each item containing:
        chunk_id, function_name, file, similarity_score, statement_range,
        refactoring_hint.
    """
    log.info(
        "analyze_chunks called: chunk_id=%s snippet_len=%s top_k=%d",
        chunk_id,
        len(code_snippet) if code_snippet else None,
        top_k,
    )

    if code_snippet is None and chunk_id is None:
        return json.dumps({"error": "Provide either code_snippet or chunk_id"})
    if code_snippet is not None and chunk_id is not None:
        return json.dumps({"error": "Provide either code_snippet or chunk_id, not both"})

    registry = _get_registry(index_dir)

    if registry.get_chunk_count() == 0:
        registry.close()
        log.info("analyze_chunks: chunk index is empty")
        return json.dumps({"results": []})

    exclude_id: int | None = None

    if chunk_id is not None:
        chunk = registry.get_chunk_by_id(chunk_id)
        if chunk is None:
            registry.close()
            return json.dumps({"error": f"Chunk id {chunk_id} not found"})
        embedding = registry.get_chunk_embedding(chunk["faiss_pos"])
        if embedding is None:
            registry.close()
            return json.dumps({"error": f"Embedding not found for chunk id {chunk_id}"})
        exclude_id = chunk_id
    else:
        # Wrap snippet in a dummy function so the normalizer can process it.
        indented = "\n".join(f"    {line}" for line in code_snippet.splitlines())
        wrapped = f"def _chunk_func():\n{indented}\n"
        normalized = normalize_code(wrapped, "python")
        embedding = _generator.encode_one(normalized)

    # Fetch enough candidates to account for self-exclusion.
    fetch_k = top_k + 1 if exclude_id is not None else top_k
    raw = registry.search_chunks(embedding, top_k=fetch_k)

    output: list[dict] = []
    for r in raw:
        if r["id"] == exclude_id:
            continue
        if len(output) >= top_k:
            break
        score = r["embedding_score"]
        output.append({
            "chunk_id": r["id"],
            "function_name": r["function_name"],
            "file": r["file_path"],
            "similarity_score": round(score, 4),
            "statement_range": [r["statement_start"], r["statement_end"]],
            "refactoring_hint": _refactoring_hint(score),
        })

    registry.close()
    log.info("analyze_chunks done: %d results returned", len(output))
    return json.dumps({"results": output}, indent=2)


# ---------------------------------------------------------------------------
# Tool: get_chunk_map
# ---------------------------------------------------------------------------


def _is_dag(n: int, deps: list[list[int]]) -> bool:
    """Return True if the directed dependency graph has no cycles.

    *deps[i]* is the list of chunk indices that chunk *i* depends on
    (i.e. directed edges go i → deps[i][j]).
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n

    def dfs(v: int) -> bool:
        color[v] = GRAY
        for w in deps[v]:
            if w < 0 or w >= n:
                continue
            if color[w] == GRAY:
                return False  # back-edge → cycle
            if color[w] == WHITE and not dfs(w):
                return False
        color[v] = BLACK
        return True

    return all(dfs(v) for v in range(n) if color[v] == WHITE)


def _normalized_code_for_chunk(
    chunk: dict,
    body_code: str,
    language: str = "python",
) -> str:
    """Reconstruct the normalized code for a stored chunk.

    Replicates the logic from :func:`embed_chunks`: extracts the source lines
    covered by the chunk's statement indices, wraps them in a dummy function,
    and normalizes with the language normalizer.
    """
    from code_similarity_mcp.normalizer.registry import get_normalizer

    statements = get_flat_statements(body_code)
    stmt_infos = [statements[idx] for idx in chunk["statement_indices"]]
    start_line = min(s.start_line for s in stmt_infos)
    end_line = max(s.end_line for s in stmt_infos)

    source_lines = body_code.splitlines()
    chunk_source = "\n".join(source_lines[start_line - 1 : end_line])
    indented = "\n".join("    " + line for line in chunk_source.splitlines())
    wrapped = "def _chunk_func():\n" + indented

    return get_normalizer(language).normalize(wrapped)


def _build_function_map(method: dict, chunks: list[dict]) -> dict:
    """Build a single-function chunk map dict from registry data."""
    sorted_chunks = sorted(chunks, key=lambda c: c["chunk_index"])
    language = method.get("language", "python")

    chunk_entries = []
    for c in sorted_chunks:
        # Use the stored normalized_code when available (populated since feature
        # #29); fall back to on-the-fly recomputation for older index records.
        norm_code = c.get("normalized_code") or ""
        if not norm_code:
            try:
                norm_code = _normalized_code_for_chunk(c, method["body_code"], language)
            except Exception:
                norm_code = ""

        chunk_entries.append({
            "chunk_id": c["id"],
            "chunk_index": c["chunk_index"],
            "statement_range": [c["statement_start"], c["statement_end"]],
            "dependencies": c["depends_on_chunks"],
            "normalized_code": norm_code,
        })

    n = len(sorted_chunks)
    deps = [c["depends_on_chunks"] for c in sorted_chunks]
    dag_valid = _is_dag(n, deps)

    return {
        "function_id": method["id"],
        "function_name": method["name"],
        "file": method["file_path"],
        "dag_valid": dag_valid,
        "chunks": chunk_entries,
    }


@app.tool()
def get_chunk_map(
    function_id: int | None = None,
    file_path: str | None = None,
    index_dir: str | None = None,
) -> str:
    """
    Return the complete chunk-level map for a given function or file.

    Returns all chunks, their metadata (statement range, inter-chunk
    dependencies, normalized code), and a DAG validity flag for each function.

    Args:
        function_id: Database id of a specific function (mutually exclusive
            with file_path).
        file_path: Absolute path to a source file — returns maps for all
            chunked functions in that file (mutually exclusive with function_id).
        index_dir: Index directory to query.

    Returns:
        JSON with a 'functions' list.  Each entry contains function_id,
        function_name, file, dag_valid, and chunks (list of chunk objects
        with chunk_id, chunk_index, statement_range, dependencies,
        normalized_code).
    """
    log.info(
        "get_chunk_map called: function_id=%s file_path=%s",
        function_id,
        file_path,
    )

    if function_id is None and file_path is None:
        return json.dumps({"error": "Provide either function_id or file_path"})
    if function_id is not None and file_path is not None:
        return json.dumps({"error": "Provide either function_id or file_path, not both"})

    registry = _get_registry(index_dir)

    if function_id is not None:
        chunks = registry.get_chunks_by_function(function_id)
        if not chunks:
            registry.close()
            log.info("get_chunk_map: no chunks for function_id=%d", function_id)
            return json.dumps({"functions": []})

        method = registry._get_method_by_id(function_id)
        if method is None:
            registry.close()
            return json.dumps({"error": f"Function id {function_id} not found"})

        result = {"functions": [_build_function_map(method, chunks)]}
    else:
        chunks = registry.get_chunks_by_file(file_path)
        if not chunks:
            registry.close()
            log.info("get_chunk_map: no chunks for file_path=%s", file_path)
            return json.dumps({"functions": []})

        # Group chunks by function_id and resolve method metadata.
        from collections import defaultdict
        by_function: dict[int, list[dict]] = defaultdict(list)
        for c in chunks:
            by_function[c["function_id"]].append(c)

        functions = []
        for fid, fchunks in by_function.items():
            method = registry._get_method_by_id(fid)
            if method is None:
                continue
            functions.append(_build_function_map(method, fchunks))

        # Sort by function name for deterministic output.
        functions.sort(key=lambda f: f["function_name"])
        result = {"functions": functions}

    registry.close()
    log.info(
        "get_chunk_map done: %d functions returned",
        len(result["functions"]),
    )
    return json.dumps(result, indent=2)


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
