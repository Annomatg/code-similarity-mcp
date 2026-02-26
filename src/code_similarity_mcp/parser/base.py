"""Base parser interface and shared data models."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import NamedTuple, Optional


@dataclass
class MethodInfo:
    """Metadata extracted from a parsed method/function."""

    file_path: str
    language: str
    name: str
    parameters: list[str]
    return_type: str | None
    body_code: str          # raw source
    normalized_code: str    # populated by normalizer
    start_line: int
    end_line: int
    dependencies: list[str] = field(default_factory=list)
    ast_fingerprint: list[str] = field(default_factory=list)  # DFS node-type sequence
    is_stub: bool = False  # True for abstract/pass-only/ellipsis-only/docstring-only methods

    @property
    def loc(self) -> int:
        return self.end_line - self.start_line + 1

    @property
    def code_hash(self) -> str:
        return hashlib.sha256(self.normalized_code.encode()).hexdigest()


class StatementInfo(NamedTuple):
    """A single top-level statement extracted from a function body."""

    index: int          # 0-based position in the function body
    node_type: str      # tree-sitter node type (e.g. "return_statement")
    start_line: int     # 1-based start line
    end_line: int       # 1-based end line
    source_text: str    # raw source text of the statement


@dataclass
class DependencyGraph:
    """Dependency graph for a function's statements.

    Nodes are zero-based flat indices spanning both top-level and nested
    statements (compound statement bodies are expanded inline).

    Attributes:
        data: Data-flow edges.  ``data[i]`` is the sorted list of statement
            indices that read a variable written by statement ``i``.
        control_flow: Control-flow edges.  ``control_flow[i]`` is the sorted
            list of statement indices controlled by statement ``i``: direct
            body children of a compound header, plus the fall-through target
            (the next statement at the same nesting level).
        num_statements: Total number of statements in the flat list.
    """

    data: dict
    control_flow: dict
    num_statements: int


def group_into_chunks(
    graph: "DependencyGraph",
    max_statements_per_chunk: int = 10,
) -> list[list[int]]:
    """Group statements into self-consistent chunks using a greedy heuristic.

    Processes statements in order.  A new chunk begins whenever:

    1. The next statement has **no** intra-function data-flow providers (all of
       its reads come from parameters or external variables), signalling a
       natural "fresh start" for an independent computation — provided the
       current chunk is already non-empty; or
    2. The next statement reads a variable written by a statement in an
       already-closed chunk (an *unresolved* cross-chunk dependency that cannot
       be fixed by further extension); or
    3. The current chunk has reached *max_statements_per_chunk* statements
       (hard size cap — the chunk is closed regardless of dependency state).

    A chunk is *self-consistent* when every variable read by statements inside
    the chunk is either written by an earlier statement in the same chunk or
    comes from outside the function (parameters / module-level names).

    Args:
        graph: A :class:`DependencyGraph` built by
            :func:`~code_similarity_mcp.parser.python.build_dependency_graph`.
        max_statements_per_chunk: Maximum number of statements allowed in a
            single chunk.  When the current chunk reaches this size the chunk
            is closed and a new one is started, regardless of dependency state.
            Must be a positive integer.  Defaults to ``10``.

    Returns:
        A list of chunks.  Each chunk is a non-empty list of consecutive
        statement indices in ascending order.  Together the chunks form a
        partition of ``range(graph.num_statements)``.  Returns an empty list
        when ``graph.num_statements == 0``.
    """
    if graph.num_statements == 0:
        return []

    # Build reverse index: providers[i] = {j : i in graph.data[j]}
    # providers[i] is the set of statements whose writes are read by statement i.
    providers: dict[int, set[int]] = {i: set() for i in range(graph.num_statements)}
    for j, consumers in graph.data.items():
        for i in consumers:
            providers[i].add(j)

    chunks: list[list[int]] = []
    current_chunk: list[int] = []
    closed_statements: set[int] = set()

    for i in range(graph.num_statements):
        # Rule 1: i depends on something in an already-closed chunk.
        has_closed_dep = any(j in closed_statements for j in providers[i])

        # Rule 2: i has no intra-function providers at all — a "fresh start"
        # that begins an independent computation.  Only split if the current
        # chunk already contains something (the very first statement always
        # goes into the first chunk regardless).
        is_fresh_start = len(providers[i]) == 0

        # Rule 3: current chunk has reached the size cap.
        max_size_reached = len(current_chunk) >= max_statements_per_chunk

        if has_closed_dep or (is_fresh_start and len(current_chunk) > 0) or max_size_reached:
            chunks.append(current_chunk)
            closed_statements.update(current_chunk)
            current_chunk = [i]
        else:
            current_chunk.append(i)

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


@dataclass
class ChunkInfo:
    """Metadata for a single statement chunk within a function.

    Attributes:
        chunk_index: 0-based position of this chunk in the function's chunk list.
        statement_start: Lowest statement index contained in this chunk.
        statement_end: Highest statement index contained in this chunk.
        statement_indices: Ordered list of all statement indices in this chunk.
        function_name: Name of the parent function/method (empty string if unknown).
        file_path: Source file path of the parent function (empty string if unknown).
        function_id: Optional registry/database id of the parent function.
        depends_on_chunks: Sorted list of chunk indices this chunk reads from
            (i.e. chunk M appears here when a variable written in chunk M is
            read by a statement in this chunk).
        depended_on_by_chunks: Sorted list of chunk indices that read from this
            chunk (inverse of ``depends_on_chunks``).
    """

    chunk_index: int
    statement_start: int
    statement_end: int
    statement_indices: list
    function_name: str
    file_path: str
    function_id: Optional[int]
    depends_on_chunks: list
    depended_on_by_chunks: list


def annotate_chunks(
    chunks: list,
    graph: "DependencyGraph",
    function_name: str = "",
    file_path: str = "",
    function_id: Optional[int] = None,
) -> list:
    """Annotate a list of statement-index chunks with rich metadata.

    Takes the raw chunk partition produced by :func:`group_into_chunks` and
    returns a :class:`ChunkInfo` for every chunk, enriched with:

    * The statement-index range (``statement_start`` / ``statement_end``).
    * Parent function metadata (``function_name``, ``file_path``,
      ``function_id``).
    * Cross-chunk dependency links derived from *data-flow* edges in *graph*:
      ``depends_on_chunks`` lists every other chunk that writes a variable read
      by this chunk; ``depended_on_by_chunks`` is the symmetric inverse.

    Cross-chunk control-flow edges are **not** included — only data-flow edges
    from ``graph.data`` are used when computing dependency links.

    Args:
        chunks: A list of chunks as returned by :func:`group_into_chunks`.
            Each chunk is a non-empty list of consecutive statement indices.
        graph: The :class:`DependencyGraph` for the same function.
        function_name: Human-readable name of the parent function/method.
        file_path: Source file from which the function was extracted.
        function_id: Optional registry id (e.g. SQLite row id) of the function.

    Returns:
        A :class:`list` of :class:`ChunkInfo` objects in the same order as
        *chunks*.  Returns an empty list when *chunks* is empty.

    Raises:
        ValueError: If the chunks do not form a valid partition of
            ``range(graph.num_statements)`` (gaps or overlaps detected).
    """
    if not chunks:
        return []

    # Validate: chunks must cover range(num_statements) exactly once.
    all_indices = [idx for chunk in chunks for idx in chunk]
    if sorted(all_indices) != list(range(graph.num_statements)):
        raise ValueError(
            f"chunks do not partition range({graph.num_statements}): "
            f"got indices {sorted(all_indices)}"
        )

    # Build statement → chunk_index lookup.
    stmt_to_chunk: dict = {}
    for chunk_idx, chunk in enumerate(chunks):
        for stmt_idx in chunk:
            stmt_to_chunk[stmt_idx] = chunk_idx

    # Build reverse data-flow index: providers[i] = {j : i in graph.data[j]}
    providers: dict = {i: set() for i in range(graph.num_statements)}
    for j, consumers in graph.data.items():
        for i in consumers:
            providers[i].add(j)

    # Compute cross-chunk dependency sets.
    n = len(chunks)
    depends_on: list = [set() for _ in range(n)]      # depends_on[N] = {M, ...}
    depended_on_by: list = [set() for _ in range(n)]  # depended_on_by[M] = {N, ...}

    for chunk_idx, chunk in enumerate(chunks):
        for stmt_idx in chunk:
            for provider_stmt in providers[stmt_idx]:
                provider_chunk = stmt_to_chunk[provider_stmt]
                if provider_chunk != chunk_idx:
                    depends_on[chunk_idx].add(provider_chunk)
                    depended_on_by[provider_chunk].add(chunk_idx)

    # Build ChunkInfo list.
    result: list = []
    for chunk_idx, chunk in enumerate(chunks):
        result.append(ChunkInfo(
            chunk_index=chunk_idx,
            statement_start=min(chunk),
            statement_end=max(chunk),
            statement_indices=list(chunk),
            function_name=function_name,
            file_path=file_path,
            function_id=function_id,
            depends_on_chunks=sorted(depends_on[chunk_idx]),
            depended_on_by_chunks=sorted(depended_on_by[chunk_idx]),
        ))

    return result


def embed_chunks(
    chunks: list,
    function_source: str,
    statements: list,
    generator,
    language: str = "python",
    return_texts: bool = False,
) -> list:
    """Generate a normalized embedding for each chunk.

    For each :class:`ChunkInfo` in *chunks*:

    1. Extracts the source lines covered by the chunk using the ``start_line``
       and ``end_line`` fields of the corresponding :class:`StatementInfo`
       entries in *statements*.
    2. Wraps the extracted lines in a minimal dummy function so the language
       normalizer can rename local identifiers consistently.
    3. Normalizes the wrapped source with the registered normalizer for
       *language*.
    4. Encodes the normalized text using *generator*.

    Args:
        chunks: Annotated chunks from :func:`annotate_chunks`.
        function_source: Full source text of the function from which the chunks
            were derived (the same string passed to ``build_dependency_graph``).
        statements: Flat statement list for the same function, as returned by
            :func:`~code_similarity_mcp.parser.python.get_flat_statements`.
            Must be indexed consistently with ``chunk.statement_indices``.
        generator: An :class:`~code_similarity_mcp.embeddings.generator.EmbeddingGenerator`
            instance used to produce the 384-dimensional embeddings.
        language: Source language used to select the normalizer (default
            ``"python"``).
        return_texts: When ``True``, return a ``(embeddings, normalized_texts)``
            tuple instead of just the embeddings list.  The ``normalized_texts``
            list contains the normalized string that was encoded for each chunk,
            in the same order as *chunks*.  Defaults to ``False`` for backward
            compatibility.

    Returns:
        When *return_texts* is ``False`` (default): a :class:`list` of
        ``numpy.ndarray`` embeddings, one per chunk, in the same order as
        *chunks*.  Each array has shape ``(384,)`` and dtype ``float32``.
        Returns an empty list when *chunks* is empty.

        When *return_texts* is ``True``: a ``(embeddings, normalized_texts)``
        tuple where ``embeddings`` is as described above and ``normalized_texts``
        is a :class:`list` of :class:`str` (one per chunk).
    """
    if not chunks:
        return ([], []) if return_texts else []

    from code_similarity_mcp.normalizer.registry import get_normalizer

    normalizer = get_normalizer(language)
    source_lines = function_source.splitlines()
    texts: list = []

    for chunk in chunks:
        # Compute the 1-based line range spanned by this chunk's statements.
        stmt_infos = [statements[idx] for idx in chunk.statement_indices]
        start_line = min(s.start_line for s in stmt_infos)
        end_line = max(s.end_line for s in stmt_infos)

        # Slice the function source (1-based → 0-based indexing).
        chunk_source = "\n".join(source_lines[start_line - 1 : end_line])

        # Wrap in a dummy function so the normalizer can rename local variables.
        indented = "\n".join("    " + line for line in chunk_source.splitlines())
        wrapped = "def _chunk_func():\n" + indented

        texts.append(normalizer.normalize(wrapped))

    import numpy as np  # noqa: F401 — ensure numpy is available for callers
    embeddings = list(generator.encode(texts))
    if return_texts:
        return embeddings, texts
    return embeddings


class BaseParser(ABC):
    """Abstract parser for a specific language."""

    @abstractmethod
    def parse_file(self, file_path: str) -> list[MethodInfo]:
        """Parse a source file and return all extracted methods."""

    @abstractmethod
    def parse_snippet(self, code: str, language: str) -> list[MethodInfo]:
        """Parse an in-memory code snippet and return extracted methods."""
