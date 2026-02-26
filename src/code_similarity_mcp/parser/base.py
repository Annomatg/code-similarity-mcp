"""Base parser interface and shared data models."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import NamedTuple


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


def group_into_chunks(graph: "DependencyGraph") -> list[list[int]]:
    """Group statements into self-consistent chunks using a greedy heuristic.

    Processes statements in order.  A new chunk begins whenever:

    1. The next statement has **no** intra-function data-flow providers (all of
       its reads come from parameters or external variables), signalling a
       natural "fresh start" for an independent computation — provided the
       current chunk is already non-empty; or
    2. The next statement reads a variable written by a statement in an
       already-closed chunk (an *unresolved* cross-chunk dependency that cannot
       be fixed by further extension).

    A chunk is *self-consistent* when every variable read by statements inside
    the chunk is either written by an earlier statement in the same chunk or
    comes from outside the function (parameters / module-level names).

    Args:
        graph: A :class:`DependencyGraph` built by
            :func:`~code_similarity_mcp.parser.python.build_dependency_graph`.

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

        if has_closed_dep or (is_fresh_start and len(current_chunk) > 0):
            chunks.append(current_chunk)
            closed_statements.update(current_chunk)
            current_chunk = [i]
        else:
            current_chunk.append(i)

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


class BaseParser(ABC):
    """Abstract parser for a specific language."""

    @abstractmethod
    def parse_file(self, file_path: str) -> list[MethodInfo]:
        """Parse a source file and return all extracted methods."""

    @abstractmethod
    def parse_snippet(self, code: str, language: str) -> list[MethodInfo]:
        """Parse an in-memory code snippet and return extracted methods."""
