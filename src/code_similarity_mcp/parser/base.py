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


class BaseParser(ABC):
    """Abstract parser for a specific language."""

    @abstractmethod
    def parse_file(self, file_path: str) -> list[MethodInfo]:
        """Parse a source file and return all extracted methods."""

    @abstractmethod
    def parse_snippet(self, code: str, language: str) -> list[MethodInfo]:
        """Parse an in-memory code snippet and return extracted methods."""
