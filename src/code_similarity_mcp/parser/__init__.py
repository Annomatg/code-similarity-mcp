"""Parsers for extracting methods from source files."""

from .base import (
    BaseParser,
    ChunkInfo,
    DependencyGraph,
    MethodInfo,
    StatementInfo,
    annotate_chunks,
    embed_chunks,
    group_into_chunks,
)
from .python import build_dependency_graph, get_flat_statements, get_top_level_statements
from .registry import get_parser, SUPPORTED_EXTENSIONS

__all__ = [
    "BaseParser",
    "ChunkInfo",
    "DependencyGraph",
    "MethodInfo",
    "StatementInfo",
    "get_parser",
    "get_flat_statements",
    "get_top_level_statements",
    "annotate_chunks",
    "build_dependency_graph",
    "embed_chunks",
    "group_into_chunks",
    "SUPPORTED_EXTENSIONS",
]
