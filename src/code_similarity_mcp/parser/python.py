"""Python parser using tree-sitter."""

from __future__ import annotations

from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from .base import BaseParser, MethodInfo

_PY_LANGUAGE = Language(tspython.language())

# Parameter node types whose first identifier child is the name
_PARAM_TYPES = frozenset({
    "identifier",
    "typed_parameter",
    "default_parameter",
    "typed_default_parameter",
})

# Splat parameter types (*args / **kwargs) — name is second child
_SPLAT_TYPES = frozenset({
    "list_splat_pattern",
    "dictionary_splat_pattern",
})


def _node_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8")


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


def _extract_params(params_node, source: bytes) -> list[str]:
    names: list[str] = []
    for child in params_node.children:
        if child.type in _PARAM_TYPES:
            if child.type == "identifier":
                name = _node_text(child, source)
                if name != "self" and name != "cls":
                    names.append(name)
            else:
                # first child is the identifier
                for sub in child.children:
                    if sub.type == "identifier":
                        names.append(_node_text(sub, source))
                        break
        elif child.type in _SPLAT_TYPES:
            # second child is the identifier (*args → args)
            for sub in child.children:
                if sub.type == "identifier":
                    names.append(_node_text(sub, source))
                    break
    return names


def _extract_fingerprint(node) -> list[str]:
    """Return a DFS sequence of named node types for the given subtree.

    Named nodes carry semantic meaning (e.g. ``function_definition``,
    ``return_statement``) while anonymous nodes are punctuation/keywords
    that are less useful for structural comparison.
    """
    result: list[str] = []
    _collect_types(node, result)
    return result


def _collect_types(node, result: list[str]) -> None:
    if node.is_named:
        result.append(node.type)
    for child in node.children:
        _collect_types(child, result)


def _extract_dependencies(func_node, source: bytes, func_name: str) -> list[str]:
    calls: set[str] = set()
    for node in _walk(func_node):
        if node.type == "call":
            callee = node.children[0] if node.children else None
            if callee is None:
                continue
            if callee.type == "identifier":
                name = _node_text(callee, source)
                if name != func_name:
                    calls.add(name)
            elif callee.type == "attribute":
                # e.g. self.helper → take 'helper'
                for sub in callee.children:
                    if sub.type == "identifier":
                        name = _node_text(sub, source)
                calls.add(name)
    return sorted(calls)


class PythonParser(BaseParser):
    """Parses Python (.py) files using tree-sitter."""

    LANGUAGE = "python"
    EXTENSIONS = [".py"]

    def __init__(self) -> None:
        self._parser = Parser(_PY_LANGUAGE)

    def parse_file(self, file_path: str) -> list[MethodInfo]:
        source = Path(file_path).read_bytes()
        return self._extract_methods(source, file_path)

    def parse_snippet(self, code: str, language: str = "python") -> list[MethodInfo]:
        source = code.encode("utf-8")
        return self._extract_methods(source, "<snippet>")

    def _extract_methods(self, source: bytes, file_path: str) -> list[MethodInfo]:
        tree = self._parser.parse(source)
        methods: list[MethodInfo] = []

        for node in _walk(tree.root_node):
            if node.type == "function_definition":
                info = self._extract_function(node, source, file_path)
                if info is not None:
                    methods.append(info)

        return methods

    def _extract_function(self, node, source: bytes, file_path: str) -> MethodInfo | None:
        name: str | None = None
        params: list[str] = []
        return_type: str | None = None

        for child in node.children:
            if child.type == "identifier":
                name = _node_text(child, source)
            elif child.type == "parameters":
                params = _extract_params(child, source)
            elif child.type == "type":
                return_type = _node_text(child, source)

        if name is None:
            return None

        start_line = node.start_point[0] + 1   # 1-based
        end_line = node.end_point[0] + 1
        body_code = _node_text(node, source)
        dependencies = _extract_dependencies(node, source, name)
        ast_fingerprint = _extract_fingerprint(node)

        return MethodInfo(
            file_path=file_path,
            language=self.LANGUAGE,
            name=name,
            parameters=params,
            return_type=return_type,
            body_code=body_code,
            normalized_code="",  # filled by normalizer
            start_line=start_line,
            end_line=end_line,
            dependencies=dependencies,
            ast_fingerprint=ast_fingerprint,
        )
