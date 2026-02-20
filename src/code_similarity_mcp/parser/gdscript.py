"""GDScript parser using tree-sitter."""

from __future__ import annotations

import re
from pathlib import Path

from .base import BaseParser, MethodInfo


def _try_tree_sitter_languages(language_name: str):
    """Try to load a language from tree_sitter_languages."""
    try:
        import tree_sitter_languages
        return tree_sitter_languages.get_language(language_name)
    except Exception:
        return None


def _get_gdscript_language():
    """Get the GDScript tree-sitter language, with fallback."""
    lang = _try_tree_sitter_languages("gdscript")
    if lang is not None:
        return lang
    raise RuntimeError(
        "GDScript tree-sitter grammar not available. "
        "Install tree-sitter-languages with GDScript support."
    )


class GDScriptParser(BaseParser):
    """Parses GDScript (.gd) files using tree-sitter."""

    LANGUAGE = "gdscript"
    EXTENSIONS = [".gd"]

    def __init__(self) -> None:
        from tree_sitter import Parser
        self._lang = _get_gdscript_language()
        self._parser = Parser()
        self._parser.set_language(self._lang)

    def parse_file(self, file_path: str) -> list[MethodInfo]:
        source = Path(file_path).read_text(encoding="utf-8")
        return self._extract_methods(source, file_path)

    def parse_snippet(self, code: str, language: str = "gdscript") -> list[MethodInfo]:
        return self._extract_methods(code, "<snippet>")

    def _extract_methods(self, source: str, file_path: str) -> list[MethodInfo]:
        tree = self._parser.parse(source.encode())
        lines = source.splitlines()
        methods: list[MethodInfo] = []

        for node in self._walk(tree.root_node):
            if node.type == "function_definition":
                info = self._extract_function(node, source, lines, file_path)
                if info:
                    methods.append(info)
        return methods

    def _walk(self, node):
        yield node
        for child in node.children:
            yield from self._walk(child)

    def _extract_function(self, node, source: str, lines: list[str], file_path: str) -> MethodInfo | None:
        name = None
        params: list[str] = []
        return_type: str | None = None

        for child in node.children:
            if child.type == "name":
                name = source[child.start_byte:child.end_byte]
            elif child.type == "parameters":
                params = self._extract_params(child, source)
            elif child.type == "return_type":
                # gdscript: -> TypeName
                rt_text = source[child.start_byte:child.end_byte]
                return_type = rt_text.lstrip("->").strip()

        if name is None:
            return None

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        body_code = source[node.start_byte:node.end_byte]
        dependencies = self._extract_dependencies(node, source, name)

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
        )

    def _extract_params(self, params_node, source: str) -> list[str]:
        params = []
        for child in params_node.children:
            if child.type in ("parameter", "typed_parameter", "default_parameter"):
                # grab just the name part
                for sub in child.children:
                    if sub.type == "identifier":
                        params.append(source[sub.start_byte:sub.end_byte])
                        break
            elif child.type == "identifier":
                params.append(source[child.start_byte:child.end_byte])
        return params

    def _extract_dependencies(self, func_node, source: str, func_name: str) -> list[str]:
        """Extract function calls made inside the body."""
        calls: set[str] = set()
        for node in self._walk(func_node):
            if node.type == "call":
                for child in node.children:
                    if child.type in ("identifier", "member_expression", "attribute"):
                        call_text = source[child.start_byte:child.end_byte]
                        if call_text != func_name:
                            calls.add(call_text)
                        break
        return sorted(calls)
