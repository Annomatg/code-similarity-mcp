"""Parser registry - maps file extensions and language names to parser instances."""

from __future__ import annotations

from .base import BaseParser

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".py": "python",
}

_parsers: dict[str, BaseParser] = {}


def get_parser(language: str) -> BaseParser:
    """Return cached parser instance for the given language."""
    if language not in _parsers:
        _parsers[language] = _create_parser(language)
    return _parsers[language]


def _create_parser(language: str) -> BaseParser:
    if language == "python":
        from .python import PythonParser
        return PythonParser()
    raise ValueError(f"Unsupported language: {language!r}")
