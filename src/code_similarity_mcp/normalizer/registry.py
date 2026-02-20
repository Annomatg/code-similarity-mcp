"""Normalizer registry - maps language names to normalizer instances."""

from __future__ import annotations

from .base import BaseNormalizer

_normalizers: dict[str, BaseNormalizer] = {}


def get_normalizer(language: str) -> BaseNormalizer:
    """Return cached normalizer instance for the given language."""
    if language not in _normalizers:
        _normalizers[language] = _create_normalizer(language)
    return _normalizers[language]


def _create_normalizer(language: str) -> BaseNormalizer:
    if language == "python":
        from .python_normalizer import PythonNormalizer
        return PythonNormalizer()
    raise ValueError(f"Unsupported language: {language!r}")
