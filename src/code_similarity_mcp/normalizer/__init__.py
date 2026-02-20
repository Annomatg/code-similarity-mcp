"""Code normalization to remove superficial variation."""

from .base import BaseNormalizer
from .registry import get_normalizer


def normalize_code(code: str, language: str = "python") -> str:
    """Normalize code for the given language."""
    return get_normalizer(language).normalize(code)


__all__ = ["BaseNormalizer", "get_normalizer", "normalize_code"]
