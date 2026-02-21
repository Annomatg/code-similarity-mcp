"""Base normalizer interface."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod


class BaseNormalizer(ABC):
    """Abstract normalizer for a specific language."""

    def normalize(self, code: str) -> str:
        """Run the full normalization pipeline and return normalized code."""
        code = self._strip_comments(code)
        code = _normalize_strings(code)
        code = _normalize_numbers(code)
        code = self._normalize_identifiers(code)
        code = _normalize_whitespace(code)
        return code

    @abstractmethod
    def _strip_comments(self, code: str) -> str:
        """Remove language-specific comments."""

    @abstractmethod
    def _normalize_identifiers(self, code: str) -> str:
        """Rename function name to FUNC_NAME and locals to v1, v2, ..."""


# ---------------------------------------------------------------------------
# Shared helpers (language-agnostic)
# ---------------------------------------------------------------------------

def _normalize_strings(code: str) -> str:
    code = re.sub(r'"(?:[^"\\]|\\.)*"', "STR_LITERAL", code)
    code = re.sub(r"'(?:[^'\\]|\\.)*'", "STR_LITERAL", code)
    return code


def _normalize_numbers(code: str) -> str:
    code = re.sub(r"\b\d+\.\d+\b", "NUM_LITERAL", code)
    code = re.sub(r"\b\d+\b", "NUM_LITERAL", code)
    return code


def _normalize_whitespace(code: str) -> str:
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = re.sub(r"[ \t]+", " ", code)
    code = re.sub(r"\n\s*\n+", "\n", code)
    return code.strip()
