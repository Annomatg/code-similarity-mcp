"""Base parser interface and shared data models."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


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

    @property
    def loc(self) -> int:
        return self.end_line - self.start_line + 1

    @property
    def code_hash(self) -> str:
        return hashlib.sha256(self.normalized_code.encode()).hexdigest()


class BaseParser(ABC):
    """Abstract parser for a specific language."""

    @abstractmethod
    def parse_file(self, file_path: str) -> list[MethodInfo]:
        """Parse a source file and return all extracted methods."""

    @abstractmethod
    def parse_snippet(self, code: str, language: str) -> list[MethodInfo]:
        """Parse an in-memory code snippet and return extracted methods."""
