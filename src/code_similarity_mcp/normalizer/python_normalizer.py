"""Python-specific normalizer."""

from __future__ import annotations

import re

from .base import BaseNormalizer


class PythonNormalizer(BaseNormalizer):
    """Normalizes Python source code."""

    def _strip_comments(self, code: str) -> str:
        return re.sub(r"#[^\n]*", "", code)

    def _normalize_identifiers(self, code: str) -> str:
        """
        1. Rename function name -> FUNC_NAME
        2. Collect parameters (excluding self/cls) -> v1, v2, ...
        3. Collect local assignment targets -> vN, vN+1, ...
        4. Rename all collected names in order of first appearance.
        """
        # Step 1: function name
        code = re.sub(r"\bdef\s+([A-Za-z_]\w*)\s*\(", "def FUNC_NAME(", code)

        # Step 2: parameters from signature
        params: list[str] = []
        sig_match = re.search(r"\bdef\s+FUNC_NAME\s*\(([^)]*)\)", code, re.DOTALL)
        if sig_match:
            for part in sig_match.group(1).split(","):
                part = part.strip().lstrip("*")
                if not part:
                    continue
                m = re.match(r"([A-Za-z_]\w*)", part)
                if m and m.group(1) not in ("self", "cls"):
                    params.append(m.group(1))

        # Step 3: local assignment targets at function-body indentation.
        # Matches:  "    name = ..." and "    name: Type = ..."
        # Excludes: augmented assignments (+=, etc.) and comparisons (==).
        var_names = re.findall(
            r"^\s+([A-Za-z_]\w*)\s*(?::[^=\n]*)?\s*=(?!=)",
            code,
            re.MULTILINE,
        )

        # Step 4: rename in first-seen order, deduplicated
        all_locals = list(dict.fromkeys(params + var_names))
        for i, name in enumerate(all_locals, start=1):
            code = re.sub(rf"\b{re.escape(name)}\b", f"v{i}", code)

        return code
