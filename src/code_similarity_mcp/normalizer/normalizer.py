"""Normalize source code to remove superficial variation."""

from __future__ import annotations

import re


def normalize_code(code: str, language: str = "gdscript") -> str:
    """
    Normalize code for similarity comparison:
    - Strip comments
    - Normalize string literals -> STR_LITERAL
    - Normalize numeric literals -> NUM_LITERAL
    - Normalize function name -> FUNC_NAME
    - Rename parameters and local var declarations -> v1, v2, ...
    - Collapse whitespace
    """
    code = _strip_comments(code, language)
    code = _normalize_strings(code)
    code = _normalize_numbers(code)
    code = _normalize_identifiers(code)
    code = _normalize_whitespace(code)
    return code


def _strip_comments(code: str, language: str) -> str:
    if language in ("gdscript", "python"):
        code = re.sub(r"#[^\n]*", "", code)
    return code


def _normalize_strings(code: str) -> str:
    code = re.sub(r'"(?:[^"\\]|\\.)*"', "STR_LITERAL", code)
    code = re.sub(r"'(?:[^'\\]|\\.)*'", "STR_LITERAL", code)
    return code


def _normalize_numbers(code: str) -> str:
    code = re.sub(r"\b\d+\.\d+\b", "NUM_LITERAL", code)
    code = re.sub(r"\b\d+\b", "NUM_LITERAL", code)
    return code


def _normalize_identifiers(code: str) -> str:
    """
    Normalize local identifiers in order of collection:
    1. Function name -> FUNC_NAME
    2. Parameters -> v1, v2, ...
    3. var declarations -> vN, vN+1, ...
    """
    # Normalize function name
    code = re.sub(r"\bfunc\s+([A-Za-z_]\w*)\s*\(", "func FUNC_NAME(", code)

    # Collect parameters from signature
    params: list[str] = []
    sig_match = re.search(r"\bfunc\s+FUNC_NAME\s*\(([^)]*)\)", code)
    if sig_match:
        params_str = sig_match.group(1)
        for part in params_str.split(","):
            part = part.strip()
            if not part:
                continue
            # Handle: name, name: Type, name: Type = default
            ident_match = re.match(r"([A-Za-z_]\w*)", part)
            if ident_match:
                name = ident_match.group(1)
                if name not in ("void",):
                    params.append(name)

    # Collect var declarations
    var_names = re.findall(r"\bvar\s+([A-Za-z_]\w*)", code)

    # Rename in order, deduplicated
    all_locals = list(dict.fromkeys(params + var_names))
    for i, name in enumerate(all_locals, start=1):
        code = re.sub(rf"\b{re.escape(name)}\b", f"v{i}", code)

    return code


def _normalize_whitespace(code: str) -> str:
    code = re.sub(r"[ \t]+", " ", code)
    code = re.sub(r"\n\s*\n+", "\n", code)
    return code.strip()
