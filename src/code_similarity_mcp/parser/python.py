"""Python parser using tree-sitter."""

from __future__ import annotations

from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from .base import BaseParser, MethodInfo, StatementInfo

_PY_LANGUAGE = Language(tspython.language())

# Statement node types used for complexity counting
_STATEMENT_TYPES = frozenset({
    # Simple statements
    "expression_statement",
    "return_statement",
    "delete_statement",
    "raise_statement",
    "pass_statement",
    "break_statement",
    "continue_statement",
    "import_statement",
    "import_from_statement",
    "assert_statement",
    "global_statement",
    "nonlocal_statement",
    # Compound statements (header counts as one statement)
    "if_statement",
    "for_statement",
    "while_statement",
    "try_statement",
    "with_statement",
    "match_statement",
    # Nested definitions inside a function body
    "function_definition",
    "class_definition",
    "decorated_definition",
})

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


def _has_abstractmethod_decorator(node, source: bytes) -> bool:
    """Return True if this function_definition is decorated with @abstractmethod."""
    parent = node.parent
    if parent is None or parent.type != "decorated_definition":
        return False
    for child in parent.children:
        if child.type != "decorator":
            continue
        decorator_text = _node_text(child, source)
        # Matches both '@abstractmethod' and '@abc.abstractmethod'
        if "abstractmethod" in decorator_text:
            return True
    return False


def _is_stub_body(block_node) -> bool:
    """Return True if the function body consists only of pass/ellipsis/docstring."""
    for child in block_node.children:
        if not child.is_named:
            continue  # skip punctuation / anonymous tokens
        if child.type == "pass_statement":
            continue
        if child.type == "expression_statement":
            named_inner = [c for c in child.children if c.is_named]
            if len(named_inner) == 1 and named_inner[0].type in ("ellipsis", "string"):
                continue
        # Anything else is a real statement — not a stub
        return False
    return True


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


def count_statements(code: str) -> int:
    """Count all statement nodes in a Python code snippet (recursive, all depths).

    Uses tree-sitter to parse *code* and walks the resulting AST, counting
    every node whose type belongs to ``_STATEMENT_TYPES``.  This gives a
    measure of function complexity that is independent of raw line count.
    """
    source = code.encode("utf-8")
    parser = Parser(_PY_LANGUAGE)
    tree = parser.parse(source)
    return sum(1 for node in _walk(tree.root_node) if node.type in _STATEMENT_TYPES)


def get_top_level_statements(code: str) -> list[StatementInfo]:
    """Return the ordered list of top-level statements in a Python function body.

    Parses *code* with tree-sitter, locates the first ``function_definition``
    node, and collects the **direct child** statement nodes of its body block.
    Statements nested inside compound blocks (e.g. the body of an ``if`` or
    ``for``) are intentionally excluded — only the outermost statements are
    returned.

    Args:
        code: Source text of a Python function (or a module that contains one).

    Returns:
        A list of :class:`StatementInfo` named-tuples, one per direct child
        statement of the function body, in source order.  Returns an empty
        list if no function definition is found in *code*.
    """
    source = code.encode("utf-8")
    parser = Parser(_PY_LANGUAGE)
    tree = parser.parse(source)

    # Find the first function_definition node
    func_node = None
    for node in _walk(tree.root_node):
        if node.type == "function_definition":
            func_node = node
            break

    if func_node is None:
        return []

    # Locate the block (body) child of the function
    block_node = None
    for child in func_node.children:
        if child.type == "block":
            block_node = child
            break

    if block_node is None:
        return []

    # Collect direct named child statement nodes — do NOT recurse
    statements: list[StatementInfo] = []
    for child in block_node.children:
        if child.is_named and child.type in _STATEMENT_TYPES:
            statements.append(StatementInfo(
                index=len(statements),
                node_type=child.type,
                start_line=child.start_point[0] + 1,   # 1-based
                end_line=child.end_point[0] + 1,
                source_text=_node_text(child, source),
            ))

    return statements


# ---------------------------------------------------------------------------
# Variable dependency graph helpers
# ---------------------------------------------------------------------------

def _collect_lhs_writes(node, source: bytes, writes: set) -> None:
    """Collect identifier names written (assigned) by an LHS pattern."""
    t = node.type
    if t == "identifier":
        writes.add(_node_text(node, source))
    elif t in ("pattern_list", "tuple_pattern", "list_pattern"):
        for child in node.children:
            if child.is_named:
                _collect_lhs_writes(child, source, writes)
    elif t == "star_expression":
        # *rest in tuple unpacking
        for child in node.children:
            if child.type == "identifier":
                writes.add(_node_text(child, source))
    # subscript (x[i] = v) and attribute (x.y = v) don't introduce new bindings


def _collect_reads_expr(node, source: bytes, reads: set) -> None:
    """Collect all identifier names appearing in an expression (all are reads)."""
    if node.type == "identifier":
        reads.add(_node_text(node, source))
        return
    for child in node.children:
        _collect_reads_expr(child, source, reads)


def _collect_import_writes(node, source: bytes, writes: set) -> None:
    """Collect names bound by an import or import-from statement."""
    t = node.type
    if t == "import_statement":
        for name_node in node.children_by_field_name("name"):
            if name_node.type == "dotted_name":
                # import foo.bar  →  binds 'foo'
                for child in name_node.children:
                    if child.type == "identifier":
                        writes.add(_node_text(child, source))
                        break
            elif name_node.type == "aliased_import":
                alias = name_node.child_by_field_name("alias")
                if alias is not None:
                    writes.add(_node_text(alias, source))
    elif t == "import_from_statement":
        for name_node in node.children_by_field_name("name"):
            if name_node.type == "dotted_name":
                # from x import foo  →  binds 'foo' (last component)
                last = None
                for child in name_node.children:
                    if child.type == "identifier":
                        last = child
                if last is not None:
                    writes.add(_node_text(last, source))
            elif name_node.type == "aliased_import":
                alias = name_node.child_by_field_name("alias")
                if alias is not None:
                    writes.add(_node_text(alias, source))


def _collect_stmt_wr(node, source: bytes, writes: set, reads: set) -> None:
    """Recursively collect writes and reads from a statement node.

    Recurses into nested blocks (if/for/while/with/try bodies) but stops
    at nested function and class definitions (they introduce a new scope).
    """
    t = node.type

    if t == "expression_statement":
        for child in node.children:
            if child.is_named:
                _collect_stmt_wr(child, source, writes, reads)

    elif t == "assignment":
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        if left is not None:
            _collect_lhs_writes(left, source, writes)
        if right is not None:
            _collect_reads_expr(right, source, reads)

    elif t == "augmented_assignment":
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        if left is not None:
            _collect_reads_expr(left, source, reads)  # reads current value
            if left.type == "identifier":
                writes.add(_node_text(left, source))
        if right is not None:
            _collect_reads_expr(right, source, reads)

    elif t == "named_expression":
        name = node.child_by_field_name("name")
        value = node.child_by_field_name("value")
        if name is not None and name.type == "identifier":
            writes.add(_node_text(name, source))
        if value is not None:
            _collect_reads_expr(value, source, reads)

    elif t == "for_statement":
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        body = node.child_by_field_name("body")
        alt = node.child_by_field_name("alternative")
        if left is not None:
            _collect_lhs_writes(left, source, writes)
        if right is not None:
            _collect_reads_expr(right, source, reads)
        if body is not None:
            _collect_stmt_wr(body, source, writes, reads)
        if alt is not None:
            _collect_stmt_wr(alt, source, writes, reads)

    elif t == "while_statement":
        cond = node.child_by_field_name("condition")
        body = node.child_by_field_name("body")
        alt = node.child_by_field_name("alternative")
        if cond is not None:
            _collect_reads_expr(cond, source, reads)
        if body is not None:
            _collect_stmt_wr(body, source, writes, reads)
        if alt is not None:
            _collect_stmt_wr(alt, source, writes, reads)

    elif t == "if_statement":
        cond = node.child_by_field_name("condition")
        consequence = node.child_by_field_name("consequence")
        if cond is not None:
            _collect_reads_expr(cond, source, reads)
        if consequence is not None:
            _collect_stmt_wr(consequence, source, writes, reads)
        for alt in node.children_by_field_name("alternative"):
            _collect_stmt_wr(alt, source, writes, reads)

    elif t == "elif_clause":
        cond = node.child_by_field_name("condition")
        consequence = node.child_by_field_name("consequence")
        if cond is not None:
            _collect_reads_expr(cond, source, reads)
        if consequence is not None:
            _collect_stmt_wr(consequence, source, writes, reads)

    elif t == "else_clause":
        body = node.child_by_field_name("body")
        if body is not None:
            _collect_stmt_wr(body, source, writes, reads)

    elif t == "with_statement":
        for child in node.children:
            if child.type == "with_clause":
                for item in child.children:
                    if item.type == "with_item":
                        val = item.child_by_field_name("value")
                        alias = item.child_by_field_name("alias")
                        if val is not None:
                            _collect_reads_expr(val, source, reads)
                        if alias is not None:
                            _collect_lhs_writes(alias, source, writes)
            elif child.type == "with_item":
                val = child.child_by_field_name("value")
                alias = child.child_by_field_name("alias")
                if val is not None:
                    _collect_reads_expr(val, source, reads)
                if alias is not None:
                    _collect_lhs_writes(alias, source, writes)
            elif child.type == "block":
                _collect_stmt_wr(child, source, writes, reads)

    elif t == "try_statement":
        for child in node.children:
            if child.is_named:
                _collect_stmt_wr(child, source, writes, reads)

    elif t == "except_clause":
        # except ExcType as name: body
        # The bound name is the identifier that follows the anonymous 'as' token.
        found_as = False
        for child in node.children:
            if not child.is_named and _node_text(child, source) == "as":
                found_as = True
            elif found_as and child.type == "identifier":
                writes.add(_node_text(child, source))
                found_as = False
            elif child.type == "block":
                _collect_stmt_wr(child, source, writes, reads)
            elif child.is_named and child.type != "block":
                _collect_reads_expr(child, source, reads)

    elif t in ("import_statement", "import_from_statement"):
        _collect_import_writes(node, source, writes)

    elif t in ("function_definition", "class_definition"):
        name = node.child_by_field_name("name")
        if name is not None and name.type == "identifier":
            writes.add(_node_text(name, source))
        # Do NOT recurse into body — nested scope

    elif t == "decorated_definition":
        for child in node.children:
            if child.type == "decorator":
                _collect_reads_expr(child, source, reads)
            elif child.type in ("function_definition", "class_definition"):
                name = child.child_by_field_name("name")
                if name is not None and name.type == "identifier":
                    writes.add(_node_text(name, source))

    elif t == "block":
        for child in node.children:
            if child.is_named:
                _collect_stmt_wr(child, source, writes, reads)

    elif t == "return_statement":
        for child in node.children:
            if child.is_named:
                _collect_reads_expr(child, source, reads)

    elif t in ("raise_statement", "assert_statement", "delete_statement"):
        for child in node.children:
            if child.is_named:
                _collect_reads_expr(child, source, reads)

    elif t in ("pass_statement", "break_statement", "continue_statement"):
        pass  # no reads or writes

    elif t in ("global_statement", "nonlocal_statement"):
        pass  # scope declarations, not data reads/writes

    else:
        # Generic fallback: treat all identifiers as reads
        _collect_reads_expr(node, source, reads)


def _get_stmt_writes_reads(stmt_node, source: bytes) -> tuple[set[str], set[str]]:
    """Return (writes, reads) variable sets for a single top-level statement."""
    writes: set[str] = set()
    reads: set[str] = set()
    _collect_stmt_wr(stmt_node, source, writes, reads)
    return writes, reads


def build_dependency_graph(code: str) -> dict[int, list[int]]:
    """Build a variable dependency graph for the first function in *code*.

    Analyzes the **top-level** statements of the function body.  For each
    ordered pair (A, B) where A.index < B.index: if statement B reads a
    variable written by statement A, a directed edge ``A → B`` is recorded.

    The returned graph is an adjacency list indexed by zero-based statement
    index: ``graph[i]`` contains the sorted list of statement indices that
    consume outputs produced by statement ``i``.

    Args:
        code: Source text of a Python function (or a module containing one).

    Returns:
        Adjacency list ``dict[int, list[int]]`` where ``graph[A] = [B, C, ...]``
        means statements B and C depend on variables written by A.  All
        statement indices 0..n-1 appear as keys (empty list when no downstream
        dependents exist).  Returns ``{}`` if *code* contains no function
        definition or the function body has no statements.
    """
    source = code.encode("utf-8")
    _parser = Parser(_PY_LANGUAGE)
    tree = _parser.parse(source)

    func_node = None
    for n in _walk(tree.root_node):
        if n.type == "function_definition":
            func_node = n
            break

    if func_node is None:
        return {}

    block_node = None
    for child in func_node.children:
        if child.type == "block":
            block_node = child
            break

    if block_node is None:
        return {}

    stmt_nodes = [
        child for child in block_node.children
        if child.is_named and child.type in _STATEMENT_TYPES
    ]

    if not stmt_nodes:
        return {}

    num = len(stmt_nodes)
    stmt_vars = [_get_stmt_writes_reads(s, source) for s in stmt_nodes]

    graph: dict[int, list[int]] = {i: [] for i in range(num)}

    for a in range(num):
        a_writes = stmt_vars[a][0]
        if not a_writes:
            continue
        for b in range(a + 1, num):
            b_reads = stmt_vars[b][1]
            if a_writes & b_reads:
                graph[a].append(b)

    return graph


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
        block_node = None

        for child in node.children:
            if child.type == "identifier":
                name = _node_text(child, source)
            elif child.type == "parameters":
                params = _extract_params(child, source)
            elif child.type == "type":
                return_type = _node_text(child, source)
            elif child.type == "block":
                block_node = child

        if name is None:
            return None

        is_stub = _has_abstractmethod_decorator(node, source) or (
            block_node is not None and _is_stub_body(block_node)
        )

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
            is_stub=is_stub,
        )
