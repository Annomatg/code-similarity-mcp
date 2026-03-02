"""Tests for the Python parser (tree-sitter based)."""

import textwrap
import hashlib

import pytest

from code_similarity_mcp.parser.python import (
    PythonParser, count_statements, get_top_level_statements, build_dependency_graph,
)
from code_similarity_mcp.parser.base import (
    DependencyGraph, MethodInfo, StatementInfo, group_into_chunks, annotate_chunks, ChunkInfo,
)
from code_similarity_mcp.parser.registry import get_parser, SUPPORTED_EXTENSIONS


def parse(code: str) -> list[MethodInfo]:
    return PythonParser().parse_snippet(textwrap.dedent(code))


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------

class TestBasicExtraction:
    def test_simple_function_name(self):
        methods = parse("""\
            def greet():
                pass
        """)
        assert len(methods) == 1
        assert methods[0].name == "greet"

    def test_function_with_params(self):
        methods = parse("""\
            def add(a, b):
                return a + b
        """)
        assert methods[0].parameters == ["a", "b"]

    def test_function_no_params(self):
        methods = parse("""\
            def init():
                pass
        """)
        assert methods[0].parameters == []

    def test_multiple_functions(self):
        methods = parse("""\
            def foo():
                pass

            def bar():
                pass

            def baz():
                pass
        """)
        assert [m.name for m in methods] == ["foo", "bar", "baz"]

    def test_language_is_python(self):
        methods = parse("def f():\n    pass\n")
        assert methods[0].language == "python"

    def test_file_path_snippet(self):
        methods = parse("def f():\n    pass\n")
        assert methods[0].file_path == "<snippet>"


# ---------------------------------------------------------------------------
# Type annotations
# ---------------------------------------------------------------------------

class TestTypeAnnotations:
    def test_typed_params_names_only(self):
        methods = parse("""\
            def scale(value: float, factor: int) -> float:
                return value * factor
        """)
        assert methods[0].parameters == ["value", "factor"]

    def test_return_type_extracted(self):
        methods = parse("""\
            def get_name() -> str:
                return self.name
        """)
        assert methods[0].return_type == "str"

    def test_return_type_none_when_absent(self):
        methods = parse("""\
            def do_thing():
                pass
        """)
        assert methods[0].return_type is None

    def test_params_with_defaults(self):
        methods = parse("""\
            def connect(host: str, port: int = 8080):
                pass
        """)
        assert methods[0].parameters == ["host", "port"]

    def test_typed_default_parameter(self):
        methods = parse("""\
            def f(x: int = 0, y: float = 1.0):
                pass
        """)
        assert methods[0].parameters == ["x", "y"]

    def test_args_kwargs(self):
        methods = parse("""\
            def f(*args, **kwargs):
                pass
        """)
        assert methods[0].parameters == ["args", "kwargs"]

    def test_self_excluded_from_params(self):
        methods = parse("""\
            def method(self, a, b):
                return a + b
        """)
        assert methods[0].parameters == ["a", "b"]

    def test_cls_excluded_from_params(self):
        methods = parse("""\
            def from_dict(cls, data):
                pass
        """)
        assert methods[0].parameters == ["data"]


# ---------------------------------------------------------------------------
# Line ranges
# ---------------------------------------------------------------------------

class TestLineRanges:
    def test_start_line_one_based(self):
        methods = parse("def f():\n    pass\n")
        assert methods[0].start_line == 1

    def test_multiline_body_end_line(self):
        methods = parse("""\
            def calculate(a, b):
                x = a + b
                y = x * 2
                return y
        """)
        m = methods[0]
        assert m.start_line == 1
        assert m.end_line == 4
        assert m.loc == 4

    def test_second_function_start_line(self):
        methods = parse("""\
            def first():
                pass

            def second():
                pass
        """)
        assert methods[1].start_line == 4

    def test_loc_property(self):
        methods = parse("""\
            def f():
                return 1
        """)
        assert methods[0].loc == 2


# ---------------------------------------------------------------------------
# Body code
# ---------------------------------------------------------------------------

class TestBodyCode:
    def test_body_includes_header(self):
        methods = parse("""\
            def add(a, b):
                return a + b
        """)
        assert "def add" in methods[0].body_code
        assert "return a + b" in methods[0].body_code

    def test_body_excludes_sibling_function(self):
        methods = parse("""\
            def foo():
                return 1

            def bar():
                return 2
        """)
        assert "bar" not in methods[0].body_code

    def test_empty_body_pass(self):
        methods = parse("""\
            def noop():
                pass
        """)
        assert "pass" in methods[0].body_code


# ---------------------------------------------------------------------------
# Dependency extraction
# ---------------------------------------------------------------------------

class TestDependencies:
    def test_direct_call_extracted(self):
        methods = parse("""\
            def process():
                x = helper()
                return x
        """)
        assert "helper" in methods[0].dependencies

    def test_method_call_extracted(self):
        methods = parse("""\
            def run(self):
                self.emit_signal("done")
        """)
        assert "emit_signal" in methods[0].dependencies

    def test_self_function_not_in_deps(self):
        methods = parse("""\
            def recurse(n):
                if n <= 0:
                    return 0
                return recurse(n - 1)
        """)
        assert "recurse" not in methods[0].dependencies

    def test_multiple_calls(self):
        methods = parse("""\
            def setup():
                load_config()
                init_ui()
                connect_signals()
        """)
        deps = methods[0].dependencies
        assert "load_config" in deps
        assert "init_ui" in deps
        assert "connect_signals" in deps

    def test_deps_are_sorted(self):
        methods = parse("""\
            def f():
                z_func()
                a_func()
                m_func()
        """)
        deps = methods[0].dependencies
        assert deps == sorted(deps)

    def test_no_deps_when_no_calls(self):
        methods = parse("""\
            def pure(a, b):
                return a + b
        """)
        assert methods[0].dependencies == []

    def test_builtin_calls_included(self):
        methods = parse("""\
            def f(items):
                return len(items)
        """)
        assert "len" in methods[0].dependencies


# ---------------------------------------------------------------------------
# Class methods
# ---------------------------------------------------------------------------

class TestClassMethods:
    def test_methods_inside_class_extracted(self):
        methods = parse("""\
            class MyClass:
                def compute(self, x):
                    return x * 2

                def reset(self):
                    self.value = 0
        """)
        names = [m.name for m in methods]
        assert "compute" in names
        assert "reset" in names

    def test_class_method_params_exclude_self(self):
        methods = parse("""\
            class C:
                def method(self, a, b):
                    return a + b
        """)
        m = next(m for m in methods if m.name == "method")
        assert m.parameters == ["a", "b"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_source(self):
        assert parse("") == []

    def test_source_with_no_functions(self):
        methods = parse("""\
            x = 1
            y = x + 2
        """)
        assert methods == []

    def test_function_with_underscore_prefix(self):
        methods = parse("""\
            def _internal():
                pass
        """)
        assert methods[0].name == "_internal"

    def test_dunder_method(self):
        methods = parse("""\
            def __init__(self, x):
                self.x = x
        """)
        assert methods[0].name == "__init__"

    def test_parse_file(self, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("def greet():\n    pass\n", encoding="utf-8")
        parser = PythonParser()
        methods = parser.parse_file(str(py_file))
        assert len(methods) == 1
        assert methods[0].name == "greet"
        assert methods[0].file_path == str(py_file)

    def test_normalized_code_empty_initially(self):
        methods = parse("def f():\n    pass\n")
        assert methods[0].normalized_code == ""

    def test_code_hash_uses_normalized(self):
        methods = parse("def f():\n    pass\n")
        m = methods[0]
        m.normalized_code = "def FUNC_NAME():\n pass\n"
        expected = hashlib.sha256(m.normalized_code.encode()).hexdigest()
        assert m.code_hash == expected

    def test_nested_functions_extracted(self):
        methods = parse("""\
            def outer():
                def inner():
                    pass
                inner()
        """)
        names = [m.name for m in methods]
        assert "outer" in names
        assert "inner" in names


# ---------------------------------------------------------------------------
# Stub detection
# ---------------------------------------------------------------------------

class TestStubDetection:
    def test_real_method_is_not_stub(self):
        methods = parse("""\
            def add(a, b):
                return a + b
        """)
        assert methods[0].is_stub is False

    def test_pass_only_body_is_stub(self):
        methods = parse("""\
            def noop():
                pass
        """)
        assert methods[0].is_stub is True

    def test_ellipsis_only_body_is_stub(self):
        methods = parse("""\
            def noop():
                ...
        """)
        assert methods[0].is_stub is True

    def test_docstring_only_body_is_stub(self):
        methods = parse('''\
            def describe():
                """Returns nothing."""
        ''')
        assert methods[0].is_stub is True

    def test_docstring_then_pass_is_stub(self):
        methods = parse('''\
            def describe():
                """Does nothing."""
                pass
        ''')
        assert methods[0].is_stub is True

    def test_docstring_then_ellipsis_is_stub(self):
        methods = parse('''\
            def describe():
                """Does nothing."""
                ...
        ''')
        assert methods[0].is_stub is True

    def test_abstractmethod_decorator_is_stub(self):
        methods = parse("""\
            class Base:
                @abstractmethod
                def process(self, data):
                    pass
        """)
        m = next(m for m in methods if m.name == "process")
        assert m.is_stub is True

    def test_abc_abstractmethod_decorator_is_stub(self):
        methods = parse("""\
            class Base:
                @abc.abstractmethod
                def process(self, data):
                    pass
        """)
        m = next(m for m in methods if m.name == "process")
        assert m.is_stub is True

    def test_abstractmethod_with_real_body_still_stub(self):
        """@abstractmethod takes precedence even if body has code."""
        methods = parse("""\
            class Base:
                @abstractmethod
                def compute(self, x):
                    return x * 2
        """)
        m = next(m for m in methods if m.name == "compute")
        assert m.is_stub is True

    def test_real_method_with_decorator_is_not_stub(self):
        methods = parse("""\
            class Service:
                @property
                def value(self):
                    return self._value
        """)
        m = next(m for m in methods if m.name == "value")
        assert m.is_stub is False

    def test_method_with_real_body_is_not_stub(self):
        methods = parse("""\
            def calculate(a, b):
                x = a + b
                y = x * 2
                return y
        """)
        assert methods[0].is_stub is False


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_python_extension_registered(self):
        assert ".py" in SUPPORTED_EXTENSIONS
        assert SUPPORTED_EXTENSIONS[".py"] == "python"

    def test_get_parser_returns_python_parser(self):
        parser = get_parser("python")
        assert isinstance(parser, PythonParser)

    def test_get_parser_cached(self):
        p1 = get_parser("python")
        p2 = get_parser("python")
        assert p1 is p2

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            get_parser("cobol")


# ---------------------------------------------------------------------------
# count_statements utility
# ---------------------------------------------------------------------------

class TestCountStatements:
    def test_single_return_is_one(self):
        code = "def f():\n    return 1\n"
        assert count_statements(code) == 2  # function_definition + return_statement

    def test_empty_function_is_one(self):
        # function_definition itself + pass_statement
        code = "def f():\n    pass\n"
        assert count_statements(code) == 2

    def test_sequential_statements(self):
        code = textwrap.dedent("""\
            def f(a, b):
                x = a + b
                y = x * 2
                return y
        """)
        # function_definition + 2 expression_statements + return_statement = 4
        assert count_statements(code) == 4

    def test_if_counts_as_one_statement(self):
        code = textwrap.dedent("""\
            def f(x):
                if x > 0:
                    return x
                return 0
        """)
        # function_definition + if_statement + return (inside if) + return = 4
        assert count_statements(code) == 4

    def test_for_loop_counts_as_one(self):
        code = textwrap.dedent("""\
            def f(items):
                for item in items:
                    print(item)
        """)
        # function_definition + for_statement + expression_statement = 3
        assert count_statements(code) == 3

    def test_nested_control_flow_counted_recursively(self):
        code = textwrap.dedent("""\
            def f(items):
                result = []
                for item in items:
                    if item > 0:
                        result.append(item)
                return result
        """)
        # function_definition + expression_stmt (result=[]) + for_stmt
        # + if_stmt + expression_stmt (append) + return_stmt = 6
        assert count_statements(code) == 6

    def test_while_loop_counted(self):
        code = textwrap.dedent("""\
            def f(n):
                while n > 0:
                    n -= 1
                return n
        """)
        # function_definition + while_statement + expression_statement + return_statement = 4
        assert count_statements(code) == 4

    def test_try_except_counted(self):
        code = textwrap.dedent("""\
            def f():
                try:
                    x = 1
                except ValueError:
                    x = 0
                return x
        """)
        # function_definition + try_statement + expression_stmt (x=1)
        # + expression_stmt (x=0) + return_statement = 5
        assert count_statements(code) == 5

    def test_import_statement_counted(self):
        code = textwrap.dedent("""\
            def f():
                import os
                return os.getcwd()
        """)
        # function_definition + import_statement + return_statement = 3
        assert count_statements(code) == 3

    def test_returns_zero_for_empty_string(self):
        assert count_statements("") == 0

    def test_returns_zero_for_no_statements(self):
        # Just an expression without a function
        assert count_statements("x = 1") == 1  # expression_statement

    def test_large_function_exceeds_threshold(self):
        """A function with many statements should clearly exceed 30."""
        lines = ["def big_func():"]
        for i in range(35):
            lines.append(f"    x_{i} = {i}")
        code = "\n".join(lines) + "\n"
        stmt_count = count_statements(code)
        assert stmt_count > 30

    def test_small_function_below_threshold(self):
        """A small function with a few statements should be below 30."""
        code = textwrap.dedent("""\
            def small(a, b):
                x = a + b
                return x
        """)
        assert count_statements(code) <= 30

    def test_nested_function_definition_counted(self):
        """A nested function_definition inside a function counts as a statement."""
        code = textwrap.dedent("""\
            def outer():
                def inner():
                    pass
                inner()
        """)
        # outer function_definition + inner function_definition + pass + expression_stmt = 4
        assert count_statements(code) == 4


# ---------------------------------------------------------------------------
# get_top_level_statements
# ---------------------------------------------------------------------------

class TestGetTopLevelStatements:
    # ------------------------------------------------------------------
    # Basic return type and structure
    # ------------------------------------------------------------------

    def test_returns_list_of_statement_info(self):
        code = "def f():\n    pass\n"
        result = get_top_level_statements(code)
        assert isinstance(result, list)
        assert all(isinstance(s, StatementInfo) for s in result)

    def test_no_function_returns_empty_list(self):
        result = get_top_level_statements("x = 1\n")
        assert result == []

    def test_empty_string_returns_empty_list(self):
        assert get_top_level_statements("") == []

    # ------------------------------------------------------------------
    # Simple single-statement functions
    # ------------------------------------------------------------------

    def test_pass_statement(self):
        code = "def f():\n    pass\n"
        stmts = get_top_level_statements(code)
        assert len(stmts) == 1
        assert stmts[0].node_type == "pass_statement"

    def test_return_statement(self):
        code = "def f():\n    return 1\n"
        stmts = get_top_level_statements(code)
        assert len(stmts) == 1
        assert stmts[0].node_type == "return_statement"

    def test_expression_statement(self):
        code = "def f():\n    x = 1\n"
        stmts = get_top_level_statements(code)
        assert len(stmts) == 1
        assert stmts[0].node_type == "expression_statement"

    # ------------------------------------------------------------------
    # Ordering and indices
    # ------------------------------------------------------------------

    def test_index_is_zero_based(self):
        code = textwrap.dedent("""\
            def f(a, b):
                x = a + b
                return x
        """)
        stmts = get_top_level_statements(code)
        assert stmts[0].index == 0
        assert stmts[1].index == 1

    def test_multiple_statements_ordered(self):
        code = textwrap.dedent("""\
            def process(items):
                result = []
                for item in items:
                    result.append(item)
                return result
        """)
        stmts = get_top_level_statements(code)
        assert len(stmts) == 3
        assert stmts[0].node_type == "expression_statement"
        assert stmts[1].node_type == "for_statement"
        assert stmts[2].node_type == "return_statement"

    def test_indices_are_sequential(self):
        code = textwrap.dedent("""\
            def f():
                a = 1
                b = 2
                c = 3
        """)
        stmts = get_top_level_statements(code)
        assert [s.index for s in stmts] == [0, 1, 2]

    # ------------------------------------------------------------------
    # Node types
    # ------------------------------------------------------------------

    def test_if_statement_type(self):
        code = textwrap.dedent("""\
            def f(x):
                if x > 0:
                    return x
                return 0
        """)
        stmts = get_top_level_statements(code)
        assert stmts[0].node_type == "if_statement"
        assert stmts[1].node_type == "return_statement"

    def test_while_statement_type(self):
        code = textwrap.dedent("""\
            def f(n):
                while n > 0:
                    n -= 1
                return n
        """)
        stmts = get_top_level_statements(code)
        assert stmts[0].node_type == "while_statement"
        assert stmts[1].node_type == "return_statement"

    def test_try_statement_type(self):
        code = textwrap.dedent("""\
            def f():
                try:
                    x = 1
                except ValueError:
                    x = 0
                return x
        """)
        stmts = get_top_level_statements(code)
        assert stmts[0].node_type == "try_statement"
        assert stmts[1].node_type == "return_statement"

    def test_nested_function_definition_type(self):
        code = textwrap.dedent("""\
            def outer():
                def inner():
                    pass
                return inner
        """)
        stmts = get_top_level_statements(code)
        assert stmts[0].node_type == "function_definition"
        assert stmts[1].node_type == "return_statement"

    # ------------------------------------------------------------------
    # Line numbers
    # ------------------------------------------------------------------

    def test_start_line_1_based(self):
        code = "def f():\n    return 1\n"
        stmts = get_top_level_statements(code)
        assert stmts[0].start_line == 2

    def test_multiline_statement_end_line(self):
        code = textwrap.dedent("""\
            def f(x):
                if x > 0:
                    return x
                return 0
        """)
        stmts = get_top_level_statements(code)
        # if_statement spans lines 2-3
        assert stmts[0].start_line == 2
        assert stmts[0].end_line == 3
        # return spans line 4
        assert stmts[1].start_line == 4
        assert stmts[1].end_line == 4

    def test_single_line_statement_same_start_end(self):
        code = "def f():\n    pass\n"
        stmts = get_top_level_statements(code)
        s = stmts[0]
        assert s.start_line == s.end_line

    # ------------------------------------------------------------------
    # Source text
    # ------------------------------------------------------------------

    def test_source_text_for_pass(self):
        code = "def f():\n    pass\n"
        stmts = get_top_level_statements(code)
        assert stmts[0].source_text == "pass"

    def test_source_text_for_return(self):
        code = "def f(a, b):\n    return a + b\n"
        stmts = get_top_level_statements(code)
        assert stmts[0].source_text == "return a + b"

    def test_source_text_for_assignment(self):
        code = "def f():\n    x = 42\n"
        stmts = get_top_level_statements(code)
        assert "x = 42" in stmts[0].source_text

    def test_source_text_for_for_loop_includes_header_only(self):
        """source_text of a for_statement covers its full extent including nested body."""
        code = textwrap.dedent("""\
            def f(items):
                for item in items:
                    print(item)
        """)
        stmts = get_top_level_statements(code)
        assert stmts[0].node_type == "for_statement"
        assert "for item in items" in stmts[0].source_text
        assert "print(item)" in stmts[0].source_text

    # ------------------------------------------------------------------
    # Nesting: nested statements must NOT appear at top level
    # ------------------------------------------------------------------

    def test_nested_statements_excluded_from_top_level(self):
        """Statements inside an if block must not appear as top-level entries."""
        code = textwrap.dedent("""\
            def process(items):
                result = []
                for item in items:
                    if item > 0:
                        result.append(item)
                return result
        """)
        stmts = get_top_level_statements(code)
        # Only 3 top-level: assignment, for_statement, return_statement
        assert len(stmts) == 3
        node_types = [s.node_type for s in stmts]
        assert "if_statement" not in node_types
        assert "expression_statement" in node_types   # result = []
        assert "for_statement" in node_types
        assert "return_statement" in node_types

    def test_deeply_nested_excluded(self):
        """Statements nested several levels deep are not at the top level."""
        code = textwrap.dedent("""\
            def f(data):
                for row in data:
                    for cell in row:
                        if cell:
                            process(cell)
                return True
        """)
        stmts = get_top_level_statements(code)
        assert len(stmts) == 2
        assert stmts[0].node_type == "for_statement"
        assert stmts[1].node_type == "return_statement"

    # ------------------------------------------------------------------
    # First function wins when there are multiple
    # ------------------------------------------------------------------

    def test_first_function_used_when_multiple(self):
        code = textwrap.dedent("""\
            def first():
                return 1

            def second():
                return 2
        """)
        stmts = get_top_level_statements(code)
        assert len(stmts) == 1
        assert stmts[0].source_text == "return 1"

    # ------------------------------------------------------------------
    # Public API: importable from parser package
    # ------------------------------------------------------------------

    def test_importable_from_package(self):
        from code_similarity_mcp.parser import get_top_level_statements as gts, StatementInfo as SI
        assert callable(gts)
        assert SI is StatementInfo


# ---------------------------------------------------------------------------
# build_dependency_graph
# ---------------------------------------------------------------------------

class TestBuildDependencyGraph:
    """Tests for build_dependency_graph.

    The function flattens ALL statements (top-level + nested) into a single
    list.  Compound statements (for/while/if/with/try) are processed
    header-only for data-flow; their body statements are separate flat nodes.

    Index mapping for compound-containing functions:
      - Simple statements keep their sequential flat index.
      - A compound header (e.g. for/while/if) at flat index i is immediately
        followed in the flat list by all its nested body statements, then by
        the next top-level statement.
    """

    # ------------------------------------------------------------------
    # Basic structure and edge cases
    # ------------------------------------------------------------------

    def test_no_function_returns_empty_graph(self):
        g = build_dependency_graph("x = 1\n")
        assert g.num_statements == 0
        assert g.data == {}
        assert g.control_flow == {}

    def test_empty_string_returns_empty_graph(self):
        g = build_dependency_graph("")
        assert g.num_statements == 0

    def test_returns_dependency_graph_type(self):
        code = "def f():\n    pass\n"
        g = build_dependency_graph(code)
        assert isinstance(g, DependencyGraph)

    def test_all_statement_indices_are_keys(self):
        code = textwrap.dedent("""\
            def f():
                a = 1
                b = 2
                c = 3
        """)
        g = build_dependency_graph(code)
        assert sorted(g.data.keys()) == [0, 1, 2]
        assert sorted(g.control_flow.keys()) == [0, 1, 2]

    def test_single_pass_statement_no_edges(self):
        code = "def f():\n    pass\n"
        g = build_dependency_graph(code)
        assert g.data == {0: []}
        assert g.control_flow == {0: []}

    # ------------------------------------------------------------------
    # Simple sequential assignments (no compound statements)
    # ------------------------------------------------------------------

    def test_simple_dependency(self):
        code = textwrap.dedent("""\
            def f():
                x = 1
                y = x + 1
        """)
        g = build_dependency_graph(code)
        assert 1 in g.data[0]

    def test_no_dependency_when_vars_independent(self):
        code = textwrap.dedent("""\
            def f():
                x = 1
                y = 2
        """)
        g = build_dependency_graph(code)
        assert g.data[0] == []
        assert g.data[1] == []

    def test_return_reads_variable(self):
        code = textwrap.dedent("""\
            def f():
                result = compute()
                return result
        """)
        g = build_dependency_graph(code)
        assert 1 in g.data[0]

    # ------------------------------------------------------------------
    # Multiple consumers and fanout
    # ------------------------------------------------------------------

    def test_multiple_consumers(self):
        code = textwrap.dedent("""\
            def f():
                x = compute()
                y = x + 1
                z = x + 2
        """)
        g = build_dependency_graph(code)
        assert 1 in g.data[0]
        assert 2 in g.data[0]

    def test_multiple_producers_one_consumer(self):
        code = textwrap.dedent("""\
            def f():
                a = 1
                b = 2
                c = a + b
        """)
        g = build_dependency_graph(code)
        assert 2 in g.data[0]
        assert 2 in g.data[1]

    # ------------------------------------------------------------------
    # Transitive / chain dependencies
    # ------------------------------------------------------------------

    def test_chain_no_skip_edges(self):
        """a → b → c: a consumed by b, b consumed by c, NOT a directly by c."""
        code = textwrap.dedent("""\
            def f():
                a = source()
                b = transform(a)
                c = finalize(b)
        """)
        g = build_dependency_graph(code)
        assert 1 in g.data[0]       # a → b
        assert 2 in g.data[1]       # b → c
        assert 2 not in g.data[0]   # no direct a → c edge

    def test_transitive_path_exists(self):
        """Transitive dependency is reachable via the graph."""
        code = textwrap.dedent("""\
            def f():
                a = 1
                b = a + 1
                c = b + 1
                d = c + 1
        """)
        g = build_dependency_graph(code)
        # Chain: 0→1→2→3
        assert 1 in g.data[0]
        assert 2 in g.data[1]
        assert 3 in g.data[2]
        # No skip edges
        assert 2 not in g.data[0]
        assert 3 not in g.data[0]
        assert 3 not in g.data[1]

    # ------------------------------------------------------------------
    # Augmented assignment (reads AND writes same variable)
    # ------------------------------------------------------------------

    def test_augmented_assignment_creates_edges(self):
        code = textwrap.dedent("""\
            def f():
                total = 0
                total += 1
                return total
        """)
        g = build_dependency_graph(code)
        assert 1 in g.data[0]  # total written at 0, read by augmented at 1
        assert 2 in g.data[1]  # total written by augmented at 1, read at 2

    # ------------------------------------------------------------------
    # For loop
    # Flat indices: 0=for_header, 1=pass (nested), 2=last=item (top-level)
    # ------------------------------------------------------------------

    def test_for_loop_writes_loop_var(self):
        # flat: 0=for(parent=None), 1=pass(parent=0), 2=last=item(parent=None)
        code = textwrap.dedent("""\
            def f(items):
                for item in items:
                    pass
                last = item
        """)
        g = build_dependency_graph(code)
        # item written by for header (0), read by assignment (2)
        assert 2 in g.data[0]

    def test_for_loop_body_writes_tracked(self):
        """Variables written inside the for body are separate flat nodes."""
        # flat: 0=total=0, 1=for_header(parent=None), 2=total+=x(parent=1), 3=return(parent=None)
        code = textwrap.dedent("""\
            def f(items):
                total = 0
                for x in items:
                    total += x
                return total
        """)
        g = build_dependency_graph(code)
        # total=0 (idx 0) → total+=x (idx 2): total is read at 2
        assert 2 in g.data[0]
        # for header (idx 1) writes x → total+=x (idx 2) reads x
        assert 2 in g.data[1]
        # total+=x (idx 2) writes total → return (idx 3) reads total
        assert 3 in g.data[2]

    # ------------------------------------------------------------------
    # If statement
    # ------------------------------------------------------------------

    def test_if_body_writes_propagate(self):
        # flat: 0=if_header(parent=None), 1=result=1(parent=0), 2=return(parent=None)
        code = textwrap.dedent("""\
            def f(cond):
                if cond:
                    result = 1
                return result
        """)
        g = build_dependency_graph(code)
        # result written in if body (idx 1), read by return (idx 2)
        assert 2 in g.data[1]

    def test_if_condition_reads_variable(self):
        # flat: 0=x=compute(), 1=if_header(parent=None), 2=pass(parent=1)
        code = textwrap.dedent("""\
            def f():
                x = compute()
                if x > 0:
                    pass
        """)
        g = build_dependency_graph(code)
        # x written at 0, read in if condition at 1 — indices unchanged
        assert 1 in g.data[0]

    def test_elif_else_writes_tracked(self):
        # flat: 0=if_header(None), 1=v=1(parent=0), 2=v=2(parent=0), 3=return(None)
        code = textwrap.dedent("""\
            def f(cond):
                if cond:
                    v = 1
                else:
                    v = 2
                return v
        """)
        g = build_dependency_graph(code)
        # v written in if branch (idx 1), read by return (idx 3)
        assert 3 in g.data[1]
        # v written in else branch (idx 2), read by return (idx 3)
        assert 3 in g.data[2]

    # ------------------------------------------------------------------
    # While loop
    # ------------------------------------------------------------------

    def test_while_loop_condition_and_body(self):
        # flat: 0=n=10, 1=while_header(None), 2=n-=1(parent=1), 3=return(None)
        code = textwrap.dedent("""\
            def f():
                n = 10
                while n > 0:
                    n -= 1
                return n
        """)
        g = build_dependency_graph(code)
        # n written at 0, read in while condition at 1
        assert 1 in g.data[0]
        # n written by n-=1 (idx 2), read by return (idx 3)
        assert 3 in g.data[2]

    # ------------------------------------------------------------------
    # Import statements
    # ------------------------------------------------------------------

    def test_import_writes_name(self):
        code = textwrap.dedent("""\
            def f():
                import re
                result = re.match(r'\\d+', text)
                return result
        """)
        g = build_dependency_graph(code)
        assert 1 in g.data[0]  # re written by import (0), read at stmt 1
        assert 2 in g.data[1]  # result written at 1, read at return 2

    def test_from_import_writes_name(self):
        code = textwrap.dedent("""\
            def f():
                from os import path
                full = path.join("a", "b")
                return full
        """)
        g = build_dependency_graph(code)
        assert 1 in g.data[0]  # path written by import (0), read at stmt 1
        assert 2 in g.data[1]

    # ------------------------------------------------------------------
    # Try/except
    # flat: 0=try_header(None), 1=result=compute()(parent=0),
    #       2=result=None(parent=0), 3=return(None)
    # ------------------------------------------------------------------

    def test_try_body_writes_tracked(self):
        code = textwrap.dedent("""\
            def f():
                try:
                    result = compute()
                except Exception:
                    result = None
                return result
        """)
        g = build_dependency_graph(code)
        # result written in try body (idx 1), read by return (idx 3)
        assert 3 in g.data[1]
        # result written in except body (idx 2), read by return (idx 3)
        assert 3 in g.data[2]

    # ------------------------------------------------------------------
    # Tuple unpacking
    # ------------------------------------------------------------------

    def test_tuple_unpack_writes_both(self):
        code = textwrap.dedent("""\
            def f():
                a, b = get_pair()
                c = a + b
        """)
        g = build_dependency_graph(code)
        assert 1 in g.data[0]  # a and b written at 0, both read at 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def test_importable_from_package(self):
        from code_similarity_mcp.parser import build_dependency_graph as bdg
        assert callable(bdg)

    def test_dependency_graph_importable_from_package(self):
        from code_similarity_mcp.parser import DependencyGraph as DG
        assert DG is DependencyGraph


# ---------------------------------------------------------------------------
# Control-flow edges in build_dependency_graph
# ---------------------------------------------------------------------------

class TestControlFlowEdges:
    """Tests that build_dependency_graph correctly populates control_flow edges.

    Control-flow edges come in two flavours:
    1. Header → body-child: each direct body statement depends on its
       enclosing compound header.
    2. Header → fall-through: the first statement at the same nesting level
       that follows the compound statement.
    """

    # ------------------------------------------------------------------
    # For loop
    # ------------------------------------------------------------------

    def test_for_body_stmts_depend_on_header(self):
        # flat: 0=for_header, 1=y=i (body), 2=z=y+1 (body), 3=return (top)
        code = textwrap.dedent("""\
            def f(items):
                for i in items:
                    y = i
                    z = y + 1
                return z
        """)
        g = build_dependency_graph(code)
        # Both body statements (1 and 2) must have a CF edge from header (0)
        assert 1 in g.control_flow[0]
        assert 2 in g.control_flow[0]

    def test_for_fall_through_to_next_top_level(self):
        # flat: 0=for_header(None), 1=pass(parent=0), 2=result(None)
        code = textwrap.dedent("""\
            def f(items):
                for item in items:
                    pass
                result = done()
        """)
        g = build_dependency_graph(code)
        # Fall-through: for (0) → result (2)
        assert 2 in g.control_flow[0]

    def test_for_body_not_cf_edge_for_simple_stmts(self):
        """Simple (non-compound) statements produce no CF edges."""
        code = textwrap.dedent("""\
            def f():
                x = 1
                y = 2
        """)
        g = build_dependency_graph(code)
        assert g.control_flow[0] == []
        assert g.control_flow[1] == []

    # ------------------------------------------------------------------
    # While loop
    # ------------------------------------------------------------------

    def test_while_body_stmts_depend_on_header(self):
        # flat: 0=n=10, 1=while_header, 2=n-=1 (body), 3=return (top)
        code = textwrap.dedent("""\
            def f():
                n = 10
                while n > 0:
                    n -= 1
                return n
        """)
        g = build_dependency_graph(code)
        # Body statement (2) has CF edge from while header (1)
        assert 2 in g.control_flow[1]

    def test_while_fall_through(self):
        code = textwrap.dedent("""\
            def f():
                n = 10
                while n > 0:
                    n -= 1
                return n
        """)
        g = build_dependency_graph(code)
        # Fall-through: while (1) → return (3)
        assert 3 in g.control_flow[1]

    # ------------------------------------------------------------------
    # If / elif / else
    # ------------------------------------------------------------------

    def test_if_body_stmt_depends_on_header(self):
        # flat: 0=if_header(None), 1=result=1(parent=0), 2=return(None)
        code = textwrap.dedent("""\
            def f(cond):
                if cond:
                    result = 1
                return result
        """)
        g = build_dependency_graph(code)
        # Body statement (1) has CF edge from if header (0)
        assert 1 in g.control_flow[0]

    def test_if_else_both_branches_depend_on_header(self):
        # flat: 0=if_header(None), 1=v=1(parent=0), 2=v=2(parent=0), 3=return(None)
        code = textwrap.dedent("""\
            def f(cond):
                if cond:
                    v = 1
                else:
                    v = 2
                return v
        """)
        g = build_dependency_graph(code)
        # Both branches (1 and 2) depend on the if header (0)
        assert 1 in g.control_flow[0]
        assert 2 in g.control_flow[0]

    def test_if_fall_through(self):
        code = textwrap.dedent("""\
            def f(cond):
                if cond:
                    v = 1
                else:
                    v = 2
                return v
        """)
        g = build_dependency_graph(code)
        # Fall-through: if (0) → return (3)
        assert 3 in g.control_flow[0]

    # ------------------------------------------------------------------
    # Nested compound statements
    # ------------------------------------------------------------------

    def test_nested_if_inside_for(self):
        # flat: 0=for(None), 1=if(parent=0), 2=y=1(parent=1), 3=return(None)
        code = textwrap.dedent("""\
            def f(items):
                for x in items:
                    if x > 0:
                        y = x
                return y
        """)
        g = build_dependency_graph(code)
        # Outer for (0): body child is if (1)
        assert 1 in g.control_flow[0]
        # Fall-through of for (0) → return (3)
        assert 3 in g.control_flow[0]
        # Inner if (1): body child is y=x (2)
        assert 2 in g.control_flow[1]

    # ------------------------------------------------------------------
    # Control-flow edges are distinct from data edges
    # ------------------------------------------------------------------

    def test_cf_and_data_edges_are_independent(self):
        """A pair of statements can have a CF edge without a data edge."""
        # flat: 0=for(None), 1=pass(parent=0), 2=x=1(None)
        # CF: 0→1 (header→body), 0→2 (fall-through)
        # Data: no shared variables between for header and x=1
        code = textwrap.dedent("""\
            def f(items):
                for item in items:
                    pass
                x = 1
        """)
        g = build_dependency_graph(code)
        # CF edge exists (fall-through)
        assert 2 in g.control_flow[0]
        # No data edge from for (0) to x=1 (2): x is unrelated to item/items
        assert 2 not in g.data[0]

    def test_num_statements_includes_nested(self):
        """num_statements counts all flat nodes including nested body stmts."""
        # flat: 0=for(None), 1=y=i(parent=0), 2=z=y(parent=0), 3=return(None)
        code = textwrap.dedent("""\
            def f(items):
                for i in items:
                    y = i
                    z = y
                return z
        """)
        g = build_dependency_graph(code)
        assert g.num_statements == 4  # for + 2 body stmts + return

    def test_all_indices_present_in_both_dicts(self):
        """Every flat index 0..n-1 appears as a key in both data and control_flow."""
        code = textwrap.dedent("""\
            def f(items):
                for i in items:
                    y = i
                return y
        """)
        g = build_dependency_graph(code)
        expected = list(range(g.num_statements))
        assert sorted(g.data.keys()) == expected
        assert sorted(g.control_flow.keys()) == expected


# ---------------------------------------------------------------------------
# group_into_chunks
# ---------------------------------------------------------------------------

class TestGroupIntoChunks:
    """Tests for group_into_chunks.

    The greedy algorithm:
    - Starts a new chunk when the current statement has no intra-function
      data-flow providers (a "fresh start" reading only params / external
      names), provided the current chunk is already non-empty.
    - Also starts a new chunk when the current statement depends on a
      variable written in an already-closed chunk (unresolved cross-chunk
      dependency).
    - Otherwise, extends the current chunk.
    """

    # ------------------------------------------------------------------
    # Edge cases / basic structure
    # ------------------------------------------------------------------

    def test_empty_graph_returns_empty_list(self):
        g = DependencyGraph(data={}, control_flow={}, num_statements=0)
        assert group_into_chunks(g) == []

    def test_single_statement_one_chunk(self):
        g = DependencyGraph(data={0: []}, control_flow={0: []}, num_statements=1)
        assert group_into_chunks(g) == [[0]]

    def test_returns_list_of_lists(self):
        g = DependencyGraph(data={0: [1], 1: []}, control_flow={0: [], 1: []},
                            num_statements=2)
        chunks = group_into_chunks(g)
        assert isinstance(chunks, list)
        assert all(isinstance(c, list) for c in chunks)

    # ------------------------------------------------------------------
    # Partition correctness
    # ------------------------------------------------------------------

    def test_chunks_cover_all_statements(self):
        """Union of all chunks is exactly range(num_statements)."""
        code = textwrap.dedent("""\
            def f(x, y):
                a = x + 1
                b = a * 2
                c = y + 1
                d = c * 2
        """)
        g = build_dependency_graph(code)
        chunks = group_into_chunks(g)
        all_stmts = [s for chunk in chunks for s in chunk]
        assert sorted(all_stmts) == list(range(g.num_statements))

    def test_chunks_are_non_overlapping(self):
        """No statement index appears in more than one chunk."""
        code = textwrap.dedent("""\
            def f(x, y):
                a = x + 1
                b = a * 2
                c = y + 1
                d = c * 2
        """)
        g = build_dependency_graph(code)
        chunks = group_into_chunks(g)
        all_stmts = [s for chunk in chunks for s in chunk]
        assert len(all_stmts) == len(set(all_stmts))

    def test_each_chunk_is_non_empty(self):
        code = textwrap.dedent("""\
            def f(x, y):
                a = x + 1
                b = a * 2
                c = y + 1
                d = c * 2
        """)
        g = build_dependency_graph(code)
        chunks = group_into_chunks(g)
        assert all(len(c) > 0 for c in chunks)

    def test_chunks_are_consecutive_ascending(self):
        """Each chunk's indices are consecutive and ascending."""
        code = textwrap.dedent("""\
            def f(x, y):
                a = x + 1
                b = a * 2
                c = y + 1
                d = c * 2
        """)
        g = build_dependency_graph(code)
        chunks = group_into_chunks(g)
        for chunk in chunks:
            assert chunk == list(range(chunk[0], chunk[-1] + 1))

    # ------------------------------------------------------------------
    # Sequential chain → one chunk
    # ------------------------------------------------------------------

    def test_sequential_chain_produces_one_chunk(self):
        """a = f(); b = a+1; c = b+1 — fully connected chain → one chunk."""
        # data[0]=[1], data[1]=[2], data[2]=[] — each stmt depends on predecessor
        g = DependencyGraph(
            data={0: [1], 1: [2], 2: []},
            control_flow={0: [], 1: [], 2: []},
            num_statements=3,
        )
        chunks = group_into_chunks(g)
        assert chunks == [[0, 1, 2]]

    def test_two_stmts_with_dependency_one_chunk(self):
        """x = 1; y = x+1 → one chunk (y depends on x)."""
        code = textwrap.dedent("""\
            def f():
                x = 1
                y = x + 1
        """)
        g = build_dependency_graph(code)
        chunks = group_into_chunks(g)
        assert len(chunks) == 1
        assert chunks[0] == [0, 1]

    def test_long_chain_one_chunk(self):
        code = textwrap.dedent("""\
            def f(x):
                a = x + 1
                b = a + 1
                c = b + 1
                d = c + 1
        """)
        g = build_dependency_graph(code)
        chunks = group_into_chunks(g)
        assert len(chunks) == 1

    # ------------------------------------------------------------------
    # Independent computations → multiple chunks
    # ------------------------------------------------------------------

    def test_two_independent_chains_two_chunks(self):
        """Two param-fed chains produce two separate chunks."""
        # stmt 0: reads from params only (providers={})
        # stmt 1: reads from stmt 0 (providers={0})
        # stmt 2: reads from params only → fresh start → new chunk
        # stmt 3: reads from stmt 2 (providers={2})
        g = DependencyGraph(
            data={0: [1], 1: [], 2: [3], 3: []},
            control_flow={0: [], 1: [], 2: [], 3: []},
            num_statements=4,
        )
        # Use max=3 (< 4 stmts) so the single-chunk short-circuit doesn't fire.
        chunks = group_into_chunks(g, max_statements_per_chunk=3)
        assert chunks == [[0, 1], [2, 3]]

    def test_real_code_two_independent_chains(self):
        code = textwrap.dedent("""\
            def process(x, y):
                a = x + 1
                b = a * 2
                c = y + 1
                d = c * 2
        """)
        g = build_dependency_graph(code)
        # 4 statements — use max=3 so fresh-start splitting is exercised.
        chunks = group_into_chunks(g, max_statements_per_chunk=3)
        assert len(chunks) == 2

    def test_all_statements_independent_each_own_chunk(self):
        """No data-flow edges at all → every statement is its own chunk when max < num."""
        g = DependencyGraph(
            data={0: [], 1: [], 2: []},
            control_flow={0: [], 1: [], 2: []},
            num_statements=3,
        )
        # Use max=2 (< 3 stmts) so the single-chunk short-circuit doesn't fire.
        chunks = group_into_chunks(g, max_statements_per_chunk=2)
        assert chunks == [[0], [1], [2]]

    def test_three_independent_param_reads_three_chunks(self):
        code = textwrap.dedent("""\
            def f(x, y, z):
                a = x + 1
                b = y + 1
                c = z + 1
        """)
        g = build_dependency_graph(code)
        # 3 statements — use max=2 so fresh-start splitting is exercised.
        chunks = group_into_chunks(g, max_statements_per_chunk=2)
        assert len(chunks) == 3

    # ------------------------------------------------------------------
    # Step 4: verify no unresolved inbound deps for clean-split functions
    # ------------------------------------------------------------------

    def _check_no_inbound_deps(self, graph: DependencyGraph,
                                chunks: list[list[int]]) -> None:
        """Assert that no chunk has a data-flow dependency on a prior chunk."""
        prior: set[int] = set()
        for chunk in chunks:
            chunk_set = set(chunk)
            for stmt in chunk:
                for producer, consumers in graph.data.items():
                    if stmt in consumers and producer in prior:
                        raise AssertionError(
                            f"Stmt {stmt} in chunk {chunk} has inbound dep "
                            f"from prior stmt {producer}"
                        )
            prior.update(chunk_set)

    def test_step4_single_chain_no_inbound_deps(self):
        code = textwrap.dedent("""\
            def f(x):
                a = x + 1
                b = a * 2
                c = b - 1
        """)
        g = build_dependency_graph(code)
        chunks = group_into_chunks(g)
        self._check_no_inbound_deps(g, chunks)

    def test_step4_two_independent_chains_no_inbound_deps(self):
        code = textwrap.dedent("""\
            def process(x, y):
                a = x + 1
                b = a * 2
                c = y + 1
                d = c * 2
        """)
        g = build_dependency_graph(code)
        chunks = group_into_chunks(g)
        self._check_no_inbound_deps(g, chunks)

    def test_step4_three_independent_chains_no_inbound_deps(self):
        code = textwrap.dedent("""\
            def f(x, y, z):
                a = x * 2
                b = a + 1
                c = y * 2
                d = c + 1
                e = z * 2
                f = e + 1
        """)
        g = build_dependency_graph(code)
        # 6 statements — use max=5 so fresh-start splitting is exercised.
        chunks = group_into_chunks(g, max_statements_per_chunk=5)
        self._check_no_inbound_deps(g, chunks)
        assert len(chunks) == 3

    def test_step4_no_data_flow_no_inbound_deps(self):
        """All-independent statements produce solo chunks, all clean."""
        g = DependencyGraph(
            data={0: [], 1: [], 2: [], 3: []},
            control_flow={0: [], 1: [], 2: [], 3: []},
            num_statements=4,
        )
        chunks = group_into_chunks(g)
        self._check_no_inbound_deps(g, chunks)

    # ------------------------------------------------------------------
    # Forced split on closed-chunk dependency
    # ------------------------------------------------------------------

    def test_forced_split_on_closed_chunk_dependency(self):
        """When stmt i reads from an already-closed chunk, a new chunk starts."""
        # Chunks: [[0, 1], [2], ...] and stmt 3 depends on stmt 0 (closed).
        # data: 0→[1], 2→[], 0→[3] as well.  Build that graph manually.
        g = DependencyGraph(
            data={0: [1, 3], 1: [], 2: [], 3: []},
            control_flow={0: [], 1: [], 2: [], 3: []},
            num_statements=4,
        )
        # Expected walk with max=3 (< 4 stmts, so no single-chunk short-circuit):
        # i=0: providers={}, current empty → [0]
        # i=1: providers={0}, extend → [0, 1]
        # i=2: providers={}, fresh start → split → [[0,1]], current=[2]
        # i=3: providers={0}, 0 is closed → split → [[0,1],[2]], current=[3]
        chunks = group_into_chunks(g, max_statements_per_chunk=3)
        assert [0, 1] in chunks
        assert [2] in chunks
        assert [3] in chunks

    # ------------------------------------------------------------------
    # Real function with compound statements
    # ------------------------------------------------------------------

    def test_for_loop_function_single_chunk(self):
        """A for loop and its body share data flow — stay in one chunk."""
        code = textwrap.dedent("""\
            def total(items):
                result = 0
                for item in items:
                    result += item
                return result
        """)
        g = build_dependency_graph(code)
        chunks = group_into_chunks(g)
        # All statements connected via result / item → one chunk
        all_stmts = [s for c in chunks for s in c]
        assert sorted(all_stmts) == list(range(g.num_statements))

    def test_first_stmt_always_starts_first_chunk(self):
        """Statement 0 is always the first element of the first chunk."""
        for code in [
            "def f(x):\n    a = x + 1\n",
            "def f():\n    pass\n",
            "def f(x, y):\n    a = x\n    b = y\n",
        ]:
            g = build_dependency_graph(code)
            chunks = group_into_chunks(g)
            assert chunks[0][0] == 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def test_importable_from_package(self):
        from code_similarity_mcp.parser import group_into_chunks as gic
        assert callable(gic)

    def test_importable_from_base(self):
        from code_similarity_mcp.parser.base import group_into_chunks as gic
        assert callable(gic)

    # ------------------------------------------------------------------
    # max_statements_per_chunk (feature #22)
    # ------------------------------------------------------------------

    def test_default_max_is_ten(self):
        """A chain of exactly 10 statements stays in one chunk by default."""
        # 10-statement chain: 0→1→2→…→9
        data = {i: [i + 1] for i in range(9)}
        data[9] = []
        g = DependencyGraph(data=data, control_flow={i: [] for i in range(10)},
                            num_statements=10)
        chunks = group_into_chunks(g)
        assert chunks == [list(range(10))]

    def test_default_max_splits_eleven_statement_chain(self):
        """A chain of 11 interdependent statements must be split (default max=10)."""
        data = {i: [i + 1] for i in range(10)}
        data[10] = []
        g = DependencyGraph(data=data, control_flow={i: [] for i in range(11)},
                            num_statements=11)
        chunks = group_into_chunks(g)
        assert all(len(c) <= 10 for c in chunks)
        # Verify full partition
        all_stmts = [s for c in chunks for s in c]
        assert sorted(all_stmts) == list(range(11))

    def test_no_chunk_exceeds_max(self):
        """No output chunk ever exceeds max_statements_per_chunk."""
        # Long chain of 20 statements, all interdependent
        data = {i: [i + 1] for i in range(19)}
        data[19] = []
        g = DependencyGraph(data=data, control_flow={i: [] for i in range(20)},
                            num_statements=20)
        for max_size in [1, 2, 3, 5, 7, 10, 15, 20]:
            chunks = group_into_chunks(g, max_statements_per_chunk=max_size)
            assert all(len(c) <= max_size for c in chunks), (
                f"max_size={max_size}: chunk sizes {[len(c) for c in chunks]}"
            )

    def test_max_one_each_stmt_own_chunk(self):
        """max_statements_per_chunk=1 forces every statement into its own chunk."""
        data = {0: [1], 1: [2], 2: []}
        g = DependencyGraph(data=data, control_flow={0: [], 1: [], 2: []},
                            num_statements=3)
        chunks = group_into_chunks(g, max_statements_per_chunk=1)
        assert chunks == [[0], [1], [2]]

    def test_max_two_splits_three_stmt_chain(self):
        """Chain of 3 with max=2 → [[0,1],[2]]."""
        data = {0: [1], 1: [2], 2: []}
        g = DependencyGraph(data=data, control_flow={0: [], 1: [], 2: []},
                            num_statements=3)
        chunks = group_into_chunks(g, max_statements_per_chunk=2)
        assert chunks == [[0, 1], [2]]

    def test_max_larger_than_stmts_no_effect(self):
        """max bigger than total statements → behaves as if no cap."""
        data = {0: [1], 1: [2], 2: []}
        g = DependencyGraph(data=data, control_flow={0: [], 1: [], 2: []},
                            num_statements=3)
        chunks_uncapped = group_into_chunks(g, max_statements_per_chunk=100)
        chunks_default = group_into_chunks(g)
        assert chunks_uncapped == chunks_default

    def test_max_partition_is_complete(self):
        """With custom max, chunks still form a complete partition."""
        data = {i: [i + 1] for i in range(14)}
        data[14] = []
        g = DependencyGraph(data=data, control_flow={i: [] for i in range(15)},
                            num_statements=15)
        chunks = group_into_chunks(g, max_statements_per_chunk=4)
        all_stmts = [s for c in chunks for s in c]
        assert sorted(all_stmts) == list(range(15))

    def test_max_with_real_code(self):
        """Real Python function: max=2 forces split inside dependent chain."""
        code = textwrap.dedent("""\
            def f(x):
                a = x + 1
                b = a * 2
                c = b - 1
                d = c + 5
        """)
        g = build_dependency_graph(code)
        chunks = group_into_chunks(g, max_statements_per_chunk=2)
        assert all(len(c) <= 2 for c in chunks)
        all_stmts = [s for c in chunks for s in c]
        assert sorted(all_stmts) == list(range(g.num_statements))

    def test_max_empty_graph_still_empty(self):
        """max_statements_per_chunk has no effect on an empty graph."""
        g = DependencyGraph(data={}, control_flow={}, num_statements=0)
        assert group_into_chunks(g, max_statements_per_chunk=1) == []

    def test_max_single_stmt_still_one_chunk(self):
        """A single statement always fits in one chunk regardless of max."""
        g = DependencyGraph(data={0: []}, control_flow={0: []}, num_statements=1)
        assert group_into_chunks(g, max_statements_per_chunk=1) == [[0]]

    # ------------------------------------------------------------------
    # Feature #38: function shorter than max_chunk_size → single chunk
    # ------------------------------------------------------------------

    def test_max_larger_than_count_independent_stmts_single_chunk(self):
        """35 fully independent statements with max=50 → one chunk covering all 35."""
        n = 35
        # All statements are independent (no intra-function providers).
        data = {i: [] for i in range(n)}
        g = DependencyGraph(
            data=data,
            control_flow={i: [] for i in range(n)},
            num_statements=n,
        )
        chunks = group_into_chunks(g, max_statements_per_chunk=50)
        assert len(chunks) == 1
        assert chunks[0] == list(range(n))

    def test_max_equal_to_count_independent_stmts_single_chunk(self):
        """35 independent statements with max=35 (exact) → one chunk."""
        n = 35
        data = {i: [] for i in range(n)}
        g = DependencyGraph(
            data=data,
            control_flow={i: [] for i in range(n)},
            num_statements=n,
        )
        chunks = group_into_chunks(g, max_statements_per_chunk=35)
        assert len(chunks) == 1
        assert chunks[0] == list(range(n))

    def test_max_one_less_than_count_independent_stmts_splits(self):
        """35 independent statements with max=34 → must split into multiple chunks."""
        n = 35
        data = {i: [] for i in range(n)}
        g = DependencyGraph(
            data=data,
            control_flow={i: [] for i in range(n)},
            num_statements=n,
        )
        chunks = group_into_chunks(g, max_statements_per_chunk=34)
        # Should have more than one chunk since max < total
        assert len(chunks) > 1
        # Full partition still holds
        all_stmts = [s for c in chunks for s in c]
        assert sorted(all_stmts) == list(range(n))

    def test_max_larger_than_count_single_chunk_annotated_no_cross_deps(self):
        """35 statements with max=50 → single ChunkInfo with empty dependency lists."""
        n = 35
        data = {i: [] for i in range(n)}
        g = DependencyGraph(
            data=data,
            control_flow={i: [] for i in range(n)},
            num_statements=n,
        )
        chunks = group_into_chunks(g, max_statements_per_chunk=50)
        result = annotate_chunks(chunks, g, function_name="big_func", file_path="x.py")
        assert len(result) == 1
        ci = result[0]
        assert ci.depends_on_chunks == []
        assert ci.depended_on_by_chunks == []
        assert ci.statement_start == 0
        assert ci.statement_end == n - 1
        assert ci.statement_indices == list(range(n))

    def test_real_35_stmt_function_max_50_one_chunk(self):
        """Integration: real Python code with 35 statements, max=50 → one chunk."""
        # Build a function with 35 independent assignments
        lines = ["def big_func(a, b, c, d, e):"]
        for i in range(35):
            lines.append(f"    v{i} = {i} + 1")
        code = "\n".join(lines)
        g = build_dependency_graph(code)
        assert g.num_statements == 35
        chunks = group_into_chunks(g, max_statements_per_chunk=50)
        assert len(chunks) == 1
        assert chunks[0] == list(range(35))

    def test_real_35_stmt_function_max_50_annotated_no_cross_deps(self):
        """Integration: annotated single chunk has no cross-chunk dependency links."""
        lines = ["def big_func(a, b, c, d, e):"]
        for i in range(35):
            lines.append(f"    v{i} = {i} + 1")
        code = "\n".join(lines)
        g = build_dependency_graph(code)
        chunks = group_into_chunks(g, max_statements_per_chunk=50)
        result = annotate_chunks(chunks, g, function_name="big_func", file_path="x.py")
        assert len(result) == 1
        ci = result[0]
        assert ci.depends_on_chunks == []
        assert ci.depended_on_by_chunks == []


# ---------------------------------------------------------------------------
# annotate_chunks (feature #23)
# ---------------------------------------------------------------------------


class TestAnnotateChunks:
    """Tests for annotate_chunks — metadata annotation for statement chunks."""

    # ------------------------------------------------------------------
    # Basic structural tests
    # ------------------------------------------------------------------

    def test_empty_chunks_returns_empty_list(self):
        """annotate_chunks([]) → []."""
        g = DependencyGraph(data={}, control_flow={}, num_statements=0)
        result = annotate_chunks([], g)
        assert result == []

    def test_single_chunk_single_stmt(self):
        """One chunk with one statement gets correct index bounds."""
        g = DependencyGraph(data={0: []}, control_flow={0: []}, num_statements=1)
        chunks = [[0]]
        result = annotate_chunks(chunks, g)
        assert len(result) == 1
        ci = result[0]
        assert isinstance(ci, ChunkInfo)
        assert ci.chunk_index == 0
        assert ci.statement_start == 0
        assert ci.statement_end == 0
        assert ci.statement_indices == [0]

    def test_single_chunk_multiple_stmts(self):
        """One chunk with 3 chained statements."""
        g = DependencyGraph(
            data={0: [1], 1: [2], 2: []},
            control_flow={0: [], 1: [], 2: []},
            num_statements=3,
        )
        chunks = [[0, 1, 2]]
        result = annotate_chunks(chunks, g)
        assert len(result) == 1
        ci = result[0]
        assert ci.statement_start == 0
        assert ci.statement_end == 2
        assert ci.statement_indices == [0, 1, 2]

    def test_chunk_indices_are_sequential(self):
        """chunk_index must equal the position of the ChunkInfo in the list."""
        g = DependencyGraph(
            data={0: [], 1: [], 2: [], 3: []},
            control_flow={0: [], 1: [], 2: [], 3: []},
            num_statements=4,
        )
        chunks = [[0], [1], [2], [3]]
        result = annotate_chunks(chunks, g)
        for i, ci in enumerate(result):
            assert ci.chunk_index == i

    # ------------------------------------------------------------------
    # statement_start / statement_end / statement_indices
    # ------------------------------------------------------------------

    def test_statement_start_end_match_chunk_bounds(self):
        """statement_start == min(chunk), statement_end == max(chunk)."""
        g = DependencyGraph(
            data={0: [1, 2], 1: [], 2: []},
            control_flow={0: [], 1: [], 2: []},
            num_statements=3,
        )
        # Two chunks: [0, 1] and [2]
        chunks = [[0, 1], [2]]
        result = annotate_chunks(chunks, g)
        assert result[0].statement_start == 0
        assert result[0].statement_end == 1
        assert result[1].statement_start == 2
        assert result[1].statement_end == 2

    def test_statement_indices_preserved(self):
        """statement_indices must be the original chunk list contents."""
        g = DependencyGraph(
            data={0: [], 1: [], 2: [], 3: []},
            control_flow={0: [], 1: [], 2: [], 3: []},
            num_statements=4,
        )
        chunks = [[0, 1], [2, 3]]
        result = annotate_chunks(chunks, g)
        assert result[0].statement_indices == [0, 1]
        assert result[1].statement_indices == [2, 3]

    # ------------------------------------------------------------------
    # Metadata fields
    # ------------------------------------------------------------------

    def test_default_metadata_empty(self):
        """When metadata is omitted, defaults are empty string / None."""
        g = DependencyGraph(data={0: []}, control_flow={0: []}, num_statements=1)
        result = annotate_chunks([[0]], g)
        ci = result[0]
        assert ci.function_name == ""
        assert ci.file_path == ""
        assert ci.function_id is None

    def test_metadata_propagated_to_all_chunks(self):
        """function_name, file_path, function_id appear on every ChunkInfo."""
        g = DependencyGraph(
            data={0: [], 1: [], 2: []},
            control_flow={0: [], 1: [], 2: []},
            num_statements=3,
        )
        chunks = [[0], [1], [2]]
        result = annotate_chunks(
            chunks, g,
            function_name="my_func",
            file_path="/src/module.py",
            function_id=42,
        )
        for ci in result:
            assert ci.function_name == "my_func"
            assert ci.file_path == "/src/module.py"
            assert ci.function_id == 42

    def test_function_id_zero_is_stored(self):
        """function_id=0 must be stored (falsy but valid)."""
        g = DependencyGraph(data={0: []}, control_flow={0: []}, num_statements=1)
        result = annotate_chunks([[0]], g, function_id=0)
        assert result[0].function_id == 0

    # ------------------------------------------------------------------
    # Cross-chunk dependency links
    # ------------------------------------------------------------------

    def test_no_cross_chunk_deps_for_independent_stmts(self):
        """Completely independent statements → no cross-chunk deps."""
        g = DependencyGraph(
            data={0: [], 1: [], 2: []},
            control_flow={0: [], 1: [], 2: []},
            num_statements=3,
        )
        chunks = [[0], [1], [2]]
        result = annotate_chunks(chunks, g)
        for ci in result:
            assert ci.depends_on_chunks == []
            assert ci.depended_on_by_chunks == []

    def test_no_cross_chunk_deps_single_chain_single_chunk(self):
        """A single-chunk function has no cross-chunk deps by definition."""
        g = DependencyGraph(
            data={0: [1], 1: [2], 2: []},
            control_flow={0: [], 1: [], 2: []},
            num_statements=3,
        )
        chunks = [[0, 1, 2]]
        result = annotate_chunks(chunks, g)
        assert result[0].depends_on_chunks == []
        assert result[0].depended_on_by_chunks == []

    def test_cross_chunk_dep_simple(self):
        """Chunk 1 reads from stmt 0 (chunk 0) → cross-chunk dep 0→1."""
        # stmt 0 writes something read by stmt 2; chunks: [[0,1],[2]]
        g = DependencyGraph(
            data={0: [2], 1: [], 2: []},
            control_flow={0: [], 1: [], 2: []},
            num_statements=3,
        )
        chunks = [[0, 1], [2]]
        result = annotate_chunks(chunks, g)
        # Chunk 1 (stmt 2) reads from chunk 0 (stmt 0)
        assert result[1].depends_on_chunks == [0]
        assert result[0].depended_on_by_chunks == [1]
        # Chunk 0 has no external deps; chunk 1 is not depended on
        assert result[0].depends_on_chunks == []
        assert result[1].depended_on_by_chunks == []

    def test_cross_chunk_dep_multiple_providers(self):
        """Chunk 2 reads from both chunk 0 and chunk 1."""
        # chunks: [[0],[1],[2]]; graph.data[0]=[2], graph.data[1]=[2]
        g = DependencyGraph(
            data={0: [2], 1: [2], 2: []},
            control_flow={0: [], 1: [], 2: []},
            num_statements=3,
        )
        chunks = [[0], [1], [2]]
        result = annotate_chunks(chunks, g)
        assert sorted(result[2].depends_on_chunks) == [0, 1]
        assert result[0].depended_on_by_chunks == [2]
        assert result[1].depended_on_by_chunks == [2]

    def test_cross_chunk_dep_chain(self):
        """Chunk 0→chunk 1→chunk 2 dep chain."""
        g = DependencyGraph(
            data={0: [1], 1: [2], 2: []},
            control_flow={0: [], 1: [], 2: []},
            num_statements=3,
        )
        chunks = [[0], [1], [2]]
        result = annotate_chunks(chunks, g)
        assert result[0].depends_on_chunks == []
        assert result[0].depended_on_by_chunks == [1]
        assert result[1].depends_on_chunks == [0]
        assert result[1].depended_on_by_chunks == [2]
        assert result[2].depends_on_chunks == [1]
        assert result[2].depended_on_by_chunks == []

    def test_depends_on_chunks_sorted(self):
        """depends_on_chunks and depended_on_by_chunks are always sorted."""
        # Chunk 3 reads from chunks 0, 1, 2
        g = DependencyGraph(
            data={0: [3], 1: [3], 2: [3], 3: []},
            control_flow={0: [], 1: [], 2: [], 3: []},
            num_statements=4,
        )
        chunks = [[0], [1], [2], [3]]
        result = annotate_chunks(chunks, g)
        assert result[3].depends_on_chunks == [0, 1, 2]

    # ------------------------------------------------------------------
    # Partition validation
    # ------------------------------------------------------------------

    def test_valid_partition_does_not_raise(self):
        """A correctly partitioned chunk list must not raise."""
        g = DependencyGraph(
            data={0: [], 1: [], 2: []},
            control_flow={0: [], 1: [], 2: []},
            num_statements=3,
        )
        annotate_chunks([[0], [1], [2]], g)  # should not raise

    def test_invalid_partition_gap_raises(self):
        """Missing statement index in chunks raises ValueError."""
        g = DependencyGraph(
            data={0: [], 1: [], 2: []},
            control_flow={0: [], 1: [], 2: []},
            num_statements=3,
        )
        with pytest.raises(ValueError):
            annotate_chunks([[0], [2]], g)  # stmt 1 is missing

    def test_invalid_partition_duplicate_raises(self):
        """Duplicate statement index in chunks raises ValueError."""
        g = DependencyGraph(
            data={0: [], 1: [], 2: []},
            control_flow={0: [], 1: [], 2: []},
            num_statements=3,
        )
        with pytest.raises(ValueError):
            annotate_chunks([[0, 1], [1, 2]], g)  # stmt 1 appears twice

    def test_all_stmts_covered_by_group_then_annotate(self):
        """group_into_chunks output always forms a valid partition for annotate_chunks."""
        g = DependencyGraph(
            data={0: [1], 1: [2], 2: []},
            control_flow={0: [], 1: [], 2: []},
            num_statements=3,
        )
        chunks = group_into_chunks(g)
        result = annotate_chunks(chunks, g)
        all_stmts = [s for ci in result for s in ci.statement_indices]
        assert sorted(all_stmts) == list(range(g.num_statements))

    # ------------------------------------------------------------------
    # Integration with build_dependency_graph + group_into_chunks
    # ------------------------------------------------------------------

    def test_integration_two_independent_chains(self):
        """Two independent chains produce two chunks; no cross-chunk deps."""
        code = textwrap.dedent("""\
            def process(x, y):
                a = x + 1
                b = a * 2
                c = y + 1
                d = c * 2
        """)
        g = build_dependency_graph(code)
        # 4 statements — use max=3 so fresh-start splitting is exercised.
        chunks = group_into_chunks(g, max_statements_per_chunk=3)
        result = annotate_chunks(chunks, g, function_name="process", file_path="mod.py")
        assert len(result) == 2
        for ci in result:
            assert ci.function_name == "process"
            assert ci.file_path == "mod.py"
        # Independent chains → no cross-chunk data-flow deps
        for ci in result:
            assert ci.depends_on_chunks == []
            assert ci.depended_on_by_chunks == []

    def test_integration_forced_split_creates_cross_chunk_dep(self):
        """A forced split on a closed-chunk dependency is reflected in ChunkInfo."""
        # stmt 3 reads from stmt 0 (in the closed first chunk)
        g = DependencyGraph(
            data={0: [1, 3], 1: [], 2: [], 3: []},
            control_flow={0: [], 1: [], 2: [], 3: []},
            num_statements=4,
        )
        # Use max=3 (< 4 stmts) so the single-chunk short-circuit doesn't fire.
        chunks = group_into_chunks(g, max_statements_per_chunk=3)
        result = annotate_chunks(chunks, g)
        # Locate the chunk containing stmt 3 and chunk containing stmt 0
        chunk_of = {idx: ci.chunk_index for ci in result for idx in ci.statement_indices}
        c0 = chunk_of[0]
        c3 = chunk_of[3]
        assert c0 != c3
        assert c0 in result[c3].depends_on_chunks
        assert c3 in result[c0].depended_on_by_chunks

    def test_integration_full_metadata(self):
        """Full pipeline: build → chunk → annotate carries all metadata."""
        code = textwrap.dedent("""\
            def compute(n):
                total = 0
                for i in range(n):
                    total += i
                return total
        """)
        g = build_dependency_graph(code)
        chunks = group_into_chunks(g)
        result = annotate_chunks(
            chunks, g,
            function_name="compute",
            file_path="/project/math.py",
            function_id=7,
        )
        assert all(ci.function_name == "compute" for ci in result)
        assert all(ci.file_path == "/project/math.py" for ci in result)
        assert all(ci.function_id == 7 for ci in result)
        # chunk_index matches list position
        for i, ci in enumerate(result):
            assert ci.chunk_index == i
        # statement_start ≤ statement_end
        for ci in result:
            assert ci.statement_start <= ci.statement_end

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def test_importable_from_package(self):
        from code_similarity_mcp.parser import annotate_chunks as ac, ChunkInfo as CI
        assert callable(ac)
        assert CI is ChunkInfo

    def test_importable_from_base(self):
        from code_similarity_mcp.parser.base import annotate_chunks as ac, ChunkInfo as CI
        assert callable(ac)
        assert CI is ChunkInfo
