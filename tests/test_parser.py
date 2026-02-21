"""Tests for the Python parser (tree-sitter based)."""

import textwrap
import hashlib

import pytest

from code_similarity_mcp.parser.python import PythonParser
from code_similarity_mcp.parser.base import MethodInfo
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
