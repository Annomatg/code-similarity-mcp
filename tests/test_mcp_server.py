"""Unit tests for MCP server setup and transport (feature #12).

Verifies that:
- The FastMCP app is created with the correct name.
- Both tools are registered and discoverable.
- The stdio transport entry point exists and is callable.
- The __main__.py entry point exposes main().
"""

from __future__ import annotations

import asyncio
import importlib
import inspect


# ---------------------------------------------------------------------------
# Tests: server module structure
# ---------------------------------------------------------------------------


def test_server_module_importable():
    """code_similarity_mcp.mcp.server must be importable without errors."""
    import code_similarity_mcp.mcp.server  # noqa: F401


def test_app_is_fastmcp_instance():
    """app must be a FastMCP instance."""
    from mcp.server.fastmcp import FastMCP
    from code_similarity_mcp.mcp.server import app

    assert isinstance(app, FastMCP)


def test_app_name():
    """FastMCP app must be named 'code-similarity-mcp'."""
    from code_similarity_mcp.mcp.server import app

    assert app.name == "code-similarity-mcp"


# ---------------------------------------------------------------------------
# Tests: tool registration
# ---------------------------------------------------------------------------


def test_registered_tools_include_index_repository():
    """index_repository must be registered as an MCP tool."""
    from code_similarity_mcp.mcp.server import app

    tools = asyncio.run(app.list_tools())
    names = {t.name for t in tools}
    assert "index_repository" in names


def test_registered_tools_include_analyze_new_code():
    """analyze_new_code must be registered as an MCP tool."""
    from code_similarity_mcp.mcp.server import app

    tools = asyncio.run(app.list_tools())
    names = {t.name for t in tools}
    assert "analyze_new_code" in names


def test_exactly_five_tools_registered():
    """Exactly the five expected tools should be registered."""
    from code_similarity_mcp.mcp.server import app

    tools = asyncio.run(app.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "index_repository",
        "analyze_new_code",
        "analyze_project",
        "find_large_functions",
        "chunk_repository",
    }


def test_index_repository_tool_has_description():
    """index_repository tool must have a non-empty description."""
    from code_similarity_mcp.mcp.server import app

    tools = asyncio.run(app.list_tools())
    tool = next(t for t in tools if t.name == "index_repository")
    assert tool.description and len(tool.description.strip()) > 0


def test_analyze_new_code_tool_has_description():
    """analyze_new_code tool must have a non-empty description."""
    from code_similarity_mcp.mcp.server import app

    tools = asyncio.run(app.list_tools())
    tool = next(t for t in tools if t.name == "analyze_new_code")
    assert tool.description and len(tool.description.strip()) > 0


def test_index_repository_tool_schema_has_repository_root():
    """index_repository input schema must include repository_root parameter."""
    from code_similarity_mcp.mcp.server import app

    tools = asyncio.run(app.list_tools())
    tool = next(t for t in tools if t.name == "index_repository")
    props = tool.inputSchema.get("properties", {})
    assert "repository_root" in props


def test_analyze_new_code_tool_schema_has_code_snippet():
    """analyze_new_code input schema must include code_snippet parameter."""
    from code_similarity_mcp.mcp.server import app

    tools = asyncio.run(app.list_tools())
    tool = next(t for t in tools if t.name == "analyze_new_code")
    props = tool.inputSchema.get("properties", {})
    assert "code_snippet" in props


# ---------------------------------------------------------------------------
# Tests: stdio transport
# ---------------------------------------------------------------------------


def test_app_has_run_stdio_async():
    """FastMCP app must expose run_stdio_async for stdio transport."""
    from code_similarity_mcp.mcp.server import app

    assert callable(getattr(app, "run_stdio_async", None))


def test_main_function_exists():
    """server.main() must exist and be callable."""
    from code_similarity_mcp.mcp.server import main

    assert callable(main)


def test_main_is_synchronous():
    """main() should be a regular (non-async) function that manages its own loop."""
    from code_similarity_mcp.mcp.server import main

    assert not inspect.iscoroutinefunction(main)


# ---------------------------------------------------------------------------
# Tests: __main__.py entry point
# ---------------------------------------------------------------------------


def test_main_module_importable():
    """code_similarity_mcp.__main__ must be importable without errors."""
    import code_similarity_mcp.__main__  # noqa: F401


def test_main_module_exposes_main():
    """__main__ module must expose a callable 'main' (for the script entry point)."""
    import code_similarity_mcp.__main__ as entry

    assert callable(getattr(entry, "main", None))


def test_entry_point_main_is_same_as_server_main():
    """The entry point main() must be the same callable as server.main()."""
    import code_similarity_mcp.__main__ as entry
    from code_similarity_mcp.mcp.server import main as server_main

    assert entry.main is server_main
