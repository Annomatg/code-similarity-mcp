---
paths:
  - "tests/test_*.py"
  - "tests/**/test_*.py"
---

# Test File Rules

## Async Tests

This project uses `pytest-asyncio` with `asyncio_mode = strict` (set in `pyproject.toml`).
Async test functions **must** be decorated explicitly:

```python
import pytest

@pytest.mark.asyncio
async def test_something():
    result = await some_async_function()
    assert result == expected
```

## Running Tests

Always run after changes:
```bash
.venv/Scripts/python.exe -m pytest tests/ -v
```

To run a single file:
```bash
.venv/Scripts/python.exe -m pytest tests/test_normalizer.py -v
```

## Test Structure

- One test file per module (e.g. `test_parser.py`, `test_normalizer.py`)
- Group related tests in classes when there are many cases
- Use `pytest.fixture` for shared setup; prefer `scope="module"` for expensive resources (e.g. model loading)
