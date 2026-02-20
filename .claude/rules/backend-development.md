---
paths:
  - "src/**/*.py"
  - "tests/**/*.py"
---

# Development Rules

## Test-Driven Development (REQUIRED)

When writing or modifying functionality:

1. **Write tests FIRST** or immediately after implementation
2. **Run tests** to verify everything works
3. **Only commit** when all tests pass

### Testing Workflow

```bash
# After making changes, ALWAYS run:
.venv/Scripts/python.exe -m pytest tests/ -v
```

**All tests must pass before committing.**

### Test Coverage Requirements

For new features:
- Success cases (happy path)
- Error cases (invalid input, unsupported language, etc.)
- Edge cases (empty input, boundaries)
