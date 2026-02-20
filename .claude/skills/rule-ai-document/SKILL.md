---
name: ai-document
description: Use when creating or editing AI-targeted markdown documentation (CLAUDE.md, README.md, SKILL.md). Ensures conciseness, eliminates duplication, and prevents common-sense bloat.
---

# AI Document Writer

## When to Use

**Use when:**
- Creating/editing CLAUDE.md files (project or module level)
- Creating/editing README.md files for modules
- Creating/editing SKILL.md or agent .md files
- Reviewing documentation for bloat or redundancy
- Ensuring documentation follows AI consumption best practices

**Don't use when:**
- Writing user-facing documentation (end-user guides, tutorials)
- Creating code comments (use coding-conventions)
- Writing commit messages
- Creating agent/skill files → Use agent-generator or skill-writer agents

## Quick Reference

**Target Locations:**
- `.claude/CLAUDE.md` - Project-level AI instructions
- `Modules/[Module]/CLAUDE.md` - Module quick reference (~25 lines)
- `Modules/[Module]/README.md` - Detailed reference (~30-60 lines, zero duplication)
- `.claude/skills/[skill]/SKILL.md` - Skill documentation (100-150 lines)
- `.claude/agents/[agent].md` - Agent documentation (150-200 lines)

**Core Principles:**
- Eliminate common sense
- No duplication across files
- Structured over prose
- Imperative language
- Scannable (bullets, tables, headers)

## Pattern 1: CLAUDE.md (Module Quick Reference)

**Target: ~25 lines**
```markdown
# Module Name

## Key Events
- EventName - When published
- EventName - When published

## Key Services
- ServiceName - Purpose

## Key Configs
- ConfigName.csv - Contents

## Dependencies
Module dependencies

## Quick Patterns
```language
// Minimal example
```
```

**Rules**: No "What is", no architecture explanations, just facts and locations

## Pattern 2: README.md (Detailed Reference)

**Target: ~30-60 lines max** (Option B: zero duplication with CLAUDE.md)
```markdown
# Module Name Reference

## Events Published
| Event | Frequency | Subscribers | Purpose |
[Detailed event table]

## Events Subscribed
| Event | Publisher | Purpose |
[Detailed subscription table]

## Configuration Schemas
### ConfigName.csv
- Field definitions with types
```

**Rules**: ONLY detailed tables/schemas not in CLAUDE.md. No prose, no duplication.

## Pattern 3: Validation Checklist

**Before finalizing any .md file:**

| Check | Requirement |
|-------|------------|
| Length | CLAUDE.md ≤30 lines, README ≤60 lines, SKILL ≤200 lines |
| Structure | Headers, bullets, tables (not paragraphs) |
| Duplication | Zero duplication between CLAUDE.md and README.md |
| Common Sense | No "What is X" or obvious concepts |
| Examples | Minimal, only if non-obvious |
| Language | Imperative, not conversational |
| Scanning | Can extract key info in 10 seconds |

## Critical Rules

1. **Eliminate Common Sense**: Remove any concept already in Claude's training (basic C#, standard patterns)
2. **No Duplication**: Each doc has a specific, non-overlapping purpose (Quick ref vs comprehensive vs implementation)
3. **Structure Over Prose**: Use bullets, tables, code blocks, not paragraphs
4. **Imperative Language**: "Use X", "Create Y", not "You should use X" or "It's recommended to Y"
5. **Length Targets**: CLAUDE.md (~25 lines), README (~150 lines), SKILL (~100-150 lines), Agent (~150-200 lines)
6. **Scannable**: Headers and bullets allow quick navigation to relevant section
7. **Examples Last**: Only include code examples if pattern is non-obvious

## Common Mistakes

| ❌ Wrong | ✅ Correct |
|---------|-----------|
| "What is EventBus? EventBus is a..." | "EventBus location: `Core/Events/System/EventBus.cs`" |
| Paragraph explanations | Bulleted lists with single-line items |
| "You should always call base._Ready()" | "Call base._Ready() first" |
| Repeating info from CLAUDE.md in README | CLAUDE.md = quick ref, README = comprehensive |
| "It's important to note that..." | Remove entirely (implied importance) |
| Multiple examples of same pattern | One minimal example if needed |
| "This is a common pattern where..." | Show pattern, no explanation |

## Documentation Hierarchy

**Purpose of each file type:**

```
.claude/CLAUDE.md
  └─ Project-level routing and guidelines
     └─ NO code examples, NO module details
     └─ Links to module docs

Modules/[Module]/CLAUDE.md
  └─ Quick reference (~25 lines)
     └─ Purpose, key files, events summary, quick tips
     └─ For routing decisions only

Modules/[Module]/README.md
  └─ Detailed reference (~30-60 lines)
     └─ ONLY: Event tables, config schemas, implementation specifics
     └─ Zero duplication with CLAUDE.md

.claude/skills/[skill]/SKILL.md
  └─ Implementation guidance (100-150 lines)
     └─ When to use, patterns, rules, mistakes
     └─ Code examples for non-obvious patterns

.claude/agents/[agent].md
  └─ Workflow execution (150-200 lines)
     └─ Phase-by-phase commands
     └─ Validation criteria, output formats
```

## Refactoring Process

**For bloated documentation:**

1. **Identify bloat**:
   - Mark "What is" sections
   - Mark common sense explanations
   - Mark redundant examples
   - Mark conversational prose

2. **Strip aggressively**:
   - Remove all marked sections
   - Convert prose to bullets
   - Remove ALL duplication between CLAUDE.md and README.md
   - Target 70-80% reduction for READMEs

3. **Verify hierarchy**:
   - Check ZERO duplication between CLAUDE.md and README.md
   - CLAUDE.md = routing only, README.md = detailed tables/schemas only
   - Confirm length targets: CLAUDE ≤30, README ≤60

4. **Validate scannability**:
   - Can find any section in <10 seconds?
   - Are headers clear and distinct?
   - Are bullets single-line?

## Related Files

- `.claude/agents/skill-writer.md` - Creating SKILL.md files
- `.claude/agents/agent-generator.md` - Creating agent .md files
- `.claude/CLAUDE.md` - Project documentation example
- `Modules/Core/CLAUDE.md` - Module quick reference example
- `Modules/Core/README.md` - Module comprehensive docs example
