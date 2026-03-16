# BBO Capstone - Round 7 (16 Data Points)

## Overview

Round 7 uses your accumulated 16 data points (6 rounds) to generate the seventh set of queries. This round focuses on **hyperparameter tuning** strategies from the module.

## Quick Start

### 1. Ensure Round 6 Data is in query_history.json

Round 6 has been added to `module17/query_history.json`. If you need to re-import:

```bash
cd notebooks/module18
python update_query_history_round6.py
```

### 2. Generate Round 7 Queries

```bash
cd notebooks/module18
python round7_generator.py
```

This will:
- Run hyperparameter tuning comparison (xi, kappa, EI vs UCB)
- Generate GP-optimized queries for all 8 functions
- Save to `round_7_portal_submission.txt`

**Alternative:** Use `run_round7.py` for a one-shot update + generation.

### 3. Submit to Portal

Copy queries from `round_7_portal_submission.txt` into the capstone portal (one per function).

**Format:** `0.XXXXXX-0.XXXXXX-...` (each value 6 decimal places)

### 4. Post Reflection

Edit `round_7_reflection.md` to answer the prompts. Keep under **700 words** for the discussion board.

## Reflection Prompts (Part 2)

1. Which hyperparameters did you choose to tune, and why?
2. How has hyperparameter tuning changed your query strategy?
3. Which tuning method(s) did you apply, and what trade-offs did you notice?
4. What limitations become clearer with 16 data points?
5. How might you apply tuning to larger data sets?
6. How does black-box tuning prepare you for real-world ML/AI?

## Files

| File | Purpose |
|------|---------|
| `round7_generator.py` | Main Round 7 query generator with hyperparameter tuning |
| `round_7_portal_submission.txt` | Queries for portal (overwritten by generator) |
| `round_7_reflection.md` | Reflection template |
| `update_query_history_round6.py` | Add Round 6 data to history (writes to module17) |
| `run_round7.py` | One-shot: add Round 6 + generate Round 7 |

**Note:** Scripts read/write `query_history.json` in `module17` (shared with BBO framework).

## Hyperparameters in the Framework

- **xi** (default 0.01): EI exploration; higher = more exploration
- **kappa** (default 2.0): UCB exploration
- **acquisition**: 'ei' or 'ucb'
- Round-adaptive: Later rounds use lower exploration (more exploitation)

## Completion Checklist

- [ ] Run `round7_generator.py` to get optimized queries
- [ ] Submit 8 queries to capstone portal
- [ ] Post reflection (<700 words) to discussion board
- [ ] Engage with peers' posts
- [ ] Record Round 7 results for Round 8
