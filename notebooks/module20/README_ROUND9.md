# BBO Capstone - Round 9 (18 Data Points)

## Overview

Round 9 uses your accumulated 18 data points (8 rounds) to generate the ninth set of queries. Module 20 focuses on **scaling laws and emergent behaviours**: how scaling and emergent properties alter efficiency, robustness and risk.

## Quick Start

### 1. Add Round 8 Results to outputs_8.txt

After receiving Round 8 results from the portal, replace the 8th row (placeholder zeros) in `results/outputs_8.txt` with the actual values.

### 2. Generate Round 9 Queries

```bash
cd notebooks/module20
python generate_round9_from_results.py
```

This will:
- Read `results/inputs_8.txt` and `results/outputs_8.txt`
- Update `module17/query_history.json` with Round 8 data
- Generate GP-optimised Round 9 queries (xi=0.005, more exploitation)
- Save to `round_9_portal_submission.txt` and `results/inputs_9.txt`
- Print a scaling analysis (improvement trajectory per function)

### 3. Submit to Portal

Copy queries from `round_9_portal_submission.txt` into the capstone portal (one per function).

**Format:** `0.XXXXXX-0.XXXXXX-...` (each value 6 decimal places)

### 4. Post Reflection

Edit `round_9_reflection.md` to answer the four prompts. Keep under **700 words**.

## Reflection Prompts (Part 2)

1. How do scaling laws influence your current query choices? Diminishing returns or steady improvements?
2. Where might emergent behaviours alter your expectations, and how are you preparing?
3. What trade-offs between cost, robustness and performance are shaping your strategy?
4. How do you balance predictable optimisation with the risk of sudden emergent capabilities?

## Files

| File | Purpose |
|------|---------|
| `generate_round9_from_results.py` | Round 9 query generator with scaling analysis |
| `round_9_portal_submission.txt` | Queries for portal (generated) |
| `round_9_reflection.md` | Reflection template |
| `README_ROUND9.md` | This file |

## Completion Checklist

- [ ] Add Round 8 results to `results/outputs_8.txt`
- [ ] Run `generate_round9_from_results.py`
- [ ] Submit 8 queries to capstone portal
- [ ] Post reflection (<700 words) to discussion board
