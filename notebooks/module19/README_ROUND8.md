# BBO Capstone - Round 8 (17 Data Points)

## Overview

Round 8 uses your accumulated 17 data points (7 rounds) to generate the eighth set of queries. This round focuses on **LLM-centred strategy**: prompting, decoding controls, token budgets, attention limits, and how they shape BBO.

## Quick Start

### 1. Add Round 7 Results to query_history.json

After receiving Round 7 results from the portal:

1. Edit `update_query_history_round7.py` and fill in the `result` values in `ROUND7`.
2. Run:

```bash
cd notebooks/module19
python update_query_history_round7.py
```

### 2. Generate Round 8 Queries

```bash
cd notebooks/module19
python round8_generator.py
```

This saves queries to `round_8_portal_submission.txt`.

### 3. Submit to Portal

Copy queries from `round_8_portal_submission.txt` into the capstone portal (one per function).

**Format:** `0.XXXXXX-0.XXXXXX-...` (each value 6 decimal places)

### 4. Post Reflection

Edit `round_8_reflection.md` to answer the seven prompts. Keep under **700 words** for the discussion board.

## Reflection Prompts (Part 2)

1. **Prompt Patterns**: Zero-shot, few-shot, simplified vs. structured prompts
2. **Decoding Settings**: temperature, top-p, top-k, max-tokens
3. **Tokenization Effects**: Token boundaries, truncation, edge cases
4. **Limitations with 17 Data Points**: Prompt overfitting, attention, diminishing returns
5. **Hallucination Reduction Strategies**
6. **Scaling Strategies** for larger data sets
7. **Practitioner's Balance**: Exploration, risk, computational constraints

## Key Questions (from Module 19)

- Do structured prompts reduce uncertainty enough to justify more exploitation?
- Or do you still explore because attention is finite and tokenisation can mangle edge cases?
- With 17 data points: prompt overfitting, tokenisation artefacts, diminishing returns from longer context?

## Files

| File | Purpose |
|------|---------|
| `round8_generator.py` | Round 8 query generator |
| `round_8_portal_submission.txt` | Queries for portal |
| `round_8_reflection.md` | LLM-focused reflection template |
| `update_query_history_round7.py` | Add Round 7 data (edit results first) |

## Completion Checklist

- [ ] Add Round 7 results to `update_query_history_round7.py`
- [ ] Run `update_query_history_round7.py`
- [ ] Run `round8_generator.py`
- [ ] Submit 8 queries to capstone portal
- [ ] Post reflection (<700 words) to discussion board
- [ ] Revisit capstone overview and Mini-lessons 12.7, 12.8
