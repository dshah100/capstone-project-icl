"""
BBO Capstone - Round 8 Query Generator
======================================
Generates Round 8 queries with 17 data points (7 rounds accumulated).
Module 19 focuses on LLM-centred strategy: prompting, decoding, token budgets, attention.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from module17.bbo_capstone_framework import BBOCapstoneManager


def ensure_round7_in_history():
    """Ensure query_history has Round 7 data."""
    history_path = Path(__file__).parent.parent / "module17" / "query_history.json"
    with open(history_path, 'r') as f:
        history = json.load(f)

    func1 = history.get('function_1', [])
    n_rounds = len(func1) if isinstance(func1, list) else 0
    if n_rounds < 7:
        print("\n⚠ query_history has only {} rounds. Run update_query_history_round7.py with portal results first.".format(n_rounds))
        return False
    return True


def generate_round8_queries(
    acquisition='ei',
    xi=0.01,
    kappa=2.0,
    round_adaptive=True
):
    """
    Generate Round 8 queries (17 data points).
    Round 8: more exploitation, less exploration.
    """
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 12 + "BBO CAPSTONE - ROUND 8 QUERY GENERATOR (17 Data Points)" + " " * 12 + "║")
    print("╚" + "═" * 68 + "╝")

    if not ensure_round7_in_history():
        return None

    manager = BBOCapstoneManager()
    history_path = str(Path(__file__).parent.parent / "module17" / "query_history.json")
    manager.load_query_history_from_file(history_path)

    # Round 8: 7 rounds of data — exploit more
    round_num = 8
    if round_adaptive:
        xi_adjusted = xi * 0.7
        kappa_adjusted = kappa * 0.85
    else:
        xi_adjusted = xi
        kappa_adjusted = kappa

    print("\n" + "=" * 70)
    print(f"GENERATING ROUND {round_num} QUERIES")
    print("=" * 70)
    print(f"Strategy: {acquisition.upper()}, xi={xi_adjusted:.4f}, kappa={kappa_adjusted:.2f}")
    print("(17 data points — balancing exploitation with residual exploration)\n")

    queries = manager.generate_round_queries(
        round_num=round_num,
        acquisition=acquisition,
        xi=xi_adjusted,
        kappa=kappa_adjusted
    )

    output_path = Path(__file__).parent / "round_8_portal_submission.txt"
    with open(output_path, 'w') as f:
        f.write("ROUND 8 QUERIES - 17 Data Points\n")
        f.write("=" * 60 + "\n\n")
        f.write("Copy and paste these queries into the capstone portal:\n\n")
        for func_id, query in queries.items():
            f.write(f"Function {func_id}:\n{query}\n\n")
        f.write("=" * 60 + "\n")
        f.write("FORMAT: Each value 0.XXXXXX (6 decimal places)\n")
        f.write("=" * 60 + "\n")

    print("\n" + "=" * 70)
    print("✓ ROUND 8 QUERIES GENERATED")
    print("=" * 70)
    print(f"\nSaved to: {output_path}")
    print("\nNext steps:")
    print("  1. Submit queries to capstone portal")
    print("  2. Post reflection (round_8_reflection.md) to discussion board")
    print("=" * 70 + "\n")

    return queries


def main():
    queries = generate_round8_queries(
        acquisition='ei',
        xi=0.01,
        round_adaptive=True
    )
    if queries:
        print("\nPortal submission format:")
        for func_id, query in sorted(queries.items()):
            print(f"  Function {func_id}: {query}")


if __name__ == "__main__":
    main()
