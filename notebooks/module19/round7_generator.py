"""
BBO Capstone - Round 7 Query Generator
======================================
Generates Round 7 queries with 16 data points (6 rounds accumulated).
Includes hyperparameter tuning support for xi, kappa, and acquisition strategy.
"""

import sys
import json
from pathlib import Path

# Add module17 for bbo_capstone_framework import
sys.path.insert(0, str(Path(__file__).parent.parent))
from module17.bbo_capstone_framework import BBOCapstoneManager


def ensure_round6_in_history():
    """Ensure query_history has Round 6 data. Run update script if needed."""
    history_path = Path(__file__).parent.parent / "module17" / "query_history.json"
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    n_rounds = len(history.get('function_1', []))
    if n_rounds < 6:
        print("\n⚠ query_history has only {} rounds. Run update_query_history_round6.py first.".format(n_rounds))
        return False
    return True


def hyperparameter_tuning_comparison(manager, func_id=1):
    """
    Compare different hyperparameter settings (manual/grid-style).
    Demonstrates hyperparameter tuning for reflection.
    """
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING COMPARISON (Function 1)")
    print("="*70)
    
    optimizer = manager.optimizers[func_id]
    if len(optimizer.y_observed) < 2:
        print("Need at least 2 observations.")
        return {}
    
    results = {}
    
    # Grid over xi values (EI exploration parameter)
    for xi in [0.005, 0.01, 0.02, 0.05]:
        try:
            query = optimizer.propose_next_query(acquisition='ei', xi=xi)
            formatted = optimizer.format_query(query)
            results[f"EI_xi={xi}"] = formatted
            print(f"  EI, xi={xi:0.3f}: {formatted}")
        except Exception as e:
            print(f"  EI, xi={xi}: Error - {e}")
    
    # UCB with different kappa
    for kappa in [1.5, 2.0, 3.0]:
        try:
            query = optimizer.propose_next_query(acquisition='ucb', kappa=kappa)
            formatted = optimizer.format_query(query)
            results[f"UCB_kappa={kappa}"] = formatted
            print(f"  UCB, kappa={kappa}: {formatted}")
        except Exception as e:
            print(f"  UCB, kappa={kappa}: Error - {e}")
    
    return results


def generate_round7_queries(
    acquisition='ei',
    xi=0.01,
    kappa=2.0,
    round_adaptive=True,
    use_tuning=False
):
    """
    Generate Round 7 queries.
    
    Parameters:
    -----------
    acquisition : str
        'ei' (Expected Improvement) or 'ucb' (Upper Confidence Bound)
    xi : float
        EI exploration parameter (higher = more exploration)
    kappa : float
        UCB exploration parameter
    round_adaptive : bool
        If True, reduce exploration in later rounds (exploit more)
    use_tuning : bool
        If True, run hyperparameter comparison for reflection
    """
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*15 + "BBO CAPSTONE - ROUND 7 QUERY GENERATOR" + " "*15 + "║")
    print("╚" + "═"*68 + "╝")
    
    if not ensure_round6_in_history():
        return None
    
    manager = BBOCapstoneManager()
    history_path = str(Path(__file__).parent.parent / "module17" / "query_history.json")
    manager.load_query_history_from_file(history_path)
    
    # Optional: Show hyperparameter tuning comparison
    if use_tuning:
        hyperparameter_tuning_comparison(manager, func_id=1)
    
    # Round 7: with 6 rounds of data, we exploit more (lower exploration)
    round_num = 7
    if round_adaptive:
        # Round 7: reduce exploration (we have 6 data points)
        xi_adjusted = xi * 0.8  # Slightly more exploitation
        kappa_adjusted = kappa * 0.9
    else:
        xi_adjusted = xi
        kappa_adjusted = kappa
    
    print("\n" + "="*70)
    print(f"GENERATING ROUND {round_num} QUERIES")
    print("="*70)
    print(f"Strategy: {acquisition.upper()}, xi={xi_adjusted:.4f}, kappa={kappa_adjusted:.2f}")
    print("(16 data points accumulated - exploiting promising regions)\n")
    
    queries = manager.generate_round_queries(
        round_num=round_num,
        acquisition=acquisition,
        xi=xi_adjusted,
        kappa=kappa_adjusted
    )
    
    # Save to portal submission format
    output_path = Path(__file__).parent / "round_7_portal_submission.txt"
    with open(output_path, 'w') as f:
        f.write(f"ROUND 7 QUERIES - 16 Data Points\n")
        f.write("="*60 + "\n\n")
        f.write("Copy and paste these queries into the capstone portal:\n\n")
        for func_id, query in queries.items():
            f.write(f"Function {func_id}:\n{query}\n\n")
        f.write("="*60 + "\n")
        f.write("FORMAT: Each value 0.XXXXXX (6 decimal places)\n")
        f.write("="*60 + "\n")
    
    print("\n" + "="*70)
    print("✓ ROUND 7 QUERIES GENERATED")
    print("="*70)
    print(f"\nSaved to: {output_path}")
    print("\nNext steps:")
    print("  1. Submit queries to capstone portal")
    print("  2. Post reflection (round_7_reflection.md) to discussion board")
    print("  3. Record results for Round 8")
    print("="*70 + "\n")
    
    return queries


def main():
    """Generate Round 7 queries with default hyperparameters."""
    queries = generate_round7_queries(
        acquisition='ei',
        xi=0.01,
        round_adaptive=True,
        use_tuning=True  # Run tuning comparison for reflection
    )
    
    if queries:
        print("\nPortal submission format:")
        for func_id, query in sorted(queries.items()):
            print(f"  Function {func_id}: {query}")


if __name__ == "__main__":
    main()
