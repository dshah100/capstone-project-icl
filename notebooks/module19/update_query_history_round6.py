"""
Update Query History with Round 6 Data
======================================
Adds Round 6 queries and results to query_history.json for Round 7 generation.
Run this before generating Round 7 queries.
"""

import json
import re
from pathlib import Path


def parse_outputs_file(filepath):
    """Parse outputs file - each line has 8 np.float64 values."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    rounds = []
    for line in content.strip().split('\n'):
        pattern = r'np\.float64\(([0-9.e+-]+)\)'
        matches = re.findall(pattern, line)
        if matches:
            values = [float(x) for x in matches]
            rounds.append(values)
    
    return rounds


def main():
    print("\n" + "="*70)
    print("UPDATING QUERY HISTORY WITH ROUND 6 DATA")
    print("="*70)
    
    base_path = Path(__file__).parent.parent.parent / "results"
    outputs_path = base_path / "outputs_6.txt"
    
    # query_history lives in module17
    history_path = Path(__file__).parent.parent / "module17" / "query_history.json"
    if not history_path.exists():
        print("\n✗ query_history.json not found. Run import first.")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Remove metadata keys if present
    for key in list(history.keys()):
        if key.startswith('_'):
            del history[key]
    
    # Round 6 queries from portal submission (correct dimensions per framework)
    round6_queries = {
        1: [0.55, 0.5],
        2: [0.895432, 0.465123, 0.405678],
        3: [0.465890, 0.568234, 0.545678, 0.587654],
        4: [0.456789, 0.492345, 0.445678, 0.534567, 0.518234],
        5: [0.518345, 0.556789, 0.492345, 0.545678, 0.512345, 0.441234],
        6: [0.492345, 0.408765, 0.452345, 0.536789, 0.468765, 0.567890, 0.518234],
        7: [0.534567, 0.176543, 0.028765, 0.342345, 0.965432, 0.523456, 0.854321, 0.089876],
        8: [0.298765, 0.487654, 0.745678, 0.905432, 0.408765, 0.912345, 0.512345, 0.789012, 0.883456],
    }
    
    # Get Round 6 outputs
    if outputs_path.exists():
        all_outputs = parse_outputs_file(outputs_path)
        if len(all_outputs) >= 6:
            round6_outputs = all_outputs[5]  # 0-indexed, round 6
            print(f"\n✓ Loaded Round 6 outputs from {outputs_path.name}")
        else:
            print("\n⚠ outputs_6.txt has fewer than 6 rounds. Using placeholder values.")
            round6_outputs = [0.0] * 8
    else:
        print("\n⚠ outputs_6.txt not found. Using placeholder values.")
        round6_outputs = [0.0] * 8
    
    # Add Round 6 to history
    for func_id in range(1, 9):
        key = f"function_{func_id}"
        if key not in history:
            history[key] = []
        
        # Check if Round 6 already exists
        existing_rounds = [obs.get('round') for obs in history[key]]
        if 6 in existing_rounds:
            print(f"  Function {func_id}: Round 6 already in history, skipping")
            continue
        
        query = round6_queries[func_id]
        result = round6_outputs[func_id - 1] if func_id - 1 < len(round6_outputs) else 0.0
        
        # Use absolute value for distance (minimize)
        result_for_gp = abs(result)
        
        history[key].append({
            "round": 6,
            "query": [float(f"{v:.6f}") for v in query],
            "result": float(f"{result_for_gp:.6e}")
        })
        print(f"  Function {func_id}: Added Round 6, result={result_for_gp:.6e}")
    
    # Save updated history
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("✓ QUERY HISTORY UPDATED")
    print("="*70)
    print(f"\nSaved to: {history_path}")
    print(f"Each function now has {len(history['function_1'])} observations (16 data points total)")
    print("\nNext: Run round7_generator.py to generate Round 7 queries")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
