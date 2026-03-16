"""
Generate Round 8 from results folder (inputs_7, outputs_7)
=========================================================
Reads inputs_7.txt and outputs_7.txt from the results folder,
updates query_history.json with Round 7 data, and generates
Round 8 queries for portal submission and inputs_8.txt.
"""

import json
import re
import sys
from pathlib import Path

# Add parent for module17 import
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_inputs_file(filepath):
    """Parse inputs file - find all array([...]) and chunk into rows of 8."""
    with open(filepath, 'r') as f:
        content = f.read()

    pattern = r'array\(\[([0-9.,\s]+)\]\)'
    matches = re.findall(pattern, content)
    if len(matches) % 8 != 0:
        raise ValueError(f"Expected multiple of 8 arrays, got {len(matches)}")

    rounds = []
    for i in range(0, len(matches), 8):
        row = []
        for m in matches[i : i + 8]:
            vals = [float(x.strip()) for x in m.split(',')]
            row.append(vals)
        rounds.append(row)
    return rounds


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
    print("\n" + "=" * 70)
    print("GENERATE ROUND 8 FROM inputs_7 & outputs_7")
    print("=" * 70)

    base_path = Path(__file__).parent.parent.parent / "results"
    inputs_path = base_path / "inputs_7.txt"
    outputs_path = base_path / "outputs_7.txt"
    history_path = Path(__file__).parent.parent / "module17" / "query_history.json"

    if not inputs_path.exists():
        print(f"\n[ERROR] {inputs_path} not found.")
        return

    if not outputs_path.exists():
        print(f"\n[ERROR] {outputs_path} not found.")
        print("  Add outputs_7.txt with Round 7 results from the portal.")
        print("  Format: one line per round, 8 values per line, e.g.")
        print("  [np.float64(0.001), np.float64(0.5), ...]")
        return

    # Parse inputs_7
    all_inputs = parse_inputs_file(inputs_path)
    if len(all_inputs) < 7:
        print(f"\n[ERROR] inputs_7.txt has only {len(all_inputs)} rounds. Need 7.")
        return

    # Parse outputs_7
    all_outputs = parse_outputs_file(outputs_path)
    if len(all_outputs) < 7:
        print(f"\n[ERROR] outputs_7.txt has only {len(all_outputs)} rounds. Need 7.")
        return

    round7_queries = all_inputs[6]  # 0-indexed
    round7_outputs = all_outputs[6]

    print(f"\n[OK] Loaded Round 7 from {inputs_path.name} and {outputs_path.name}")

    # Load query_history
    with open(history_path, 'r') as f:
        history = json.load(f)

    for key in list(history.keys()):
        if key.startswith('_'):
            del history[key]

    # Add Round 7 to history
    for func_id in range(1, 9):
        key = f"function_{func_id}"
        if key not in history:
            history[key] = []

        existing_rounds = [obs.get('round') for obs in history[key] if isinstance(obs, dict)]
        if 7 in existing_rounds:
            print(f"  Function {func_id}: Round 7 already in history, skipping")
            continue

        query = round7_queries[func_id - 1]
        result = round7_outputs[func_id - 1] if func_id - 1 < len(round7_outputs) else 0.0
        result_for_gp = abs(result)

        history[key].append({
            "round": 7,
            "query": [float(f"{v:.6f}") for v in query],
            "result": float(f"{result_for_gp:.6e}")
        })
        print(f"  Function {func_id}: Added Round 7, result={result_for_gp:.6e}")

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n[OK] Updated query_history.json with Round 7")

    # Generate Round 8
    from module17.bbo_capstone_framework import BBOCapstoneManager

    manager = BBOCapstoneManager()
    manager.load_query_history_from_file(str(history_path))

    queries = manager.generate_round_queries(
        round_num=8,
        acquisition='ei',
        xi=0.007,
        kappa=1.7
    )

    # Save portal submission
    portal_path = Path(__file__).parent / "round_8_portal_submission.txt"
    with open(portal_path, 'w') as f:
        f.write("ROUND 8 QUERIES - 17 Data Points\n")
        f.write("=" * 60 + "\n\n")
        f.write("Copy and paste these queries into the capstone portal:\n\n")
        for func_id, query in queries.items():
            f.write(f"Function {func_id}:\n{query}\n\n")
        f.write("=" * 60 + "\n")
        f.write("FORMAT: Each value 0.XXXXXX (6 decimal places)\n")
        f.write("=" * 60 + "\n")

    print(f"\n[OK] Saved portal submission to {portal_path}")

    # Save inputs_8.txt (inputs_7 + Round 8 row in results format)
    round8_parts = []
    for func_id in sorted(queries.keys()):
        qstr = queries[func_id]
        vals = [float(x) for x in qstr.split('-')]
        round8_parts.append(f"array([{', '.join(f'{v:.6f}' for v in vals)}])")

    inputs_8_path = base_path / "inputs_8.txt"
    with open(inputs_path, 'r') as f:
        inputs_7_content = f.read()

    round8_line = '[' + ', '.join(round8_parts) + ']'
    with open(inputs_8_path, 'w') as f:
        f.write(inputs_7_content.rstrip())
        f.write("\n")
        f.write(round8_line)
        f.write("\n")

    print(f"[OK] Saved {inputs_8_path}")

    print("\n" + "=" * 70)
    print("ROUND 8 QUERIES (portal format)")
    print("=" * 70)
    for func_id, query in sorted(queries.items()):
        print(f"  Function {func_id}: {query}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
