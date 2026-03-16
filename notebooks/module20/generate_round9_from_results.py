"""
Generate Round 9 from results folder (inputs_8, outputs_8)
=========================================================
Reads inputs_8.txt and outputs_8.txt from the results folder,
updates query_history.json with Round 8 data, and generates
Round 9 queries for portal submission and inputs_9.txt.

Module 20 focus: scaling laws, emergent behaviours, cost-robustness trade-offs.
"""

import json
import re
import sys
from pathlib import Path

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
    print("GENERATE ROUND 9 FROM inputs_8 & outputs_8")
    print("=" * 70)

    base_path = Path(__file__).parent.parent.parent / "results"
    inputs_path = base_path / "inputs_8.txt"
    outputs_path = base_path / "outputs_8.txt"
    history_path = Path(__file__).parent.parent / "module17" / "query_history.json"

    if not inputs_path.exists():
        print(f"\n[ERROR] {inputs_path} not found.")
        return

    if not outputs_path.exists():
        print(f"\n[ERROR] {outputs_path} not found.")
        print("  Add outputs_8.txt with Round 8 results from the portal.")
        print("  Format: one line per round, 8 values per line, e.g.")
        print("  [np.float64(0.001), np.float64(0.5), ...]")
        return

    all_inputs = parse_inputs_file(inputs_path)
    if len(all_inputs) < 8:
        print(f"\n[ERROR] inputs_8.txt has only {len(all_inputs)} rounds. Need 8.")
        return

    all_outputs = parse_outputs_file(outputs_path)
    if len(all_outputs) < 8:
        print(f"\n[ERROR] outputs_8.txt has only {len(all_outputs)} rounds. Need 8.")
        return

    round8_queries = all_inputs[7]  # 0-indexed
    round8_outputs = all_outputs[7]

    print(f"\n[OK] Loaded Round 8 from {inputs_path.name} and {outputs_path.name}")

    # Load query_history
    with open(history_path, 'r') as f:
        history = json.load(f)

    for key in list(history.keys()):
        if key.startswith('_'):
            del history[key]

    # Add Round 8 to history
    for func_id in range(1, 9):
        key = f"function_{func_id}"
        if key not in history:
            history[key] = []

        existing_rounds = [obs.get('round') for obs in history[key] if isinstance(obs, dict)]
        if 8 in existing_rounds:
            print(f"  Function {func_id}: Round 8 already in history, skipping")
            continue

        query = round8_queries[func_id - 1]
        result = round8_outputs[func_id - 1] if func_id - 1 < len(round8_outputs) else 0.0
        result_for_gp = abs(result)

        history[key].append({
            "round": 8,
            "query": [float(f"{v:.6f}") for v in query],
            "result": float(f"{result_for_gp:.6e}")
        })
        print(f"  Function {func_id}: Added Round 8, result={result_for_gp:.6e}")

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n[OK] Updated query_history.json with Round 8")

    # Generate Round 9 (18 data points -- more exploitation, tighter bounds)
    from module17.bbo_capstone_framework import BBOCapstoneManager

    manager = BBOCapstoneManager()
    manager.load_query_history_from_file(str(history_path))

    queries = manager.generate_round_queries(
        round_num=9,
        acquisition='ei',
        xi=0.005,
        kappa=1.5
    )

    # Save portal submission
    portal_path = Path(__file__).parent / "round_9_portal_submission.txt"
    with open(portal_path, 'w') as f:
        f.write("ROUND 9 QUERIES - 18 Data Points\n")
        f.write("=" * 60 + "\n\n")
        f.write("Copy and paste these queries into the capstone portal:\n\n")
        for func_id, query in queries.items():
            f.write(f"Function {func_id}:\n{query}\n\n")
        f.write("=" * 60 + "\n")
        f.write("FORMAT: Each value 0.XXXXXX (6 decimal places)\n")
        f.write("=" * 60 + "\n")

    print(f"\n[OK] Saved portal submission to {portal_path}")

    # Save inputs_9.txt (inputs_8 + Round 9 row)
    round9_parts = []
    for func_id in sorted(queries.keys()):
        qstr = queries[func_id]
        vals = [float(x) for x in qstr.split('-')]
        round9_parts.append(f"array([{', '.join(f'{v:.6f}' for v in vals)}])")

    inputs_9_path = base_path / "inputs_9.txt"
    with open(inputs_path, 'r') as f:
        inputs_8_content = f.read()

    round9_line = '[' + ', '.join(round9_parts) + ']'
    with open(inputs_9_path, 'w') as f:
        f.write(inputs_8_content.rstrip())
        f.write("\n")
        f.write(round9_line)
        f.write("\n")

    print(f"[OK] Saved {inputs_9_path}")

    # Print summary with scaling analysis
    print("\n" + "=" * 70)
    print("ROUND 9 QUERIES (portal format)")
    print("=" * 70)
    for func_id, query in sorted(queries.items()):
        opt = manager.optimizers[func_id]
        best_x, best_y = opt.get_best_observation()
        print(f"  Function {func_id}: {query}")
        print(f"    Best so far: {best_y:.6e}")
    print("=" * 70)

    # Scaling analysis: show improvement trajectory
    print("\n" + "=" * 70)
    print("SCALING ANALYSIS: Improvement per Round")
    print("=" * 70)
    for func_id in range(1, 9):
        opt = manager.optimizers[func_id]
        if len(opt.y_observed) < 2:
            continue
        ys = opt.y_observed
        best_so_far = [min(ys[:i+1]) for i in range(len(ys))]
        improvements = [best_so_far[i-1] - best_so_far[i] for i in range(1, len(best_so_far))]
        print(f"  Function {func_id}: best_trajectory={[f'{b:.4e}' for b in best_so_far]}")
        diminishing = all(improvements[i] <= improvements[i-1] for i in range(1, len(improvements))) if len(improvements) > 1 else False
        if diminishing:
            print(f"    -> Diminishing returns detected")
        else:
            print(f"    -> Non-monotonic improvements (possible emergent jump)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
