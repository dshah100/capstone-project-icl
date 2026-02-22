"""
Automated Import of All Your BBO Data
======================================
Imports all 5 rounds with actual output values from your files.
"""

import json
import re
import numpy as np


def parse_outputs_file(filepath):
    """Parse outputs file and extract all rounds."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by lines that start with numbers
    lines = content.strip().split('\n')
    
    all_rounds = []
    for line in lines:
        # Extract values: np.float64(value)
        pattern = r'np\.float64\(([0-9.e+-]+)\)'
        matches = re.findall(pattern, line)
        
        if matches:
            values = [float(x) for x in matches]
            all_rounds.append(values)
    
    return all_rounds


def parse_run_log(filepath):
    """Parse a run log file to extract queries."""
    queries = {}
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern 1: "Function X:" format (Round 1)
    pattern1 = r'Function (\d+):\s*\n\s*([0-9.-]+)'
    matches1 = re.findall(pattern1, content)
    
    if matches1:
        for func_id, query_str in matches1:
            func_id = int(func_id)
            query = [float(x) for x in query_str.split('-')]
            queries[func_id] = query
        return queries
    
    # Pattern 2: "pricing_model_version_X (YD):" or "Model X (YD)" format
    pattern2 = r'(?:pricing_model_version_|Model )(\d+) \((\d+)D\)(?:\s+Query)?:\s*([0-9.-]+)'
    matches2 = re.findall(pattern2, content)
    
    if matches2:
        for model_id, dims, query_str in matches2:
            func_id = int(model_id) + 1  # version_0/Model 0 = Function 1
            query = [float(x) for x in query_str.split('-')]
            queries[func_id] = query
        return queries
    
    return queries


def main():
    print("\n" + "╔"+"═"*68+"╗")
    print("║" + " "*15 + "IMPORTING ALL YOUR BBO DATA" + " "*26 + "║")
    print("╚"+"═"*68+"╝")
    
    base_path = r"C:\work\AICourse\SalesforceImperialCourse1\capstone-project-icl\results"
    
    # Parse all run logs
    run_files = [
        (1, f"{base_path}\\1stRun.log"),
        (2, f"{base_path}\\2ndRun.txt"),
        (3, f"{base_path}\\3rdRun.txt"),
        (4, f"{base_path}\\4thRun.log"),
        (5, f"{base_path}\\5thRun.txt"),
    ]
    
    # Parse output file (has all 5 rounds)
    print("\nReading outputs...")
    outputs_path = f"{base_path}\\outputs_5.txt"
    all_rounds_outputs = parse_outputs_file(outputs_path)
    
    print(f"Found outputs for {len(all_rounds_outputs)} rounds")
    
    # Initialize history
    history = {f"function_{i}": [] for i in range(1, 9)}
    
    # Process each round
    for round_num, run_file in run_files:
        print(f"\nProcessing Round {round_num}...")
        
        # Parse queries
        queries = parse_run_log(run_file)
        print(f"  Found {len(queries)} queries")
        
        # Get outputs for this round (round_num - 1 because 0-indexed)
        if round_num - 1 < len(all_rounds_outputs):
            round_outputs = all_rounds_outputs[round_num - 1]
        else:
            round_outputs = []
        
        # Match queries with outputs
        for func_id in range(1, 9):
            if func_id in queries:
                query = queries[func_id]
                
                # Get output value
                if func_id - 1 < len(round_outputs):
                    output_value = round_outputs[func_id - 1]
                else:
                    output_value = 0.0
                
                history[f"function_{func_id}"].append({
                    "round": round_num,
                    "query": [float(f"{v:.6f}") for v in query],
                    "result": float(f"{output_value:.6f}")
                })
                
                print(f"    Function {func_id} ({len(query)}D): result={output_value:.6f}")
    
    # Save to file
    output_file = 'query_history_with_outputs.json'
    with open(output_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("✓ IMPORT COMPLETE")
    print("="*70)
    print(f"\nSaved to: {output_file}")
    
    # Summary
    print("\nSummary:")
    for func_id in range(1, 9):
        key = f"function_{func_id}"
        n_obs = len(history[key])
        if n_obs > 0:
            dims = len(history[key][0]['query'])
            results = [obs['result'] for obs in history[key]]
            min_result = min(results)
            max_result = max(results)
            print(f"  Function {func_id} ({dims}D): {n_obs} observations, "
                  f"range: [{min_result:.6f}, {max_result:.6f}]")
    
    print("\n" + "="*70)
    print("⚠️ IMPORTANT NOTE")
    print("="*70)
    print("\nThe output values appear to be transformed (possibly negative")
    print("log-likelihood), NOT standard Euclidean distances.")
    print("\nTypical BBO distances are positive and usually 0-2 for unit hypercube.")
    print("\nThese values include:")
    print("  • Very small numbers (e.g., 1e-117)")
    print("  • Negative values")
    print("  • Large positive values")
    print("\n❓ Are these the ACTUAL distances from the capstone portal?")
    print("\nIf YES: We can proceed with these values")
    print("If NO: You need to get actual distance results from the portal")
    print("\n")
    
    return history


if __name__ == "__main__":
    main()
