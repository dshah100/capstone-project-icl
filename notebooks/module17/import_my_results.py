"""
Import Actual Results from SalesforceImperialCourse1 Directory
===============================================================
Converts your run logs and outputs into query_history.json
"""

import json
import re
import numpy as np


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
    
    # Pattern 2: "pricing_model_version_X (YD):" format (Round 2)
    pattern2 = r'pricing_model_version_(\d+) \((\d+)D\): ([0-9.-]+)'
    matches2 = re.findall(pattern2, content)
    
    if matches2:
        for model_id, dims, query_str in matches2:
            func_id = int(model_id) + 1  # version_0 = Function 1
            query = [float(x) for x in query_str.split('-')]
            queries[func_id] = query
        return queries
    
    # Pattern 3: "Model X (YD) Query:" format (Round 3)
    pattern3 = r'Model (\d+) \((\d+)D\) Query: ([0-9.-]+)'
    matches3 = re.findall(pattern3, content)
    
    if matches3:
        for model_id, dims, query_str in matches3:
            func_id = int(model_id) + 1  # Model 0 = Function 1
            query = [float(x) for x in query_str.split('-')]
            queries[func_id] = query
        return queries
    
    return queries


def parse_outputs_python_format(filepath):
    """Parse outputs file in Python array format."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract the list of values
    # The format is: [np.float64(value1), np.float64(value2), ...]
    pattern = r'np\.float64\(([0-9.e+-]+)\)'
    matches = re.findall(pattern, content)
    
    if matches:
        results = [float(x) for x in matches]
        return results
    
    return []


def convert_output_to_distance(output_value):
    """
    Convert output value to distance.
    The outputs appear to be negative log-likelihood or similar.
    We need actual distances from the portal.
    
    For now, we'll use absolute value as placeholder.
    YOU SHOULD REPLACE THIS WITH ACTUAL PORTAL RESULTS!
    """
    # These are NOT real distances - they're placeholders!
    # The actual portal returns positive distance values
    return abs(output_value)


def create_query_history():
    """Create query_history.json from your results files."""
    
    print("\n" + "="*70)
    print("IMPORTING YOUR RESULTS")
    print("="*70)
    
    base_path = r"C:\work\AICourse\SalesforceImperialCourse1\capstone-project-icl\results"
    
    # Parse run logs
    run_files = [
        (1, f"{base_path}\\1stRun.log"),
        (2, f"{base_path}\\2ndRun.txt"),
        (3, f"{base_path}\\3rdRun.txt"),
        (4, f"{base_path}\\4thRun.log"),
        (5, f"{base_path}\\5thRun.txt"),
    ]
    
    # Parse output files (only available for rounds 1-3)
    output_files = [
        (1, f"{base_path}\\outputs_1.txt"),
        (2, f"{base_path}\\outputs_2.txt"),
        (3, f"{base_path}\\outputs_3.txt"),
    ]
    
    # Initialize history
    history = {f"function_{i}": [] for i in range(1, 9)}
    
    # Process each round
    for round_num, run_file in run_files:
        print(f"\nProcessing Round {round_num}...")
        
        # Parse queries
        queries = parse_run_log(run_file)
        print(f"  Found {len(queries)} queries")
        
        # Parse outputs for this round
        output_file = [f for r, f in output_files if r == round_num]
        if output_file:
            results = parse_outputs_python_format(output_file[0])
            print(f"  Found {len(results)} results")
        else:
            results = []
            print(f"  ⚠ No output file found (need portal results for Round {round_num})")
        
        # Match queries with results
        for func_id in range(1, 9):
            if func_id in queries:
                query = queries[func_id]
                
                # Get result (if available)
                if func_id - 1 < len(results):
                    result_value = results[func_id - 1]
                    distance = convert_output_to_distance(result_value)
                else:
                    distance = 0.0  # Placeholder
                
                history[f"function_{func_id}"].append({
                    "round": round_num,
                    "query": [float(f"{v:.6f}") for v in query],
                    "result": float(f"{distance:.6f}")
                })
                
                print(f"    Function {func_id}: {len(query)}D, distance={distance:.6f}")
    
    return history


def main():
    """Main function."""
    print("\n" + "╔"+"═"*68+"╗")
    print("║" + " "*15 + "IMPORTING YOUR BBO RESULTS" + " "*27 + "║")
    print("╚"+"═"*68+"╝")
    
    print("\n⚠ IMPORTANT NOTE:")
    print("The output files contain transformed values (possibly negative log-likelihood).")
    print("These are NOT the actual distances from the portal.")
    print("\nYou need to:")
    print("  1. Get the actual distance values from the capstone portal")
    print("  2. Replace the 'result' values in query_history.json")
    print("  3. Then run Round 6 query generation")
    
    input("\nPress Enter to continue with import (using placeholder distances)...")
    
    # Create history
    history = create_query_history()
    
    # Save to file
    output_file = 'query_history.json'
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
            print(f"  Function {func_id} ({dims}D): {n_obs} observations")
        else:
            print(f"  Function {func_id}: No data")
    
    print("\n" + "="*70)
    print("⚠ CRITICAL NEXT STEP")
    print("="*70)
    print("\nThe 'result' values in query_history.json are PLACEHOLDERS!")
    print("\nYou MUST:")
    print("  1. Go to the capstone portal")
    print("  2. Look up the actual distance results for your queries")
    print("  3. Edit query_history.json and replace the 'result' values")
    print("  4. Then run: python quick_round6_generator.py")
    
    print("\nExample of what to fix in query_history.json:")
    print('  "result": 1.568616  ← Replace with actual portal distance')
    
    print("\n")


if __name__ == "__main__":
    main()
