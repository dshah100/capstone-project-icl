"""
Import Previous Results Tool
=============================
Imports data from your previous BBO runs and converts it to the format
needed for Round 6 query generation.
"""

import json
import numpy as np
import os
from pathlib import Path


def import_from_separate_files(inputs_file, outputs_file):
    """
    Import from separate inputs and outputs files.
    
    Expected format:
    - inputs: One line per query (space or comma separated values)
    - outputs: One distance value per line
    """
    print("\n" + "="*70)
    print("IMPORTING FROM SEPARATE FILES")
    print("="*70)
    
    # Check files exist
    if not os.path.exists(inputs_file):
        print(f"\n✗ Error: Cannot find {inputs_file}")
        return None
    
    if not os.path.exists(outputs_file):
        print(f"\n✗ Error: Cannot find {outputs_file}")
        return None
    
    # Read inputs
    print(f"\nReading inputs from: {inputs_file}")
    with open(inputs_file, 'r') as f:
        input_lines = [line.strip() for line in f if line.strip()]
    
    # Read outputs
    print(f"Reading outputs from: {outputs_file}")
    with open(outputs_file, 'r') as f:
        output_lines = [line.strip() for line in f if line.strip()]
    
    if len(input_lines) != len(output_lines):
        print(f"\n⚠ Warning: {len(input_lines)} inputs but {len(output_lines)} outputs")
        print("They should match. Using minimum of both.")
        n_lines = min(len(input_lines), len(output_lines))
        input_lines = input_lines[:n_lines]
        output_lines = output_lines[:n_lines]
    
    print(f"\nFound {len(input_lines)} query-result pairs")
    
    # Show sample
    print("\nSample (first 3 lines):")
    for i in range(min(3, len(input_lines))):
        print(f"  Input:  {input_lines[i]}")
        print(f"  Output: {output_lines[i]}")
        print()
    
    return input_lines, output_lines


def parse_queries(input_lines):
    """
    Parse query lines into function groups based on dimensions.
    """
    print("\n" + "="*70)
    print("PARSING QUERIES")
    print("="*70)
    
    queries = []
    
    for line in input_lines:
        # Try different separators
        if '-' in line:
            parts = line.split('-')
        elif ',' in line:
            parts = line.split(',')
        else:
            parts = line.split()
        
        # Convert to floats
        try:
            query = [float(p.strip()) for p in parts if p.strip()]
            queries.append(query)
        except ValueError as e:
            print(f"⚠ Warning: Could not parse line: {line}")
            continue
    
    # Group by dimensions
    by_dimension = {}
    for query in queries:
        dim = len(query)
        if dim not in by_dimension:
            by_dimension[dim] = []
        by_dimension[dim].append(query)
    
    print("\nQueries grouped by dimension:")
    for dim in sorted(by_dimension.keys()):
        print(f"  {dim}D: {len(by_dimension[dim])} queries")
    
    return queries


def organize_by_function(queries, results):
    """
    Organize queries and results by function ID based on dimensions.
    """
    print("\n" + "="*70)
    print("ORGANIZING BY FUNCTION")
    print("="*70)
    
    # Expected dimensions for each function
    function_dims = {
        1: 2, 2: 3, 3: 4, 4: 5,
        5: 6, 6: 7, 7: 8, 8: 9
    }
    
    # Reverse mapping: dimension -> function_id
    dim_to_func = {v: k for k, v in function_dims.items()}
    
    # Group data
    history = {f"function_{i}": [] for i in range(1, 9)}
    
    for query, result in zip(queries, results):
        dim = len(query)
        
        if dim in dim_to_func:
            func_id = dim_to_func[dim]
            
            # Determine round number (count existing + 1)
            round_num = len(history[f"function_{func_id}"]) + 1
            
            history[f"function_{func_id}"].append({
                "round": round_num,
                "query": [float(f"{v:.6f}") for v in query],
                "result": float(result)
            })
        else:
            print(f"⚠ Warning: Unexpected dimension {dim}D - skipping query")
    
    # Summary
    print("\nData organized by function:")
    for func_id in range(1, 9):
        key = f"function_{func_id}"
        n_obs = len(history[key])
        if n_obs > 0:
            best_result = min([obs['result'] for obs in history[key]])
            print(f"  Function {func_id} ({function_dims[func_id]}D): "
                  f"{n_obs} observations, best distance: {best_result:.6f}")
        else:
            print(f"  Function {func_id} ({function_dims[func_id]}D): No data")
    
    return history


def auto_detect_format(filepath):
    """
    Auto-detect the format of the data file.
    """
    print(f"\nAuto-detecting format of: {filepath}")
    
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        second_line = f.readline().strip() if f else ""
    
    print(f"First line: {first_line[:80]}...")
    
    # Check for common formats
    if first_line.startswith('{') or first_line.startswith('['):
        return "json"
    elif '\t' in first_line:
        return "tsv"
    elif ',' in first_line and not '-' in first_line:
        return "csv"
    elif '-' in first_line:
        return "portal_format"
    else:
        return "space_separated"


def import_from_directory(directory_path):
    """
    Import data from a directory containing inputs and outputs files.
    """
    print("\n" + "="*70)
    print("IMPORTING FROM DIRECTORY")
    print("="*70)
    print(f"\nDirectory: {directory_path}")
    
    # Look for inputs/outputs files with various names
    possible_input_names = ['inputs', 'inputs.txt', 'queries', 'queries.txt', 'input.txt']
    possible_output_names = ['outputs', 'outputs.txt', 'results', 'results.txt', 'output.txt', 'distances.txt']
    
    inputs_file = None
    outputs_file = None
    
    for name in possible_input_names:
        path = os.path.join(directory_path, name)
        if os.path.exists(path):
            inputs_file = path
            print(f"✓ Found inputs: {name}")
            break
    
    for name in possible_output_names:
        path = os.path.join(directory_path, name)
        if os.path.exists(path):
            outputs_file = path
            print(f"✓ Found outputs: {name}")
            break
    
    if not inputs_file:
        print("\n✗ Could not find inputs file")
        print(f"   Looked for: {', '.join(possible_input_names)}")
        return None
    
    if not outputs_file:
        print("\n✗ Could not find outputs file")
        print(f"   Looked for: {', '.join(possible_output_names)}")
        return None
    
    # Import data
    data = import_from_separate_files(inputs_file, outputs_file)
    if not data:
        return None
    
    input_lines, output_lines = data
    
    # Parse
    queries = parse_queries(input_lines)
    
    # Convert outputs to floats
    results = []
    for line in output_lines:
        try:
            results.append(float(line.strip()))
        except ValueError:
            print(f"⚠ Warning: Could not parse result: {line}")
            results.append(0.0)  # Placeholder
    
    # Organize
    history = organize_by_function(queries, results)
    
    return history


def save_to_query_history(history, output_file='query_history.json'):
    """
    Save imported data to query_history.json format.
    """
    print("\n" + "="*70)
    print("SAVING TO QUERY HISTORY")
    print("="*70)
    
    with open(output_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Saved to: {output_file}")
    print("\nYou can now run:")
    print("  python quick_round6_generator.py")
    print("or")
    print("  python capstone_round6.py")
    
    return output_file


def interactive_import():
    """
    Interactive mode to help user import their data.
    """
    print("\n" + "╔"+"═"*68+"╗")
    print("║" + " "*20 + "IMPORT PREVIOUS RESULTS" + " "*25 + "║")
    print("╚"+"═"*68+"╝")
    
    print("\nThis tool helps you import data from your previous BBO runs.")
    print("\nWhat format is your data in?\n")
    print("1. Separate 'inputs' and 'outputs' files in a folder")
    print("2. Single combined file")
    print("3. Portal copy-paste format")
    print("4. Let me browse for the files")
    print("5. Exit")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == '1':
        directory = input("\nEnter the path to the folder: ").strip().strip('"')
        if not directory:
            directory = './results'
        
        if not os.path.exists(directory):
            print(f"\n✗ Directory not found: {directory}")
            print("\nCurrent directory:", os.getcwd())
            print("\nAvailable directories:")
            for item in os.listdir('.'):
                if os.path.isdir(item):
                    print(f"  - {item}")
            return None
        
        history = import_from_directory(directory)
        
        if history:
            save_to_query_history(history)
            return history
    
    elif choice == '2':
        print("\nSingle file import (coming soon)")
        print("For now, please split into separate inputs and outputs files")
    
    elif choice == '3':
        print("\nPortal format import (coming soon)")
        print("For now, please use Option 1")
    
    elif choice == '4':
        print("\nCurrent directory:", os.getcwd())
        print("\nAvailable files:")
        for item in os.listdir('.'):
            if os.path.isfile(item):
                print(f"  - {item}")
        
        print("\nAvailable directories:")
        for item in os.listdir('.'):
            if os.path.isdir(item):
                print(f"  - {item}/")
        
        directory = input("\nEnter directory path: ").strip().strip('"')
        if directory and os.path.exists(directory):
            history = import_from_directory(directory)
            if history:
                save_to_query_history(history)
                return history
    
    elif choice == '5':
        print("\nExiting...")
        return None
    
    else:
        print("\n✗ Invalid option")
        return None


def main():
    """Main function."""
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode
        directory = sys.argv[1]
        history = import_from_directory(directory)
        if history:
            save_to_query_history(history)
    else:
        # Interactive mode
        interactive_import()


if __name__ == "__main__":
    main()
