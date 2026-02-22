"""
Quick Round 6 Query Generator
==============================
This script helps you quickly enter your portal results and generate Round 6 queries.
Run this script and follow the prompts.
"""

import numpy as np
from bbo_capstone_framework import BBOCapstoneManager
import json


def quick_data_entry():
    """
    Quick data entry interface for portal results.
    Guides you through entering results from previous rounds.
    """
    print("\n" + "="*70)
    print("QUICK DATA ENTRY FOR BBO CAPSTONE PROJECT")
    print("="*70)
    print("\nYou need to enter the RESULTS (distances) from the portal for your")
    print("previous queries. The initial queries are already known.\n")
    
    # Known initial queries from 1st.log
    initial_queries = {
        1: [0.374546, 0.950714],
        2: [0.156019, 0.598658],
        3: [0.058084, 0.155995, 0.601115],
        4: [0.020584, 0.212339, 0.708073, 0.832443],
        5: [0.181825, 0.183405, 0.291229, 0.304242],
        6: [0.139494, 0.199674, 0.292145, 0.366362, 0.456078],
        7: [0.046456, 0.065052, 0.170524, 0.514234, 0.592415, 0.607545],
        8: [0.034389, 0.097672, 0.122038, 0.304614, 0.440152, 0.495177, 0.684233, 0.808397]
    }
    
    history = {}
    
    for func_id in range(1, 9):
        print(f"\n{'─'*70}")
        print(f"FUNCTION {func_id} ({len(initial_queries[func_id])}D)")
        print(f"{'─'*70}")
        
        func_history = []
        
        # Round 1 (initial query)
        print(f"\nRound 1 - Initial Query: {initial_queries[func_id]}")
        
        # Ask if they have results or if they should skip
        use_mock = input("Do you have the actual result from the portal? (y/n): ").lower().strip()
        
        if use_mock == 'n':
            print("\nWARNING: You need portal results to generate accurate Round 6 queries!")
            print("I'll use mock data for demonstration, but you MUST replace with real data.")
            
            # Generate mock data for demonstration
            for round_num in range(1, 6):
                if round_num == 1:
                    query = initial_queries[func_id]
                else:
                    # Mock subsequent queries (you'll need real ones)
                    query = list(np.random.uniform(0, 1, size=len(initial_queries[func_id])))
                
                # Mock distance (decreasing trend for realism)
                mock_result = 1.0 / (round_num + 1) + np.random.uniform(0, 0.1)
                
                func_history.append({
                    "round": round_num,
                    "query": [float(f"{v:.6f}") for v in query],
                    "result": float(f"{mock_result:.6f}")
                })
            
            print(f"Generated {len(func_history)} mock observations for Function {func_id}")
        else:
            # Enter actual portal results
            n_rounds = 5
            print(f"\nYou need to enter results for {n_rounds} rounds.")
            print("(If you haven't done 5 rounds yet, enter what you have)\n")
            
            actual_rounds = int(input(f"How many rounds have you completed (1-5)? "))
            
            for round_num in range(1, actual_rounds + 1):
                print(f"\n  Round {round_num}:")
                
                if round_num == 1:
                    query = initial_queries[func_id]
                    print(f"    Query: {query}")
                else:
                    print(f"    Enter your Round {round_num} query (space-separated):")
                    query_str = input(f"    > ")
                    query = [float(x) for x in query_str.split()]
                
                result = float(input(f"    Distance result from portal: "))
                
                func_history.append({
                    "round": round_num,
                    "query": [float(f"{v:.6f}") for v in query],
                    "result": float(f"{result:.6f}")
                })
        
        history[f"function_{func_id}"] = func_history
    
    # Save history
    with open('query_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("✓ Data saved to query_history.json")
    print("="*70)
    
    return history


def generate_round_6_queries(use_existing_history=False):
    """Generate Round 6 queries using the entered data."""
    
    if not use_existing_history:
        # Fresh data entry
        history = quick_data_entry()
    else:
        # Load existing history
        try:
            with open('query_history.json', 'r') as f:
                history = json.load(f)
            print("✓ Loaded existing query_history.json")
        except FileNotFoundError:
            print("No existing history found. Starting data entry...")
            history = quick_data_entry()
    
    # Initialize manager
    print("\n" + "="*70)
    print("INITIALIZING BBO OPTIMIZER")
    print("="*70)
    
    manager = BBOCapstoneManager()
    
    # Load history
    manager.load_query_history_from_file('query_history.json')
    
    # Generate Round 6 queries with Expected Improvement
    print("\nGenerating Round 6 queries using Expected Improvement...")
    print("(Balancing exploration vs exploitation based on your data)\n")
    
    queries = manager.generate_round_queries(
        round_num=6,
        acquisition='ei',  # Expected Improvement
        xi=0.01  # Exploration parameter
    )
    
    # Save queries
    filepath = manager.save_queries_to_file(queries, round_num=6)
    
    # Also save in portal-ready format
    print("\n" + "="*70)
    print("PORTAL SUBMISSION FORMAT")
    print("="*70 + "\n")
    
    with open('round_6_portal_submission.txt', 'w') as f:
        f.write("Copy and paste these queries into the capstone portal:\n\n")
        for func_id, query in queries.items():
            output = f"Function {func_id}:\n{query}\n"
            print(output)
            f.write(output + "\n")
    
    print("\n✓ Saved to: round_6_portal_submission.txt")
    
    # Generate progress visualizations
    print("\n" + "="*70)
    print("GENERATING PROGRESS VISUALIZATIONS")
    print("="*70 + "\n")
    
    for func_id in [1, 2, 3]:  # Visualize first 3 functions
        try:
            manager.visualize_progress(func_id)
        except Exception as e:
            print(f"Could not visualize Function {func_id}: {e}")
    
    return queries


def main():
    """Main execution function."""
    print("\n" + "╔"+"═"*68+"╗")
    print("║" + " "*15 + "BBO CAPSTONE - ROUND 6 QUERY GENERATOR" + " "*15 + "║")
    print("╚"+"═"*68+"╝")
    
    print("\nThis script will help you:")
    print("  1. Enter your portal results from Rounds 1-5")
    print("  2. Generate optimized Round 6 queries")
    print("  3. Visualize your optimization progress")
    
    print("\n" + "─"*70)
    choice = input("\nDo you have an existing query_history.json file? (y/n): ").lower().strip()
    
    use_existing = (choice == 'y')
    
    queries = generate_round_6_queries(use_existing_history=use_existing)
    
    print("\n" + "="*70)
    print("✓ COMPLETE! Next steps:")
    print("="*70)
    print("\n1. Review round_6_portal_submission.txt")
    print("2. Submit queries to the capstone portal")
    print("3. When you get results, add them to query_history.json")
    print("4. Run this script again for Round 7")
    print("\n")


if __name__ == "__main__":
    main()
