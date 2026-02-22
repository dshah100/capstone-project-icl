"""
BBO Capstone - Round 6 Master Script
=====================================
Simple menu interface to access all Round 6 tools.
"""

import os
import sys


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print the main header."""
    print("\n" + "╔"+"═"*68+"╗")
    print("║" + " "*15 + "BBO CAPSTONE - ROUND 6 TOOLS" + " "*25 + "║")
    print("╚"+"═"*68+"╝")


def print_menu():
    """Print the main menu."""
    print("\n" + "="*70)
    print("MAIN MENU")
    print("="*70)
    print("""
1. 🎬 Run Demo (First-time users - see how everything works)
2. 📝 Enter Data & Generate Queries (Main workflow)
3. ✅ Validate Submission (Check your work before submitting)
4. 📊 View Progress (Visualize your optimization)
5. 📚 View Documentation (Open README)
6. 🔧 Advanced Options (Custom strategies)
7. ❓ Help & FAQ
8. 🚪 Exit

""")


def run_demo():
    """Run the demonstration workflow."""
    print("\n" + "="*70)
    print("RUNNING DEMONSTRATION")
    print("="*70)
    print("\nThis will show you the complete workflow with mock data.")
    print("Perfect for first-time users to understand the process.\n")
    
    confirm = input("Continue with demo? (y/n): ").lower().strip()
    if confirm == 'y':
        os.system('python demo_workflow.py')
    else:
        print("Demo cancelled.")
    
    input("\nPress Enter to return to menu...")


def generate_queries():
    """Run the quick query generator."""
    print("\n" + "="*70)
    print("GENERATE ROUND 6 QUERIES")
    print("="*70)
    print("\nThis will guide you through entering your portal results")
    print("and generate optimized Round 6 queries.\n")
    
    if not os.path.exists('query_history.json'):
        print("⚠ No query_history.json found.")
        print("This tool will help you create it.\n")
    else:
        print("✓ Found existing query_history.json")
        print("You can update it or create a new one.\n")
    
    confirm = input("Continue? (y/n): ").lower().strip()
    if confirm == 'y':
        os.system('python quick_round6_generator.py')
    else:
        print("Cancelled.")
    
    input("\nPress Enter to return to menu...")


def validate_submission():
    """Run the validation script."""
    print("\n" + "="*70)
    print("VALIDATE SUBMISSION")
    print("="*70)
    print("\nThis will check your queries and reflection for common errors")
    print("before you submit to the portal.\n")
    
    if not os.path.exists('round_6_portal_submission.txt'):
        print("⚠ No round_6_portal_submission.txt found.")
        print("Please generate queries first (Option 2).\n")
        input("Press Enter to return to menu...")
        return
    
    os.system('python validate_submission.py')
    input("\nPress Enter to return to menu...")


def view_progress():
    """Visualize optimization progress."""
    print("\n" + "="*70)
    print("VIEW OPTIMIZATION PROGRESS")
    print("="*70)
    
    if not os.path.exists('query_history.json'):
        print("\n⚠ No query_history.json found.")
        print("Please enter your data first (Option 2).\n")
        input("Press Enter to return to menu...")
        return
    
    print("\nGenerating progress visualizations for all functions...")
    
    try:
        from bbo_capstone_framework import BBOCapstoneManager
        import matplotlib.pyplot as plt
        
        manager = BBOCapstoneManager()
        manager.load_query_history_from_file('query_history.json')
        
        for func_id in range(1, 9):
            try:
                manager.visualize_progress(func_id)
            except Exception as e:
                print(f"Could not visualize Function {func_id}: {e}")
        
        print("\n✓ Visualizations complete!")
        print("Check the generated .png files.\n")
    except Exception as e:
        print(f"\n✗ Error: {e}\n")
    
    input("Press Enter to return to menu...")


def view_documentation():
    """Open the README."""
    print("\n" + "="*70)
    print("DOCUMENTATION")
    print("="*70)
    print("""
Available documentation files:

1. README_ROUND6.md         - Complete guide for Round 6
2. round_6_reflection.md    - Reflection template with examples
3. This script's docstrings - Technical details

""")
    
    choice = input("Open README_ROUND6.md in text editor? (y/n): ").lower().strip()
    if choice == 'y':
        if os.path.exists('README_ROUND6.md'):
            # Try different text editors
            if os.name == 'nt':  # Windows
                os.system('notepad README_ROUND6.md')
            else:  # Unix/Linux/Mac
                os.system('less README_ROUND6.md || cat README_ROUND6.md')
        else:
            print("\n⚠ README_ROUND6.md not found.")
    
    input("\nPress Enter to return to menu...")


def advanced_options():
    """Advanced optimization options."""
    print("\n" + "="*70)
    print("ADVANCED OPTIONS")
    print("="*70)
    print("""
1. Compare acquisition strategies (EI vs UCB)
2. Test different exploration parameters
3. Generate queries for specific functions only
4. Export data for external analysis
5. Back to main menu

""")
    
    choice = input("Select option (1-5): ").strip()
    
    if choice == '1':
        compare_strategies()
    elif choice == '2':
        test_parameters()
    elif choice == '3':
        generate_specific_functions()
    elif choice == '4':
        export_data()
    elif choice == '5':
        return
    else:
        print("Invalid option.")
    
    input("\nPress Enter to return to menu...")


def compare_strategies():
    """Compare different acquisition strategies."""
    print("\nComparing EI vs UCB strategies...")
    
    if not os.path.exists('query_history.json'):
        print("⚠ Need query_history.json first.")
        return
    
    try:
        from bbo_capstone_framework import BBOCapstoneManager
        
        manager = BBOCapstoneManager()
        manager.load_query_history_from_file('query_history.json')
        
        func_id = int(input("\nEnter function ID (1-8): "))
        if func_id not in range(1, 9):
            print("Invalid function ID.")
            return
        
        print(f"\nFunction {func_id} - Strategy Comparison:\n")
        
        print("Expected Improvement (EI):")
        for xi in [0.005, 0.01, 0.02]:
            query = manager.optimizers[func_id].propose_next_query(
                acquisition='ei',
                xi=xi
            )
            formatted = manager.optimizers[func_id].format_query(query)
            print(f"  xi={xi:0.3f}: {formatted}")
        
        print("\nUpper Confidence Bound (UCB):")
        for kappa in [1.5, 2.0, 3.0]:
            query = manager.optimizers[func_id].propose_next_query(
                acquisition='ucb',
                kappa=kappa
            )
            formatted = manager.optimizers[func_id].format_query(query)
            print(f"  kappa={kappa}: {formatted}")
        
        print("\nInterpretation:")
        print("  • Lower parameters = More exploitation")
        print("  • Higher parameters = More exploration")
        
    except Exception as e:
        print(f"Error: {e}")


def test_parameters():
    """Test different parameter values."""
    print("\nTesting exploration parameters...")
    print("(Feature coming soon - use Python directly for now)")


def generate_specific_functions():
    """Generate queries for specific functions only."""
    print("\nGenerate queries for specific functions...")
    print("(Feature coming soon - use Python directly for now)")


def export_data():
    """Export data for external analysis."""
    print("\nExporting data...")
    
    if not os.path.exists('query_history.json'):
        print("⚠ No data to export.")
        return
    
    try:
        import json
        import csv
        
        with open('query_history.json', 'r') as f:
            history = json.load(f)
        
        # Export to CSV
        with open('query_history_export.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Function', 'Round', 'Query', 'Result'])
            
            for func_key, observations in history.items():
                func_id = func_key.split('_')[1]
                for obs in observations:
                    query_str = '-'.join([f"{v:.6f}" for v in obs['query']])
                    writer.writerow([func_id, obs.get('round', '?'), query_str, obs['result']])
        
        print("✓ Exported to: query_history_export.csv")
        
    except Exception as e:
        print(f"Error: {e}")


def show_help():
    """Show help and FAQ."""
    print("\n" + "="*70)
    print("HELP & FAQ")
    print("="*70)
    print("""
QUICK START:
  First time? Run Option 1 (Demo) to see how everything works.
  Have portal results? Run Option 2 to generate Round 6 queries.
  Before submitting? Run Option 3 to validate your work.

COMMON QUESTIONS:

Q: I don't have results from Rounds 1-5 yet. What should I do?
A: Option 1 (Demo) will show you the process with mock data.
   Then submit your initial queries from 1st.log to the portal.

Q: What format do queries need to be in?
A: Format: x1-x2-x3-...-xn
   Each value: 0.XXXXXX (starts with 0, 6 decimal places)
   Example: 0.123456-0.789012

Q: How do I know if my queries are good?
A: Run Option 3 (Validate) to check format.
   Run Option 4 (View Progress) to see if you're converging.
   Distance should generally decrease over rounds.

Q: What's the difference between EI and UCB?
A: Both balance exploration vs exploitation.
   EI (Expected Improvement): Better for limited budgets
   UCB (Upper Confidence Bound): More aggressive exploration
   We recommend EI for Round 6.

Q: My reflection is too long. How do I trim it?
A: Focus on specific examples from YOUR data
   Remove generic CNN descriptions
   Keep under 700 words total

Q: Where do I submit?
A: Queries → Capstone portal (link from Module 12 email)
   Reflection → Discussion board (< 700 words)

Q: Can I change my queries after submission?
A: Check with your instructor. Usually submissions are final.

Q: What if my distance isn't improving?
A: Normal! Some functions are harder than others.
   Discuss challenges in your reflection.
   Try increasing exploration parameter (xi or kappa).

NEED MORE HELP?
  • Read README_ROUND6.md for detailed instructions
  • Check round_6_reflection.md for examples
  • Ask on the discussion board
  • Review the demo output

""")
    
    input("Press Enter to return to menu...")


def main():
    """Main menu loop."""
    while True:
        clear_screen()
        print_header()
        print_menu()
        
        choice = input("Select an option (1-8): ").strip()
        
        if choice == '1':
            run_demo()
        elif choice == '2':
            generate_queries()
        elif choice == '3':
            validate_submission()
        elif choice == '4':
            view_progress()
        elif choice == '5':
            view_documentation()
        elif choice == '6':
            advanced_options()
        elif choice == '7':
            show_help()
        elif choice == '8':
            print("\n✓ Good luck with your Round 6 submission!")
            print("Remember to:")
            print("  1. Submit queries to portal")
            print("  2. Post reflection to discussion board")
            print("  3. Engage with peers")
            print("\n")
            break
        else:
            print("\n⚠ Invalid option. Please select 1-8.")
            input("Press Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Exiting...")
        sys.exit(0)
