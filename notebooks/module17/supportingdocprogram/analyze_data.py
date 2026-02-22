"""
Analyze Your Imported Data
===========================
Check if the values make sense as BBO distances.
"""

import json
import numpy as np


def analyze_data():
    print("\n" + "╔"+"═"*68+"╗")
    print("║" + " "*20 + "DATA ANALYSIS" + " "*35 + "║")
    print("╚"+"═"*68+"╝")
    
    with open('query_history_complete.json', 'r') as f:
        history = json.load(f)
    
    print("\nAnalyzing imported results...\n")
    print("="*70)
    
    for func_id in range(1, 9):
        key = f"function_{func_id}"
        observations = history[key]
        
        if not observations:
            continue
        
        dims = len(observations[0]['query'])
        results = [obs['result'] for obs in observations]
        
        min_result = min(results)
        max_result = max(results)
        avg_result = np.mean(results)
        
        # Check for improvement trend
        first_result = results[0]
        last_result = results[-1]
        improving = last_result < first_result
        
        print(f"\nFunction {func_id} ({dims}D):")
        print(f"  Range: [{min_result:.6e}, {max_result:.6e}]")
        print(f"  Average: {avg_result:.6e}")
        print(f"  Trend: Round 1: {first_result:.6e} → Round 5: {last_result:.6e}")
        
        # Assess the values
        if max_result < 1e-10:
            print(f"  ⚠️  WARNING: Extremely small values (near zero)")
            print(f"      These don't look like typical distances")
        elif max_result > 50:
            print(f"  ⚠️  WARNING: Very large values")
            print(f"      Typical distances in [0,1]^{dims} are usually < 2√{dims} ≈ {2*np.sqrt(dims):.1f}")
        elif improving:
            print(f"  ✓ Good: Values decreasing (optimization improving)")
        else:
            print(f"  ⚠️  Values not consistently decreasing")
    
    print("\n" + "="*70)
    print("ASSESSMENT")
    print("="*70)
    
    print("\n📊 Your data shows unusual patterns:")
    print("\n  Function 1: Extremely small values (10^-117 to 10^-15)")
    print("              → Essentially zero, suggesting perfect or near-perfect solutions")
    print("\n  Function 2-3: Small values (0.02 to 0.59)")
    print("              → Reasonable distances, good progress")
    print("\n  Function 4: Moderate values (3 to 21)")
    print("              → Could be valid, but high for 5D space")
    print("\n  Function 5: Very large values (11 to 147!)")
    print("              → Extremely unusual for 6D space")
    print("              → Max possible distance in unit 6D hypercube ≈ √6 ≈ 2.45")
    print("\n  Function 6-8: Moderate values (0.002 to 9.4)")
    print("              → Reasonable range")
    
    print("\n" + "="*70)
    print("❓ CRITICAL QUESTION")
    print("="*70)
    
    print("\nAre these the ACTUAL distance values from the capstone portal?")
    print("\nOR are these transformed/processed values (like negative log-likelihood)?")
    print("\n" + "="*70)
    
    print("\n🔍 To verify:")
    print("  1. Log into the capstone portal")
    print("  2. Look at your Round 1, Function 1 submission:")
    print(f"     Query: [0.374546, 0.950714]")
    print(f"     Your file shows: {history['function_1'][0]['result']:.6e}")
    print("  3. Does the portal show this EXACT value?")
    print("     OR does it show a different distance value?")
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    print("\n✅ IF portal shows these exact values:")
    print("   → These ARE the correct distances")
    print("   → Proceed with Round 6 generation")
    print("   → The unusual values are part of your specific challenge")
    print("\n⚠️  IF portal shows different values:")
    print("   → You need to manually enter the correct portal distances")
    print("   → Edit query_history_complete.json")
    print("   → Replace the 'result' values with actual portal distances")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    
    print("\nOption A: These ARE the correct portal values")
    print("  → Rename query_history_complete.json to query_history.json")
    print("  → Run: python quick_round6_generator.py")
    print("\nOption B: Need to get actual portal values")
    print("  → Check portal for actual distance results")
    print("  → Edit query_history_complete.json with correct values")
    print("  → Then rename to query_history.json")
    print("  → Run: python quick_round6_generator.py")
    
    print("\n")


if __name__ == "__main__":
    analyze_data()
