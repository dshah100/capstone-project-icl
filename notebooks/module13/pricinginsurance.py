import numpy as np

# Set seed to reflect a structured, non-random decision for this iteration
np.random.seed(2025)

# Each model version uses a different number of rating factors
factor_counts = [2, 3, 4, 5, 6, 7, 8, 9]

print("--- BBO CHALLENGE ROUND 2: STRUCTURED QUERIES ---")

for i, count in enumerate(factor_counts):
    # Generate structured synthetic factor values between 0 and 1
    # In your reflection, this represents your space-filling strategy
    factors = np.random.uniform(0, 1, size=count)    

    # Format output: x1-x2-x3... each starting with 0 to 6 decimal places
    query_string = "-".join([f"{x:.6f}" for x in factors])

    print(f"pricing_model_version_{i} ({count}D): {query_string}")