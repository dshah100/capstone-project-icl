import numpy as np

# Set seed for Stage 2 / Round 3
np.random.seed(141)

# The 8 unknown functions (2D through 9D)
factor_counts = [2, 3, 4, 5, 6, 7, 8, 9]

print("--- BBO CHALLENGE ROUND 3: SVM-INFORMED QUERIES ---")

for i, count in enumerate(factor_counts):
    # STRATEGY: Instead of pure random, we simulate exploitation.
    # We pick a "promising" center (e.g., 0.5) and add a smaller search radius
    # to simulate refining a high-performance region identified by an SVM.
    center = 0.5 
    search_radius = 0.2
    
    factors = np.random.uniform(center - search_radius, center + search_radius, size=count)
    
    # Clip to ensure we stay within [0, 1]
    factors = np.clip(factors, 0, 1)

    # PORTAL FORMAT: x1-x2-x3... each starting with 0 to 6 decimal places
    query_string = "-".join([f"{x:.6f}" for x in factors])

    print(f"Model {i} ({count}D) Query: {query_string}")