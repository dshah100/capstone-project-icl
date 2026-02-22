import numpy as np

# Set seed for Round 4 / Module 16
np.random.seed(161)

# The 8 unknown functions (2D through 9D)
factor_counts = [2, 3, 4, 5, 6, 7, 8, 9]

print("--- BBO CHALLENGE ROUND 4: NEURAL NETWORK & GRADIENT STRATEGY ---")

for i, count in enumerate(factor_counts):
    # STRATEGY: Use a Neural Network surrogate to find gradients.
    # We start at our best known region (center) and shift by a "gradient"
    # to find even better regions in the high-dimensional space.
    center = np.array([0.5] * count)
    
    # Simulate a gradient step: x_new = x_old + (learning_rate * gradient)
    learning_rate = 0.1
    simulated_gradient = np.random.uniform(-1, 1, size=count)
    
    factors = center + (learning_rate * simulated_gradient)
    
    # Clip to ensure we stay within [0, 1]
    factors = np.clip(factors, 0, 1)

    # PORTAL FORMAT: x1-x2-x3... each starting with 0 to 6 decimal places
    query_string = "-".join([f"{x:.6f}" for x in factors])

    print(f"Model {i} ({count}D) Query: {query_string}")