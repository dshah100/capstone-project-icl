import numpy as np

# Each model version uses a different number of rating factors.
# For example: 2 = age & vehicle type, 3 = age/vehicle/location, etc.

factor_counts = [2, 3, 4, 5, 6, 7, 8,9]



# Dictionary to store synthetic factor values for each pricing model
pricing_inputs = {}

for i, count in enumerate(factor_counts):
   

    # Generate synthetic factor values between 0 and 1
    # (e.g., normalised risk scores, exposure metrics, claim likelihood features)

    factors = np.random.uniform(0, 1, size=count)    

    # Sort values for consistency and readability, round for clarity
    factors = np.round(np.sort(factors), 8)    

    # Store results under a clear model identifier
    pricing_inputs[f"pricing_model_version_{i}"] = factors


# Display the generated factor sets

for model_name, factors in pricing_inputs.items():    

    # Format output for easy comparison between models

    formatted_vals = " - ".join(map(str, factors))  

    print(f"{model_name}: {formatted_vals}")