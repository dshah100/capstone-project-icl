import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# 1. SETUP: The Hidden 3D Target
# Using the values from your "Function 3"
TRUE_OPTIMUM = np.array([0.058084,0.155995,0.601115])

def get_distance(x):
    """Calculate Euclidean distance in 3D space."""
    return np.linalg.norm(x - TRUE_OPTIMUM)

# 2. UCB ACQUISITION FUNCTION
def ucb_acquisition(model, X_candidates, kappa=2.0):
    mu, sigma = model.predict(X_candidates, return_std=True)
    return mu + kappa * sigma

# 3. RUN OPTIMIZATION
np.random.seed(42)
# Bounds for 3 dimensions
bounds = np.array([[0, 1], [0, 1], [0, 1]])

# Initial: Start with 3 random observations in 3D
X_train = np.random.uniform(0, 1, size=(3, 3))
y_train = np.array([-get_distance(x) for x in X_train])

path_x, path_y, path_z = [], [], []

print(f"{'Step':<5} | {'Distance to Target'}")
print("-" * 30)

# Run 20 Optimization Steps
for i in range(20):
    # Fit GP
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
    gp.fit(X_train, y_train)
    
    # Generate random 3D candidates
    X_candidates = np.random.uniform(0, 1, size=(5000, 3))
    
    # UCB Selection
    ucb_scores = ucb_acquisition(gp, X_candidates, kappa=2.0)
    next_x = X_candidates[np.argmax(ucb_scores)]
    
    # Evaluate
    real_dist = get_distance(next_x)
    
    # Update
    X_train = np.vstack([X_train, next_x])
    y_train = np.append(y_train, -real_dist)
    
    # Track path
    path_x.append(next_x[0])
    path_y.append(next_x[1])
    path_z.append(next_x[2])
    
    print(f"{i+1:<5} | {real_dist:.4f}")

# 4. 3D VISUALIZATION
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the Hidden Target (Red Star)
ax.scatter(TRUE_OPTIMUM[0], TRUE_OPTIMUM[1], TRUE_OPTIMUM[2], 
           c='red', s=400, marker='*', label=f'Target {TRUE_OPTIMUM}', zorder=10)

# Plot the Optimization Path (Line)
ax.plot(path_x, path_y, path_z, c='gray', linestyle='--', alpha=0.4, label='Path')

# Plot the Steps (Colored Dots)
sc = ax.scatter(path_x, path_y, path_z, c=range(len(path_x)), cmap='viridis', s=100, edgecolors='k', depthshade=False, label='Steps')

# Labels and View
ax.set_title(f"3D Bayesian Optimization (Hunting in 3 Dimensions)", fontsize=14)
ax.set_xlabel("Factor 1 (X)")
ax.set_ylabel("Factor 2 (Y)")
ax.set_zlabel("Factor 3 (Z)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.legend()

# Add colorbar
cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label('Step Number (Yellow = Later)')

plt.show()