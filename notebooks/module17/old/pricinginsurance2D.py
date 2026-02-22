import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

# 1. SETUP: The Hidden Target
# We want the optimizer to find this specific coordinate without knowing it beforehand.
TRUE_OPTIMUM = np.array([0.156019, 0.598658])

def get_distance(x):
    """Calculate how far a point x is from the target."""
    return np.linalg.norm(x - TRUE_OPTIMUM)

# 2. UCB ACQUISITION FUNCTION
def ucb_acquisition(model, X_candidates, kappa=2.0):
    """
    Upper Confidence Bound (UCB).
    Score = Mean (Exploitation) + Kappa * Uncertainty (Exploration)
    """
    mu, sigma = model.predict(X_candidates, return_std=True)
    # We use negative distance as the target (max is 0), so we add kappa * sigma
    return mu + kappa * sigma

# 3. RUN OPTIMIZATION
np.random.seed(50) # Seed 50 was used for the specific path in the example
bounds = np.array([[0, 1], [0, 1]])

# Initial: Start with 3 random observations
X_train = np.random.uniform(0, 1, size=(3, 2))
# The GP tries to maximize, so we use negative distance (closer is less negative)
y_train = np.array([-get_distance(x) for x in X_train])

path_x, path_y = [], []
print(f"{'Step':<5} | {'Distance to Target'}")
print("-" * 30)

# Run 15 Optimization Steps
for i in range(15):
    # A. Fit the Surrogate Model (Gaussian Process)
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
    gp.fit(X_train, y_train)
    
    # B. Generate random candidate points to check
    X_candidates = np.random.uniform(0, 1, size=(2000, 2))
    
    # C. Select the best point using UCB
    ucb_scores = ucb_acquisition(gp, X_candidates, kappa=2.0)
    next_x = X_candidates[np.argmax(ucb_scores)]
    
    # D. Evaluate the real distance (The "Cost")
    real_dist = get_distance(next_x)
    
    # E. Update the training data
    X_train = np.vstack([X_train, next_x])
    y_train = np.append(y_train, -real_dist)
    
    # Track path for plotting
    path_x.append(next_x[0])
    path_y.append(next_x[1])
    
    print(f"{i+1:<5} | {real_dist:.4f}")

# 4. PLOTTING
plt.figure(figsize=(10, 8))

# Plot the Hidden Target
plt.scatter(TRUE_OPTIMUM[0], TRUE_OPTIMUM[1], c='red', s=300, marker='*', label='Hidden Target', zorder=10)

# Plot the Optimization Path
plt.plot(path_x, path_y, c='gray', linestyle='--', alpha=0.3)
sc = plt.scatter(path_x, path_y, c=range(len(path_x)), cmap='viridis', s=120, edgecolors='k', zorder=5, label='Search Steps')

# Annotations
plt.annotate('Start', (path_x[0], path_y[0]), xytext=(path_x[0]-0.15, path_y[0]-0.1), 
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('End', (path_x[-1], path_y[-1]), xytext=(path_x[-1]+0.1, path_y[-1]-0.1), 
             arrowprops=dict(facecolor='red', shrink=0.05))

plt.title(f"Bayesian Optimization: Exploration vs Exploitation (Kappa=2.0)\nTarget: {TRUE_OPTIMUM}", fontsize=14)
plt.xlabel("X Dimension")
plt.ylabel("Y Dimension")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.colorbar(sc, label='Step Number (Yellow = Later)')
plt.show()