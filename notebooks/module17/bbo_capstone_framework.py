"""
Black-Box Optimization Capstone Framework
==========================================
This framework manages query history, implements Bayesian Optimization
for all 8 functions, and generates the next round of queries.

Author: BBO Capstone Project
Date: February 6, 2026
Round: 6 (with 15 accumulated data points)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C
from scipy.optimize import differential_evolution
from scipy.stats import norm
import json
from datetime import datetime


class BBOOptimizer:
    """
    Bayesian Black-Box Optimizer for multi-dimensional functions.
    Uses Gaussian Process with Expected Improvement acquisition function.
    """
    
    def __init__(self, n_dimensions, function_id, bounds=None):
        """
        Initialize the optimizer.
        
        Parameters:
        -----------
        n_dimensions : int
            Number of input dimensions for the function
        function_id : int
            Function identifier (1-8)
        bounds : list of tuples, optional
            Bounds for each dimension. Defaults to [0, 1] for all dimensions
        """
        self.n_dimensions = n_dimensions
        self.function_id = function_id
        self.bounds = bounds if bounds else [(0, 1)] * n_dimensions
        
        # Query history
        self.X_observed = []
        self.y_observed = []
        
        # GP model
        self.gp = None
        
    def add_observation(self, x, y):
        """Add a new observation to the dataset."""
        self.X_observed.append(x)
        self.y_observed.append(y)
        
    def load_history(self, query_history):
        """
        Load query history from previous rounds.
        
        Parameters:
        -----------
        query_history : list of dict
            Each dict should have 'query' (list) and 'result' (float)
        """
        for obs in query_history:
            self.add_observation(obs['query'], obs['result'])
    
    def fit_gp(self):
        """Fit Gaussian Process to observed data."""
        if len(self.X_observed) < 2:
            raise ValueError("Need at least 2 observations to fit GP")
        
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        
        # Use Matern kernel with length scale optimization
        kernel = C(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(self.n_dimensions),
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5
        )
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10
        )
        
        self.gp.fit(X, y)
        
    def expected_improvement(self, X, xi=0.01):
        """
        Calculate Expected Improvement acquisition function.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_dimensions)
            Points at which to evaluate EI
        xi : float, optional
            Exploration parameter (higher = more exploration)
        
        Returns:
        --------
        ei : array-like, shape (n_samples,)
            Expected improvement at each point
        """
        if self.gp is None:
            raise ValueError("GP model not fitted. Call fit_gp() first.")
        
        mu, sigma = self.gp.predict(X, return_std=True)
        
        # Minimize distance (since we're finding minimum)
        # Best observed value
        y_best = np.min(self.y_observed)
        
        # Calculate improvement
        with np.errstate(divide='warn', invalid='warn'):
            improvement = y_best - mu - xi
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def ucb_acquisition(self, X, kappa=2.0):
        """
        Upper Confidence Bound acquisition function.
        
        Parameters:
        -----------
        X : array-like
            Candidate points
        kappa : float
            Exploration parameter (higher = more exploration)
        """
        if self.gp is None:
            raise ValueError("GP model not fitted. Call fit_gp() first.")
        
        mu, sigma = self.gp.predict(X, return_std=True)
        # For minimization, we use Lower Confidence Bound (LCB)
        return mu - kappa * sigma
    
    def propose_next_query(self, acquisition='ei', n_candidates=5000, xi=0.01, kappa=2.0):
        """
        Propose next query point using the specified acquisition function.
        
        Parameters:
        -----------
        acquisition : str
            'ei' for Expected Improvement or 'ucb' for Upper Confidence Bound
        n_candidates : int
            Number of random candidates to evaluate
        xi : float
            Exploration parameter for EI
        kappa : float
            Exploration parameter for UCB
        
        Returns:
        --------
        next_query : array
            Proposed next query point
        """
        self.fit_gp()
        
        # Generate random candidates
        X_candidates = np.random.uniform(
            low=[b[0] for b in self.bounds],
            high=[b[1] for b in self.bounds],
            size=(n_candidates, self.n_dimensions)
        )
        
        if acquisition == 'ei':
            scores = self.expected_improvement(X_candidates, xi=xi)
            # Maximize EI
            best_idx = np.argmax(scores)
        elif acquisition == 'ucb':
            scores = self.ucb_acquisition(X_candidates, kappa=kappa)
            # Minimize LCB
            best_idx = np.argmin(scores)
        else:
            raise ValueError("acquisition must be 'ei' or 'ucb'")
        
        next_query = X_candidates[best_idx]
        
        # Refine with local optimization
        if acquisition == 'ei':
            result = differential_evolution(
                lambda x: -self.expected_improvement(x.reshape(1, -1), xi=xi)[0],
                bounds=self.bounds,
                seed=42,
                maxiter=100,
                x0=next_query
            )
        else:
            result = differential_evolution(
                lambda x: self.ucb_acquisition(x.reshape(1, -1), kappa=kappa)[0],
                bounds=self.bounds,
                seed=42,
                maxiter=100,
                x0=next_query
            )
        
        return result.x
    
    def format_query(self, x):
        """Format query point for portal submission."""
        return '-'.join([f"{val:.6f}" for val in x])
    
    def get_best_observation(self):
        """Return the best observed point and its value."""
        if not self.y_observed:
            return None, None
        
        best_idx = np.argmin(self.y_observed)
        return self.X_observed[best_idx], self.y_observed[best_idx]


class BBOCapstoneManager:
    """Manages the entire capstone project with 8 functions."""
    
    def __init__(self):
        # Function dimensions based on the initial queries
        self.function_dims = {
            1: 2,  # Function 1: 2D
            2: 3,  # Function 2: 3D  (note: initial had 2, but version_1 suggests 3)
            3: 4,  # Function 3: 4D
            4: 5,  # Function 4: 5D (note: initial had 4)
            5: 6,  # Function 5: 6D (note: initial had 4)
            6: 7,  # Function 6: 7D (note: initial had 5)
            7: 8,  # Function 7: 8D (note: initial had 6)
            8: 9   # Function 8: 9D (note: initial had 8)
        }
        
        self.optimizers = {}
        for func_id, dims in self.function_dims.items():
            self.optimizers[func_id] = BBOOptimizer(dims, func_id)
    
    def load_query_history_from_file(self, filepath):
        """Load query history from JSON file."""
        with open(filepath, 'r') as f:
            history = json.load(f)
        
        for func_id, queries in history.items():
            func_id_int = int(func_id.split('_')[1])
            for query_data in queries:
                self.optimizers[func_id_int].add_observation(
                    query_data['query'],
                    query_data['result']
                )
    
    def generate_round_queries(self, round_num, acquisition='ei', xi=0.01, kappa=2.0):
        """
        Generate queries for the specified round.
        
        Parameters:
        -----------
        round_num : int
            Current round number
        acquisition : str
            Acquisition function to use ('ei' or 'ucb')
        xi : float
            Exploration parameter for EI
        kappa : float
            Exploration parameter for UCB
        
        Returns:
        --------
        queries : dict
            Dictionary mapping function_id to formatted query string
        """
        queries = {}
        
        print(f"\n{'='*60}")
        print(f"GENERATING ROUND {round_num} QUERIES")
        print(f"{'='*60}\n")
        
        for func_id, optimizer in self.optimizers.items():
            if len(optimizer.y_observed) == 0:
                print(f"Function {func_id}: No observations yet. Skipping.")
                continue
            
            # Adjust exploration based on round number
            # Early rounds: explore more; later rounds: exploit more
            if acquisition == 'ei':
                current_xi = xi * (1.0 + (5 - round_num) / 5.0)
                next_query = optimizer.propose_next_query(
                    acquisition='ei',
                    xi=current_xi
                )
            else:
                current_kappa = kappa * (1.0 + (5 - round_num) / 5.0)
                next_query = optimizer.propose_next_query(
                    acquisition='ucb',
                    kappa=current_kappa
                )
            
            formatted = optimizer.format_query(next_query)
            queries[func_id] = formatted
            
            best_x, best_y = optimizer.get_best_observation()
            
            print(f"Function {func_id} ({optimizer.n_dimensions}D):")
            print(f"  Best so far: {best_y:.6f}")
            print(f"  Next query:  {formatted}")
            print()
        
        return queries
    
    def save_queries_to_file(self, queries, round_num, filepath=None):
        """Save generated queries to file."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"round_{round_num}_queries_{timestamp}.txt"
        
        with open(filepath, 'w') as f:
            f.write(f"ROUND {round_num} QUERIES - {datetime.now()}\n")
            f.write("="*60 + "\n\n")
            for func_id, query in queries.items():
                f.write(f"Function {func_id}:\n{query}\n\n")
        
        print(f"Queries saved to: {filepath}")
        return filepath
    
    def visualize_progress(self, func_id):
        """Visualize optimization progress for a specific function."""
        optimizer = self.optimizers[func_id]
        
        if len(optimizer.y_observed) < 2:
            print(f"Not enough data for Function {func_id}")
            return
        
        rounds = list(range(1, len(optimizer.y_observed) + 1))
        distances = optimizer.y_observed
        
        # Calculate best so far at each round
        best_so_far = [min(distances[:i+1]) for i in range(len(distances))]
        
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, distances, 'o-', label='Distance at each query', alpha=0.6)
        plt.plot(rounds, best_so_far, 's-', label='Best so far', linewidth=2, color='red')
        plt.xlabel('Query Round')
        plt.ylabel('Distance to Optimum')
        plt.title(f'Function {func_id} ({optimizer.n_dimensions}D) - Optimization Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'function_{func_id}_progress.png', dpi=150)
        plt.show()
        
        print(f"Progress plot saved as: function_{func_id}_progress.png")


def create_sample_history_template():
    """
    Create a template JSON file for entering query history.
    You need to fill this with your actual results from the portal.
    """
    template = {
        "function_1": [],
        "function_2": [],
        "function_3": [],
        "function_4": [],
        "function_5": [],
        "function_6": [],
        "function_7": [],
        "function_8": []
    }
    
    # Example format (you need to replace with actual data from portal)
    example_entry = {
        "round": 1,
        "query": [0.374546, 0.950714],  # Your query values
        "result": 0.123456  # Distance result from portal
    }
    
    template["_EXAMPLE_FORMAT"] = example_entry
    template["_INSTRUCTIONS"] = (
        "Fill in your query history for each function. "
        "Each entry should have: round (int), query (list of floats), result (float). "
        "Remove this _INSTRUCTIONS and _EXAMPLE_FORMAT keys before loading."
    )
    
    with open('query_history_template.json', 'w') as f:
        json.dump(template, f, indent=2)
    
    print("Created query_history_template.json")
    print("Please fill it with your actual results from the capstone portal.")


if __name__ == "__main__":
    print("BBO Capstone Framework")
    print("="*60)
    
    # Create template for entering history
    print("\n1. Creating query history template...")
    create_sample_history_template()
    
    print("\n2. To use this framework:")
    print("   a) Fill query_history_template.json with your portal results")
    print("   b) Load the history: manager.load_query_history_from_file('query_history.json')")
    print("   c) Generate next round: queries = manager.generate_round_queries(6)")
    print("   d) Submit queries to the portal")
    print("   e) Add new results and repeat")
    
    print("\nExample usage in Python:")
    print("""
    # Initialize manager
    manager = BBOCapstoneManager()
    
    # Load your history
    manager.load_query_history_from_file('query_history.json')
    
    # Generate round 6 queries
    queries = manager.generate_round_queries(round_num=6, acquisition='ei', xi=0.01)
    
    # Save queries
    manager.save_queries_to_file(queries, round_num=6)
    
    # Visualize progress for a function
    manager.visualize_progress(func_id=1)
    """)
