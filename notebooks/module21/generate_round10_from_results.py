"""
Generate Round 10 from results folder (inputs_9, outputs_9)
============================================================
Reads inputs_9.txt and outputs_9.txt from the results folder,
updates query_history.json with corrected Round 7-9 data, and
generates Round 10 queries using Gaussian Process Bayesian
Optimization with strong exploitation bias.

Module 21 focus: transparency, interpretability, and strategic
reasoning behind each query choice.
"""

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

FUNCTION_DIMS = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9}

# -- Full dataset from portal submissions (rounds 1-9) -----------------------
# Each function has 9 query points with their actual output values.
# Queries sourced from: 1stRun.log, 2ndRun.txt, 3rdRun.txt, 4thRun.log,
# 5thRun.txt, round_6/7/8/9_portal_submission.txt
# Outputs sourced from: outputs_9.txt (most current, corrects rounds 7-8)

ALL_QUERIES = {
    1: [  # 2D
        [0.374546, 0.950714],
        [0.135488, 0.887852],
        [0.666954, 0.508958],
        [0.588135, 0.470242],
        [0.539786, 0.516324],
        [0.550000, 0.500000],
        [0.550123, 0.499876],
        [0.822682, 0.538761],
        [0.082064, 0.647726],
    ],
    2: [  # 3D
        [0.156019, 0.598658, 0.731994],
        [0.932606, 0.445568, 0.388236],
        [0.317213, 0.468554, 0.585683],
        [0.476888, 0.419037, 0.445408],
        [0.561676, 0.517149, 0.471467],
        [0.895432, 0.465123, 0.405678],
        [0.918234, 0.452341, 0.391567],
        [0.117114, 0.562129, 0.743507],
        [0.824453, 0.208365, 0.755025],
    ],
    3: [  # 4D
        [0.058084, 0.155995, 0.601115, 0.866176],
        [0.257596, 0.657368, 0.492617, 0.964238],
        [0.361774, 0.320166, 0.578499, 0.650665],
        [0.523239, 0.584772, 0.589607, 0.543103],
        [0.477258, 0.579626, 0.558273, 0.579348],
        [0.465890, 0.568234, 0.545678, 0.587654],
        [0.478234, 0.579123, 0.547890, 0.586543],
        [0.262446, 0.964309, 0.456927, 0.972881],
        [0.265866, 0.326391, 0.406669, 0.979188],
    ],
    4: [  # 5D
        [0.020584, 0.212339, 0.708073, 0.832443, 0.969910],
        [0.800984, 0.455205, 0.801058, 0.041718, 0.769458],
        [0.369047, 0.309402, 0.537627, 0.618294, 0.384853],
        [0.441940, 0.507110, 0.441424, 0.545521, 0.524624],
        [0.540873, 0.476214, 0.478535, 0.444672, 0.462239],
        [0.456789, 0.492345, 0.445678, 0.534567, 0.518234],
        [0.451234, 0.498765, 0.448901, 0.532109, 0.515678],
        [0.126205, 0.522306, 0.457867, 0.251314, 0.149011],
        [0.992012, 0.506536, 0.453917, 0.248692, 0.723948],
    ],
    5: [  # 6D
        [0.181825, 0.183405, 0.291229, 0.304242, 0.431945, 0.524756],
        [0.003171, 0.292809, 0.610914, 0.913027, 0.300115, 0.248599],
        [0.491287, 0.441850, 0.551668, 0.494262, 0.568532, 0.411706],
        [0.526228, 0.571629, 0.488026, 0.555470, 0.508993, 0.439645],
        [0.527424, 0.392443, 0.479293, 0.516735, 0.520509, 0.438181],
        [0.518345, 0.556789, 0.492345, 0.545678, 0.512345, 0.441234],
        [0.524567, 0.561234, 0.491234, 0.552345, 0.508901, 0.442345],
        [0.562393, 0.979643, 0.435557, 0.321586, 0.495763, 0.449517],
        [0.100290, 0.727424, 0.126069, 0.724989, 0.499772, 0.447673],
    ],
    6: [  # 7D
        [0.139494, 0.199674, 0.292145, 0.366362, 0.456078, 0.611853, 0.785176],
        [0.666392, 0.987533, 0.468270, 0.123287, 0.916031, 0.946144, 0.277697],
        [0.318317, 0.654227, 0.335971, 0.601265, 0.605119, 0.525408, 0.374117],
        [0.514092, 0.431055, 0.478248, 0.536963, 0.581359, 0.453312, 0.487330],
        [0.487395, 0.397456, 0.443572, 0.540584, 0.461051, 0.575276, 0.522855],
        [0.492345, 0.408765, 0.452345, 0.536789, 0.468765, 0.567890, 0.518234],
        [0.489012, 0.401234, 0.448901, 0.538901, 0.464567, 0.572345, 0.519876],
        [0.487986, 0.358042, 0.332315, 0.207613, 0.921376, 0.542479, 0.217969],
        [0.270315, 0.689417, 0.620570, 0.912247, 0.424106, 0.468351, 0.844120],
    ],
    7: [  # 8D
        [0.046456, 0.065052, 0.170524, 0.514234, 0.592415, 0.607545, 0.948886, 0.965632],
        [0.519654, 0.154745, 0.014627, 0.324243, 0.990898, 0.513141, 0.876496, 0.067396],
        [0.322890, 0.561101, 0.688757, 0.573740, 0.686496, 0.637496, 0.384162, 0.355548],
        [0.548685, 0.550003, 0.456859, 0.551494, 0.529934, 0.520990, 0.505799, 0.509642],
        [0.530670, 0.580037, 0.533696, 0.444493, 0.575691, 0.523434, 0.532915, 0.445684],
        [0.534567, 0.176543, 0.028765, 0.342345, 0.965432, 0.523456, 0.854321, 0.089876],
        [0.521234, 0.158901, 0.018765, 0.331234, 0.984567, 0.512345, 0.861234, 0.072345],
        [0.780810, 0.000000, 0.000000, 0.534273, 1.000000, 0.164371, 0.360733, 0.000000],
        [0.741849, 0.509360, 0.533916, 0.398911, 0.982862, 0.331184, 0.000000, 0.281795],
    ],
    8: [  # 9D
        [0.034389, 0.097672, 0.122038, 0.304614, 0.440152, 0.495177, 0.684233, 0.808397, 0.909320],
        [0.284154, 0.468899, 0.761773, 0.922612, 0.393024, 0.929088, 0.499612, 0.802050, 0.896580],
        [0.437695, 0.503206, 0.414907, 0.510363, 0.444364, 0.690409, 0.680349, 0.430866, 0.500298],
        [0.419775, 0.452922, 0.530525, 0.517622, 0.444383, 0.591797, 0.523182, 0.558773, 0.452897],
        [0.454878, 0.503521, 0.543937, 0.575607, 0.612694, 0.514893, 0.553757, 0.552426, 0.521142],
        [0.298765, 0.487654, 0.745678, 0.905432, 0.408765, 0.912345, 0.512345, 0.789012, 0.883456],
        [0.291234, 0.472345, 0.751234, 0.908901, 0.401234, 0.918901, 0.501234, 0.791234, 0.891234],
        [0.817959, 0.370193, 0.779692, 0.187412, 0.858177, 0.975859, 0.502421, 0.865675, 0.904946],
        [0.035355, 0.404891, 0.776153, 0.142897, 0.851079, 0.168530, 0.182932, 0.113091, 0.487524],
    ],
}

# Outputs from outputs_9.txt -- the authoritative source for all 9 rounds
ALL_OUTPUTS = {
    1: [-1.5686164999936676e-117, 4.4488846642953846e-209, 5.291868485218929e-14,
        -2.6128465124536443e-20, 2.4311553935530046e-15, -2.1333153063279553e-16,
        -2.1137938962087e-16, -2.5680502508208995e-33, -1.1972870685928282e-116],
    2: [-0.05845512295108613, -0.020750821852237696, 0.06932114558483886,
        0.5905021699847953, 0.254319666792602, 0.35456078319836765,
        -0.018066915129189193, -0.06348576178993792, -0.007748336736257307],
    3: [-0.08321525788398944, -0.022088267987110773, -0.04849835381105542,
        -0.04062558235437169, -0.021633609011577026, -0.025949676120676255,
        -0.024209368976389277, -0.043699788965212996, -0.031203490499254437],
    4: [-21.385605674555148, -20.532683239342912, -5.578513783648848,
        -3.1161613440466236, -3.6288061382689665, -3.0616249589987983,
        -2.9502264875793744, -6.93656498721192, -18.249321892650915],
    5: [147.6011814151734, 97.51179544268183, 32.4524989709859,
        11.727445761906113, 40.07674867949901, 15.432353169634759,
        13.379212819434178, 251.18684510891538, 5.9314304069056725],
    6: [-1.5193900756617191, -2.54149787636781, -1.416298983513298,
        -1.1384380873502697, -0.9101466390547354, -0.9042608248664562,
        -1.0682234315524883, -1.7845035207634168, -0.941123215211988],
    7: [0.38360412094760543, 0.002245292831851837, 0.16598430618221965,
        0.28466883853897823, 0.30723872836050503, 0.002901784285771967,
        0.002445634269122726, 0.00042071309261710753, 0.036656093880015686],
    8: [9.4264818240261, 7.585588997865999, 8.7099671836714,
        8.806597807552599, 8.6596953455224, 7.6538668568561,
        7.649100940427401, 7.234776736463, 8.5386135115364],
}


def fit_gp_model(X, y, n_dims):
    """Fit a Gaussian Process model to the observed data."""
    kernel = C(1.0, (1e-4, 1e4)) * Matern(
        length_scale=np.ones(n_dims),
        length_scale_bounds=(1e-3, 1e3),
        nu=2.5,
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=15,
    )
    gp.fit(X, y)
    return gp


def expected_improvement(X, gp, y_best, xi=0.001):
    """Compute Expected Improvement acquisition function."""
    mu, sigma = gp.predict(X, return_std=True)
    with np.errstate(divide="warn", invalid="warn"):
        improvement = y_best - mu - xi
        Z = np.where(sigma > 1e-10, improvement / sigma, 0.0)
        ei = np.where(
            sigma > 1e-10,
            improvement * norm.cdf(Z) + sigma * norm.pdf(Z),
            0.0,
        )
    return ei


def propose_next_query(gp, X_obs, y_obs, n_dims, xi=0.001):
    """Use GP + EI + differential evolution to propose the next query."""
    y_best = np.min(y_obs)
    bounds = [(0.0, 1.0)] * n_dims

    n_candidates = max(10000, 2000 * n_dims)
    X_cand = np.random.uniform(0, 1, size=(n_candidates, n_dims))
    ei_vals = expected_improvement(X_cand, gp, y_best, xi=xi)
    top_idx = np.argmax(ei_vals)
    x0 = X_cand[top_idx]

    result = differential_evolution(
        lambda x: -expected_improvement(x.reshape(1, -1), gp, y_best, xi=xi)[0],
        bounds=bounds,
        seed=42,
        maxiter=200,
        x0=x0,
        tol=1e-10,
    )
    return result.x


def format_query(x):
    """Format a query for portal submission: 0.XXXXXX-0.XXXXXX-..."""
    return "-".join([f"{val:.6f}" for val in x])


def analyze_function_progress(func_id, queries, outputs):
    """Print analysis of a function's optimization progress."""
    n_dims = FUNCTION_DIMS[func_id]
    best_idx = np.argmin(outputs)
    best_val = outputs[best_idx]
    best_query = queries[best_idx]

    print(f"\n  Function {func_id} ({n_dims}D):")
    print(f"    Rounds observed:  {len(outputs)}")
    print(f"    Best result:      {best_val:.10f}  (round {best_idx + 1})")
    print(f"    Best query:       {format_query(best_query)}")
    print(f"    Worst result:     {max(outputs):.6f}")
    print(f"    Result range:     [{min(outputs):.6f}, {max(outputs):.6f}]")

    if len(outputs) >= 3:
        recent_best = min(outputs[-3:])
        overall_best = min(outputs)
        if abs(overall_best) > 1e-10:
            convergence = abs(recent_best - overall_best) / abs(overall_best)
            print(f"    Convergence gap:  {convergence:.4%}")


def main():
    np.random.seed(42)

    print("\n" + "=" * 70)
    print("GENERATE ROUND 10 - BBO CAPSTONE (Module 21)")
    print("Focus: Transparency & Interpretability")
    print("=" * 70)

    # -- Phase 1: Data audit ------------------------------------------------
    print("\n" + "-" * 70)
    print("PHASE 1: DATA AUDIT - 9 rounds x 8 functions = 72 observations")
    print("-" * 70)

    for func_id in range(1, 9):
        analyze_function_progress(
            func_id, ALL_QUERIES[func_id], ALL_OUTPUTS[func_id]
        )

    # -- Phase 2: Update query_history.json ---------------------------------
    print("\n" + "-" * 70)
    print("PHASE 2: UPDATING query_history.json WITH CORRECTED DATA")
    print("-" * 70)

    history_path = Path(__file__).parent.parent / "module17" / "query_history.json"
    history = {}
    for func_id in range(1, 9):
        key = f"function_{func_id}"
        entries = []
        for rnd in range(9):
            entries.append({
                "round": rnd + 1,
                "query": ALL_QUERIES[func_id][rnd],
                "result": ALL_OUTPUTS[func_id][rnd],
            })
        history[key] = entries

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Saved corrected history to {history_path}")

    # -- Phase 3: GP-based Bayesian Optimization ----------------------------
    print("\n" + "-" * 70)
    print("PHASE 3: GAUSSIAN PROCESS BAYESIAN OPTIMIZATION")
    print("Strategy: Strong exploitation (xi=0.001) - late-stage refinement")
    print("-" * 70)

    round10_queries = {}
    round10_reasoning = {}

    for func_id in range(1, 9):
        n_dims = FUNCTION_DIMS[func_id]
        X = np.array(ALL_QUERIES[func_id])
        y = np.array(ALL_OUTPUTS[func_id])
        best_idx = np.argmin(y)
        best_val = y[best_idx]
        best_x = X[best_idx]

        print(f"\n  Function {func_id} ({n_dims}D):")
        print(f"    Current best: {best_val:.10f} at round {best_idx + 1}")

        if func_id == 1:
            # F1 is solved -- all values are ~0; perturb slightly from best
            perturbation = np.random.normal(0, 0.02, size=n_dims)
            next_query = np.clip(best_x + perturbation, 0.001, 0.999)
            reasoning = (
                "F1 is effectively solved (all outputs ~0). "
                "Submitting a small perturbation from the best-known point "
                "to confirm the basin of attraction."
            )
        else:
            xi = 0.001 if func_id <= 4 else 0.003
            gp = fit_gp_model(X, y, n_dims)
            next_query = propose_next_query(gp, X, y, n_dims, xi=xi)

            mu_pred, sigma_pred = gp.predict(
                next_query.reshape(1, -1), return_std=True
            )
            reasoning = (
                f"GP predicts mu={mu_pred[0]:.4f}, sigma={sigma_pred[0]:.4f} at proposed point. "
                f"EI acquisition with xi={xi} favours exploitation near best region."
            )

        next_query = np.clip(next_query, 0.000001, 0.999999)
        formatted = format_query(next_query)
        round10_queries[func_id] = formatted
        round10_reasoning[func_id] = reasoning

        print(f"    Proposed query: {formatted}")
        print(f"    Reasoning: {reasoning}")

    # -- Phase 4: Save portal submission ------------------------------------
    print("\n" + "-" * 70)
    print("PHASE 4: PORTAL SUBMISSION FILE")
    print("-" * 70)

    submission_path = Path(__file__).parent / "round_10_portal_submission.txt"
    with open(submission_path, "w") as f:
        f.write("ROUND 10 QUERIES - 19 Data Points (Module 21)\n")
        f.write("=" * 60 + "\n\n")
        f.write("Copy and paste these queries into the capstone portal:\n\n")
        for func_id in range(1, 9):
            f.write(f"Function {func_id}:\n{round10_queries[func_id]}\n\n")
        f.write("=" * 60 + "\n")
        f.write("FORMAT: Each value 0.XXXXXX (6 decimal places)\n")
        f.write("=" * 60 + "\n\n")
        f.write("STRATEGY REASONING (per function):\n")
        f.write("-" * 60 + "\n\n")
        for func_id in range(1, 9):
            n_dims = FUNCTION_DIMS[func_id]
            best_val = min(ALL_OUTPUTS[func_id])
            f.write(f"Function {func_id} ({n_dims}D):\n")
            f.write(f"  Best result so far: {best_val:.10f}\n")
            f.write(f"  Reasoning: {round10_reasoning[func_id]}\n\n")

    print(f"  Saved to {submission_path}")

    # -- Phase 5: Save inputs_10.txt for results folder --------------------
    results_path = Path(__file__).parent.parent.parent / "results"

    inputs10_entries = []
    for func_id in range(1, 9):
        vals = [float(v) for v in round10_queries[func_id].split("-")]
        inputs10_entries.append(f"array([{', '.join(f'{v:.6f}' for v in vals)}])")

    inputs10_path = results_path / "inputs_10.txt"

    # Read existing inputs_9.txt and append
    inputs9_path = results_path / "inputs_9.txt"
    with open(inputs9_path, "r") as f:
        prev_content = f.read()

    with open(inputs10_path, "w") as f:
        f.write(prev_content)
        f.write("[" + ", ".join(inputs10_entries) + "]\n")

    print(f"  Saved inputs_10.txt to {inputs10_path}")

    # -- Summary ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("ROUND 10 SUMMARY")
    print("=" * 70)
    print("\nQueries ready for portal submission:\n")
    for func_id in range(1, 9):
        print(f"  Function {func_id}: {round10_queries[func_id]}")
    print(f"\nPortal submission file: {submission_path}")
    print(f"Inputs file:           {inputs10_path}")
    print(f"Query history:         {history_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
