"""
Generate Round 12 from results folder (inputs_11, outputs_11)
=============================================================
Reads outputs_11.txt for all 11 rounds of results, performs
PCA-guided analysis on the query history for each function, and
generates Round 12 queries using principal-component-aware
Gaussian Process Bayesian Optimisation.

Module 23 focus: PCA lens -- principal directions of variance,
dimension reduction, redundancy removal, and exploitation of
dominant performance drivers.
"""

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm, pearsonr
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

FUNCTION_DIMS = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9}

# ---------------------------------------------------------------------------
# Complete dataset: rounds 1-11, compiled from portal submissions
# ---------------------------------------------------------------------------
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
        [0.559934, 0.497235],
        [0.557451, 0.497926],
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
        [0.999844, 0.000034, 0.708236],
        [0.000021, 0.999991, 0.790833],
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
        [0.000001, 0.000001, 0.615263, 0.842771],
        [0.000007, 0.000001, 0.685479, 0.330449],
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
        [0.999999, 0.000001, 0.992515, 0.000002, 0.895787],
        [0.999999, 0.000001, 0.999999, 0.000001, 0.999999],
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
        [0.313471, 0.488795, 0.999999, 0.602369, 0.849144, 0.657693],
        [0.999999, 0.579618, 0.999420, 0.904498, 0.334814, 0.427739],
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
        [0.283844, 0.993469, 0.472347, 0.679504, 0.054585, 0.634531, 0.667397],
        [0.839883, 0.597888, 0.452427, 0.124543, 0.320003, 0.593155, 0.516219],
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
        [0.999882, 0.000349, 0.999999, 0.999877, 0.843777, 0.000007, 0.000032, 0.000001],
        [0.999943, 0.999902, 0.999997, 0.999999, 0.000090, 0.000008, 0.000001, 0.195919],
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
        [0.999999, 0.000001, 0.999999, 0.000001, 0.114554, 0.999999, 0.999999, 0.988025, 0.999999],
        [0.999999, 0.999999, 0.999999, 0.999999, 0.999999, 0.999999, 0.999999, 0.000001, 0.000001],
    ],
}

ALL_OUTPUTS = {
    1: [-1.5686164999936676e-117, 4.4488846642953846e-209, 5.291868485218929e-14,
        -2.6128465124536443e-20, 2.4311553935530046e-15, -2.1333153063279553e-16,
        -2.1137938962087e-16, -2.5680502508208995e-33, -1.1972870685928282e-116,
        -5.355390299358942e-16, -4.113648226657962e-16],
    2: [-0.05845512295108613, -0.020750821852237696, 0.06932114558483886,
        0.5905021699847953, 0.254319666792602, 0.35456078319836765,
        -0.018066915129189193, -0.06348576178993792, -0.007748336736257307,
        0.07174667606311323, -0.03690556974491386],
    3: [-0.08321525788398944, -0.022088267987110773, -0.04849835381105542,
        -0.04062558235437169, -0.021633609011577026, -0.025949676120676255,
        -0.024209368976389277, -0.043699788965212996, -0.031203490499254437,
        -0.12859019857443935, -0.17917377603603626],
    4: [-21.385605674555148, -20.532683239342912, -5.578513783648848,
        -3.1161613440466236, -3.6288061382689665, -3.0616249589987983,
        -2.9502264875793744, -6.93656498721192, -18.249321892650915,
        -39.93467478186553, -40.34835395617227],
    5: [147.6011814151734, 97.51179544268183, 32.4524989709859,
        11.727445761906113, 40.07674867949901, 15.432353169634759,
        13.379212819434178, 251.18684510891538, 5.9314304069056725,
        372.69959512382667, 3730.890274283472],
    6: [-1.5193900756617191, -2.54149787636781, -1.416298983513298,
        -1.1384380873502697, -0.9101466390547354, -0.9042608248664562,
        -1.0682234315524883, -1.7845035207634168, -0.941123215211988,
        -1.0607142535163498, -1.7883030998957097],
    7: [0.38360412094760543, 0.002245292831851837, 0.16598430618221965,
        0.28466883853897823, 0.30723872836050503, 0.002901784285771967,
        0.002445634269122726, 0.00042071309261710753, 0.036656093880015686,
        2.664723365466533e-06, 0.0011096537105992036],
    8: [9.4264818240261, 7.585588997865999, 8.7099671836714,
        8.806597807552599, 8.6596953455224, 7.6538668568561,
        7.649100940427401, 7.234776736463, 8.5386135115364,
        4.2843391704695, 3.0783167399894014],
}


def format_query(x):
    return "-".join([f"{val:.6f}" for val in x])


def clip_x0_to_bounds(x0, bounds):
    """Clip initial point to lie strictly within DE bounds."""
    eps = 1e-8
    x0_clipped = np.array(x0, dtype=float)
    for i, (lo, hi) in enumerate(bounds):
        x0_clipped[i] = np.clip(x0_clipped[i], lo + eps, hi - eps)
    return x0_clipped


def pca_analysis(func_id, X, y):
    """Run PCA on query history, correlate components with outputs."""
    n_dims = FUNCTION_DIMS[func_id]
    X_arr = np.array(X)
    y_arr = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)

    n_components = min(len(X_arr), n_dims)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    print(f"\n  Function {func_id} ({n_dims}D) - PCA Analysis:")
    print(f"    Explained variance ratios: "
          f"[{', '.join(f'{v:.3f}' for v in pca.explained_variance_ratio_)}]")

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_90 = np.searchsorted(cumulative, 0.90) + 1
    print(f"    Components for 90% variance: {n_90} of {n_dims}")

    pc_correlations = []
    for pc_idx in range(min(n_components, n_dims)):
        scores = X_pca[:, pc_idx]
        if np.std(scores) > 1e-10 and np.std(y_arr) > 1e-10:
            corr, pval = pearsonr(scores, y_arr)
        else:
            corr, pval = 0.0, 1.0
        pc_correlations.append((pc_idx, corr, pval))

    pc_correlations.sort(key=lambda t: abs(t[1]), reverse=True)
    print(f"    PC-output correlations (by |r|):")
    for pc_idx, corr, pval in pc_correlations[:3]:
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        loadings = pca.components_[pc_idx]
        top_dims = np.argsort(np.abs(loadings))[::-1][:3]
        dim_str = ", ".join(f"x{d+1}({loadings[d]:+.2f})" for d in top_dims)
        print(f"      PC{pc_idx}: r={corr:+.3f} (p={pval:.3f}) {sig}")
        print(f"        Top loadings: {dim_str}")

    return pca, scaler, X_pca, pc_correlations


def fit_gp_model(X, y, n_dims):
    kernel = C(1.0, (1e-4, 1e4)) * Matern(
        length_scale=np.ones(n_dims),
        length_scale_bounds=(1e-3, 1e3),
        nu=2.5,
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-6,
        normalize_y=True, n_restarts_optimizer=15,
    )
    gp.fit(X, y)
    return gp


def expected_improvement(X, gp, y_best, xi=0.001):
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


def propose_query_pca_gp(func_id, X, y, pca, scaler, pc_correlations):
    """PCA-guided GP-BO: generate candidates biased along the principal
    components that correlate most strongly with low output values."""
    n_dims = FUNCTION_DIMS[func_id]
    X_arr = np.array(X)
    y_arr = np.array(y)

    gp = fit_gp_model(X_arr, y_arr, n_dims)
    y_best = np.min(y_arr)
    best_idx = np.argmin(y_arr)
    best_x = X_arr[best_idx]

    top_3_best = np.argsort(y_arr)[:3]
    elite_centre = np.mean(X_arr[top_3_best], axis=0)

    n_total = max(20000, 4000 * n_dims)

    # 40% along dominant PCs from the elite centre
    n_pc_biased = int(n_total * 0.40)
    cands_pc = np.tile(elite_centre, (n_pc_biased, 1))

    most_correlated_pc = pc_correlations[0][0]
    pc_direction = pca.components_[most_correlated_pc]
    pc_direction_orig = pc_direction / (scaler.scale_ + 1e-12)
    pc_direction_orig /= np.linalg.norm(pc_direction_orig) + 1e-12

    corr_sign = pc_correlations[0][1]
    search_dir = -np.sign(corr_sign) * pc_direction_orig

    alphas = np.random.uniform(-0.3, 0.3, size=(n_pc_biased, 1))
    perturbation = np.random.normal(0, 0.05, size=(n_pc_biased, n_dims))
    cands_pc += alphas * search_dir + perturbation

    # 35% tight exploitation around the best known point
    n_exploit = int(n_total * 0.35)
    tight_r = 0.06
    cands_exploit = best_x + np.random.uniform(-tight_r, tight_r,
                                                size=(n_exploit, n_dims))

    # 25% global exploration
    n_global = n_total - n_pc_biased - n_exploit
    cands_global = np.random.uniform(0, 1, size=(n_global, n_dims))

    X_cand = np.clip(
        np.vstack([cands_pc, cands_exploit, cands_global]),
        0.000001, 0.999999,
    )

    xi = 0.0005 if n_dims <= 5 else 0.002
    ei_vals = expected_improvement(X_cand, gp, y_best, xi=xi)
    top_idx = np.argmax(ei_vals)
    x0 = X_cand[top_idx]

    bounds = [(0.000001, 0.999999)] * n_dims
    result = differential_evolution(
        lambda x: -expected_improvement(x.reshape(1, -1), gp, y_best, xi=xi)[0],
        bounds=bounds, seed=42, maxiter=300, x0=x0, tol=1e-12,
    )

    next_query = np.clip(result.x, 0.000001, 0.999999)

    mu_pred, sigma_pred = gp.predict(next_query.reshape(1, -1), return_std=True)
    dist_to_best = np.linalg.norm(next_query - best_x)
    dist_to_elite = np.linalg.norm(next_query - elite_centre)

    reasoning = (
        f"PCA-guided GP-BO (PC{most_correlated_pc} r={corr_sign:+.3f} "
        f"drives search). "
        f"GP: mu={mu_pred[0]:.4f}, sigma={sigma_pred[0]:.4f}. "
        f"Dist to best: {dist_to_best:.4f}, to elite centre: {dist_to_elite:.4f}."
    )

    return next_query, reasoning, mu_pred[0], sigma_pred[0]


def main():
    np.random.seed(42)

    print("\n" + "=" * 70)
    print("GENERATE ROUND 12 - BBO CAPSTONE (Module 23)")
    print("Focus: PCA Lens - principal directions, variance, dimension")
    print("       reduction, redundancy removal")
    print("=" * 70)

    # -- Phase 1: Round 11 results audit -------------------------------------
    print("\n" + "-" * 70)
    print("PHASE 1: ROUND 11 RESULTS AUDIT (11 rounds, 22 data points total)")
    print("-" * 70)

    for func_id in range(1, 9):
        y = ALL_OUTPUTS[func_id]
        best_idx = int(np.argmin(y))
        r11_val = y[10]
        prev_best = min(y[:10])
        improved = r11_val < prev_best
        print(f"  F{func_id} ({FUNCTION_DIMS[func_id]}D): "
              f"R11={r11_val:>14.6f}  "
              f"prev_best={prev_best:>14.6f}  "
              f"overall_best=R{best_idx+1}  "
              f"{'** NEW BEST **' if improved else ''}")

    # -- Phase 2: PCA analysis -----------------------------------------------
    print("\n" + "-" * 70)
    print("PHASE 2: PCA ANALYSIS (11 queries per function)")
    print("-" * 70)

    pca_results = {}
    for func_id in range(1, 9):
        pca_obj, scaler, X_pca, pc_corrs = pca_analysis(
            func_id, ALL_QUERIES[func_id], ALL_OUTPUTS[func_id],
        )
        pca_results[func_id] = (pca_obj, scaler, X_pca, pc_corrs)

    # -- Phase 3: Per-dimension importance ------------------------------------
    print("\n" + "-" * 70)
    print("PHASE 3: PER-DIMENSION IMPORTANCE (raw correlation with output)")
    print("-" * 70)

    for func_id in range(1, 9):
        n_dims = FUNCTION_DIMS[func_id]
        X_arr = np.array(ALL_QUERIES[func_id])
        y_arr = np.array(ALL_OUTPUTS[func_id])

        print(f"\n  Function {func_id} ({n_dims}D):")
        dim_corrs = []
        for d in range(n_dims):
            if np.std(X_arr[:, d]) > 1e-10:
                corr, _ = pearsonr(X_arr[:, d], y_arr)
            else:
                corr = 0.0
            dim_corrs.append(corr)
            sign = "+" if corr > 0 else "-"
            bar = "#" * int(abs(corr) * 20)
            print(f"    x{d+1}: r={corr:+.3f} {bar}")

    # -- Phase 4: Update query_history.json ----------------------------------
    print("\n" + "-" * 70)
    print("PHASE 4: UPDATING query_history.json WITH 11-ROUND DATA")
    print("-" * 70)

    history_path = Path(__file__).parent.parent / "module17" / "query_history.json"
    history = {}
    for func_id in range(1, 9):
        key = f"function_{func_id}"
        entries = []
        for rnd in range(11):
            entries.append({
                "round": rnd + 1,
                "query": ALL_QUERIES[func_id][rnd],
                "result": ALL_OUTPUTS[func_id][rnd],
            })
        history[key] = entries

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Saved to {history_path}")

    # -- Phase 5: Generate round 12 queries ----------------------------------
    print("\n" + "-" * 70)
    print("PHASE 5: PCA-GUIDED GP BAYESIAN OPTIMISATION FOR ROUND 12")
    print("-" * 70)

    round12_queries = {}
    round12_reasoning = {}

    for func_id in range(1, 9):
        n_dims = FUNCTION_DIMS[func_id]
        X = ALL_QUERIES[func_id]
        y = ALL_OUTPUTS[func_id]
        pca_obj, scaler, X_pca, pc_corrs = pca_results[func_id]

        if func_id == 1:
            best_x = np.array([0.55, 0.50])
            perturbation = np.random.normal(0, 0.008, size=n_dims)
            next_query = np.clip(best_x + perturbation, 0.000001, 0.999999)
            reasoning = (
                "F1 solved (all outputs ~0). Tiny perturbation from "
                "(0.55, 0.50); PCA shows both PCs have negligible "
                "correlation with output."
            )

        elif func_id == 3:
            # PCA PC0 (r=-0.882***): low x1, low x2, high x3 drive
            # low output. R11 best at [~0, ~0, 0.685, 0.330].
            # Raw correlations: x1(+0.82), x2(+0.78) → minimize these.
            # x3(-0.72) → maximize this. x4(+0.33) → minimize.
            gp = fit_gp_model(np.array(X), np.array(y), n_dims)
            y_best = min(y)
            best_x = np.array(X[10])  # R11

            n_cand = 25000
            cands_tight = best_x + np.random.uniform(
                -0.08, 0.08, size=(int(n_cand * 0.5), n_dims)
            )
            cands_tight[:, 0] = np.random.uniform(0, 0.05, int(n_cand * 0.5))
            cands_tight[:, 1] = np.random.uniform(0, 0.05, int(n_cand * 0.5))

            cands_x4_scan = np.tile(best_x, (int(n_cand * 0.3), 1))
            cands_x4_scan[:, 0] = np.random.uniform(0, 0.05, int(n_cand * 0.3))
            cands_x4_scan[:, 1] = np.random.uniform(0, 0.05, int(n_cand * 0.3))
            cands_x4_scan[:, 2] = np.random.uniform(0.55, 0.80, int(n_cand * 0.3))
            cands_x4_scan[:, 3] = np.random.uniform(0.0, 1.0, int(n_cand * 0.3))

            cands_global = np.random.uniform(0, 1, size=(int(n_cand * 0.2), n_dims))

            X_cand = np.clip(
                np.vstack([cands_tight, cands_x4_scan, cands_global]),
                0.000001, 0.999999,
            )
            xi = 0.001
            ei_vals = expected_improvement(X_cand, gp, y_best, xi=xi)
            x0 = X_cand[np.argmax(ei_vals)]

            # Constrain DE to the PCA-identified basin
            bounds_f3 = [
                (0.000001, 0.10),   # x1: low (r=+0.82)
                (0.000001, 0.10),   # x2: low (r=+0.78)
                (0.50, 0.85),       # x3: moderate-high (r=-0.72)
                (0.000001, 0.999999),  # x4: scan full range
            ]
            x0 = clip_x0_to_bounds(x0, bounds_f3)
            result = differential_evolution(
                lambda x: -expected_improvement(
                    x.reshape(1, -1), gp, y_best, xi=xi
                )[0],
                bounds=bounds_f3, seed=42, maxiter=300, x0=x0, tol=1e-12,
            )
            next_query = np.clip(result.x, 0.000001, 0.999999)

            mu_pred, sigma_pred = gp.predict(
                next_query.reshape(1, -1), return_std=True
            )
            reasoning = (
                f"F3: PCA PC0 (r=-0.882***) confirms low-x1/x2, "
                f"high-x3 basin. DE bounded to [0,0.1]x[0,0.1]x[0.5,0.85]. "
                f"Scanning x4. "
                f"GP: mu={mu_pred[0]:.4f}, sigma={sigma_pred[0]:.4f}."
            )

        elif func_id == 4:
            # PCA PC0 (r=-0.962***): high x3, low x2, high x5.
            # Raw: x3(-0.94), x5(-0.82), x1(-0.63) → high=good.
            #      x2(+0.87), x4(+0.65) → low=good.
            # R11 best at [1,0,1,0,1]. Fine-tune around this corner.
            gp = fit_gp_model(np.array(X), np.array(y), n_dims)
            y_best = min(y)
            best_x = np.array(X[10])  # R11: [1,0,1,0,1]

            n_cand = 25000
            cands_tight = best_x + np.random.uniform(
                -0.04, 0.04, size=(int(n_cand * 0.6), n_dims)
            )

            second_x = np.array(X[9])  # R10
            cands_second = second_x + np.random.uniform(
                -0.06, 0.06, size=(int(n_cand * 0.2), n_dims)
            )

            corner_pattern = np.array([1, 0, 1, 0, 1], dtype=float)
            cands_corner = np.tile(corner_pattern, (int(n_cand * 0.2), 1))
            cands_corner += np.random.normal(0, 0.08, size=cands_corner.shape)

            X_cand = np.clip(
                np.vstack([cands_tight, cands_second, cands_corner]),
                0.000001, 0.999999,
            )
            xi = 0.0005
            ei_vals = expected_improvement(X_cand, gp, y_best, xi=xi)
            x0 = X_cand[np.argmax(ei_vals)]

            # Constrain DE to the [1,0,1,0,1] corner neighbourhood
            bounds_f4 = [
                (0.85, 0.999999),   # x1: high (r=-0.63)
                (0.000001, 0.15),   # x2: low (r=+0.87)
                (0.85, 0.999999),   # x3: high (r=-0.94)
                (0.000001, 0.15),   # x4: low (r=+0.65)
                (0.80, 0.999999),   # x5: high (r=-0.82)
            ]
            x0 = clip_x0_to_bounds(x0, bounds_f4)
            result = differential_evolution(
                lambda x: -expected_improvement(
                    x.reshape(1, -1), gp, y_best, xi=xi
                )[0],
                bounds=bounds_f4, seed=42, maxiter=300, x0=x0, tol=1e-12,
            )
            next_query = np.clip(result.x, 0.000001, 0.999999)

            mu_pred, sigma_pred = gp.predict(
                next_query.reshape(1, -1), return_std=True
            )
            reasoning = (
                f"F4: PCA PC0 (r=-0.962***) aligns with [1,0,1,0,1] "
                f"corner. DE bounded near corner. "
                f"GP: mu={mu_pred[0]:.4f}, sigma={sigma_pred[0]:.4f}."
            )

        elif func_id == 5:
            # PCA PC1 (r=+0.820***): high x1/x3/x4 → HIGH output.
            # Raw: x1(+0.68), x3(+0.63), x4(+0.51) → low=good.
            #      x5(-0.34) → high=good. x2(+0.10) → neutral.
            # Boundaries catastrophic (R10=373, R11=3731).
            # Best = R9 (5.93): [0.10, 0.73, 0.13, 0.72, 0.50, 0.45]
            # Pattern: low x1, low x3, moderate-high x2/x4.
            gp = fit_gp_model(np.array(X), np.array(y), n_dims)
            y_best = min(y)
            best_x = np.array(X[8])  # R9

            n_cand = 30000
            cands_tight = best_x + np.random.uniform(
                -0.08, 0.08, size=(int(n_cand * 0.6), n_dims)
            )

            second_best_x = np.array(X[3])  # R4 (11.73)
            cands_second = second_best_x + np.random.uniform(
                -0.08, 0.08, size=(int(n_cand * 0.2), n_dims)
            )

            cands_interior = np.random.uniform(0.05, 0.85,
                                               size=(int(n_cand * 0.2), n_dims))
            cands_interior[:, 0] = np.random.uniform(0.02, 0.30,
                                                     int(n_cand * 0.2))
            cands_interior[:, 2] = np.random.uniform(0.02, 0.30,
                                                     int(n_cand * 0.2))

            X_cand = np.clip(
                np.vstack([cands_tight, cands_second, cands_interior]),
                0.020000, 0.900000,
            )
            xi = 0.005
            ei_vals = expected_improvement(X_cand, gp, y_best, xi=xi)
            x0 = X_cand[np.argmax(ei_vals)]

            # Tight interior bounds informed by PCA direction
            bounds_f5 = [
                (0.020000, 0.350000),  # x1: low (r=+0.68)
                (0.200000, 0.900000),  # x2: moderate (r=+0.10)
                (0.020000, 0.350000),  # x3: low (r=+0.63)
                (0.300000, 0.900000),  # x4: moderate-high
                (0.300000, 0.750000),  # x5: moderate (r=-0.34)
                (0.250000, 0.650000),  # x6: moderate (r=-0.01)
            ]
            x0 = clip_x0_to_bounds(x0, bounds_f5)
            result = differential_evolution(
                lambda x: -expected_improvement(
                    x.reshape(1, -1), gp, y_best, xi=xi
                )[0],
                bounds=bounds_f5, seed=42, maxiter=300, x0=x0, tol=1e-12,
            )
            next_query = np.clip(result.x, 0.020000, 0.900000)

            mu_pred, sigma_pred = gp.predict(
                next_query.reshape(1, -1), return_std=True
            )
            reasoning = (
                f"F5: PCA PC1 (r=+0.820***) shows high x1/x3 drive "
                f"HIGH output. DE bounded to interior [low x1/x3, "
                f"moderate x2/x4] near R9 basin. "
                f"GP: mu={mu_pred[0]:.2f}, sigma={sigma_pred[0]:.2f}."
            )

        else:
            next_query, reasoning, _, _ = propose_query_pca_gp(
                func_id, X, y, pca_obj, scaler, pc_corrs,
            )

        formatted = format_query(next_query)
        round12_queries[func_id] = formatted
        round12_reasoning[func_id] = reasoning

        print(f"\n  Function {func_id} ({n_dims}D):")
        print(f"    Best so far: {min(y):.10f}")
        print(f"    R12 query:   {formatted}")
        print(f"    Reasoning:   {reasoning}")

    # -- Phase 6: Save portal submission -------------------------------------
    print("\n" + "-" * 70)
    print("PHASE 6: PORTAL SUBMISSION FILE")
    print("-" * 70)

    submission_path = Path(__file__).parent / "round_12_portal_submission.txt"
    with open(submission_path, "w") as f:
        f.write("ROUND 12 QUERIES - 22 Data Points (Module 23)\n")
        f.write("=" * 60 + "\n\n")
        f.write("Copy and paste these queries into the capstone portal:\n\n")
        for func_id in range(1, 9):
            f.write(f"Function {func_id}:\n{round12_queries[func_id]}\n\n")
        f.write("=" * 60 + "\n")
        f.write("FORMAT: Each value 0.XXXXXX (6 decimal places)\n")
        f.write("=" * 60 + "\n\n")
        f.write("PCA-GUIDED REASONING (per function):\n")
        f.write("-" * 60 + "\n\n")
        for func_id in range(1, 9):
            n_dims = FUNCTION_DIMS[func_id]
            best_val = min(ALL_OUTPUTS[func_id])
            f.write(f"Function {func_id} ({n_dims}D):\n")
            f.write(f"  Best result so far: {best_val:.10f}\n")
            f.write(f"  Reasoning: {round12_reasoning[func_id]}\n\n")

    print(f"  Saved to {submission_path}")

    # -- Phase 7: Save inputs_12.txt -----------------------------------------
    results_path = Path(__file__).parent.parent.parent / "results"

    inputs12_entries = []
    for func_id in range(1, 9):
        vals = [float(v) for v in round12_queries[func_id].split("-")]
        inputs12_entries.append(
            f"array([{', '.join(f'{v:.6f}' for v in vals)}])"
        )

    inputs12_path = results_path / "inputs_12.txt"
    inputs11_path = results_path / "inputs_11.txt"
    with open(inputs11_path, "r") as f:
        prev_content = f.read()
    with open(inputs12_path, "w") as f:
        f.write(prev_content)
        f.write("[" + ", ".join(inputs12_entries) + "]\n")
    print(f"  Saved inputs_12.txt to {inputs12_path}")

    # -- Summary -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ROUND 12 SUMMARY")
    print("=" * 70)
    print("\nQueries ready for portal submission:\n")
    for func_id in range(1, 9):
        print(f"  Function {func_id}: {round12_queries[func_id]}")
    print(f"\nPortal submission: {submission_path}")
    print(f"Inputs file:      {inputs12_path}")
    print(f"Query history:    {history_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
