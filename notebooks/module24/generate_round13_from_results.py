"""
Generate Round 13 (FINAL) from results folder (inputs_12, outputs_11)
=====================================================================
Reads outputs_11.txt for all 11 rounds of actual results and inputs_12.txt
for the complete query history (12 rounds submitted, 11 with results).

Module 24 focus: RL-informed final exploitation
  - UCB-style acquisition (MAB exploration-exploitation)
  - Q-value-inspired region scoring for each function
  - Policy adaptation based on cumulative feedback
  - Heavy exploitation (~95%) for the final submission

Strategy: For each function, we score candidate regions using a
Q-value analogue (discounted cumulative improvement), then apply
GP-BO with a UCB/EI hybrid acquisition to generate the final query.
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
# Complete dataset: rounds 1-11 outputs, rounds 1-12 inputs
# (R12 submitted but no output yet — use structural insights only)
# ---------------------------------------------------------------------------
ALL_QUERIES = {
    1: [  # 2D — F1
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
        [0.553974, 0.498894],  # R12 (no output yet)
    ],
    2: [  # 3D — F2
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
        [0.999997, 0.000004, 0.811363],  # R12
    ],
    3: [  # 4D — F3
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
        [0.100000, 0.000001, 0.695744, 0.999999],  # R12
    ],
    4: [  # 5D — F4
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
        [0.850000, 0.000001, 0.999999, 0.150000, 0.800000],  # R12
    ],
    5: [  # 6D — F5
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
        [0.154671, 0.900000, 0.020000, 0.900000, 0.300000, 0.250000],  # R12
    ],
    6: [  # 7D — F6
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
        [0.771469, 0.649998, 0.809398, 0.121973, 0.381539, 0.971928, 0.838881],  # R12
    ],
    7: [  # 8D — F7
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
        [0.999998, 0.000007, 0.999999, 0.999998, 0.000001, 0.000004, 0.000001, 0.153647],  # R12
    ],
    8: [  # 9D — F8
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
        [0.999999, 0.804464, 0.999999, 0.000001, 0.999999, 0.000001, 0.999999, 0.999999, 0.999999],  # R12
    ],
}

ALL_OUTPUTS = {
    1: [-1.5686e-117, 4.4489e-209, 5.2919e-14, -2.6128e-20, 2.4312e-15,
        -2.1333e-16, -2.1138e-16, -2.5681e-33, -1.1973e-116, -5.3554e-16,
        -4.1136e-16],
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


def clip_to_bounds(x0, bounds):
    eps = 1e-8
    x0_c = np.array(x0, dtype=float)
    for i, (lo, hi) in enumerate(bounds):
        x0_c[i] = np.clip(x0_c[i], lo + eps, hi - eps)
    return x0_c


def compute_q_values(y_history, gamma=0.9):
    """Compute Q-value analogue: discounted cumulative reward for improvement.
    Recent improvements weighted more heavily, like temporal-difference updates.
    'Reward' is the improvement over the running best at each step."""
    y = np.array(y_history)
    running_best = np.minimum.accumulate(y)
    improvements = np.zeros(len(y))
    improvements[0] = 0
    for t in range(1, len(y)):
        improvements[t] = max(0, running_best[t - 1] - y[t])

    q = 0.0
    q_values = np.zeros(len(y))
    for t in range(len(y) - 1, -1, -1):
        q = improvements[t] + gamma * q
        q_values[t] = q
    return q_values, running_best


def compute_ucb_score(mu, sigma, n_total, t, kappa_base=2.0):
    """UCB1-inspired score: balances exploitation (low mu) with exploration
    (high sigma), scaled by a decaying exploration bonus like MAB."""
    exploration_decay = kappa_base * np.sqrt(np.log(n_total + 1) / (t + 1))
    return mu - exploration_decay * sigma


def fit_gp_model(X, y, n_dims):
    kernel = C(1.0, (1e-4, 1e4)) * Matern(
        length_scale=np.ones(n_dims),
        length_scale_bounds=(1e-3, 1e3),
        nu=2.5,
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-6,
        normalize_y=True, n_restarts_optimizer=20,
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


def ucb_acquisition(X, gp, kappa=0.5):
    """Lower Confidence Bound for minimisation: mu - kappa * sigma."""
    mu, sigma = gp.predict(X, return_std=True)
    return -(mu - kappa * sigma)


def hybrid_acquisition(X, gp, y_best, xi=0.0005, kappa=0.3, w_ei=0.7, w_ucb=0.3):
    """Hybrid EI + UCB acquisition for the final round:
    heavy exploitation (EI) with a small UCB hedge."""
    ei = expected_improvement(X, gp, y_best, xi=xi)
    lcb = ucb_acquisition(X, gp, kappa=kappa)

    ei_norm = ei / (np.max(ei) + 1e-20)
    lcb_norm = lcb / (np.max(np.abs(lcb)) + 1e-20)

    return w_ei * ei_norm + w_ucb * lcb_norm


def pca_analysis(func_id, X, y):
    n_dims = FUNCTION_DIMS[func_id]
    X_arr = np.array(X)
    y_arr = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)

    n_components = min(len(X_arr), n_dims)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    pc_correlations = []
    for pc_idx in range(min(n_components, n_dims)):
        scores = X_pca[:, pc_idx]
        if np.std(scores) > 1e-10 and np.std(y_arr) > 1e-10:
            corr, pval = pearsonr(scores, y_arr)
        else:
            corr, pval = 0.0, 1.0
        pc_correlations.append((pc_idx, corr, pval))

    pc_correlations.sort(key=lambda t: abs(t[1]), reverse=True)
    return pca, scaler, X_pca, pc_correlations


def dimension_correlations(X, y, n_dims):
    X_arr = np.array(X)
    y_arr = np.array(y)
    corrs = []
    for d in range(n_dims):
        if np.std(X_arr[:, d]) > 1e-10:
            r, _ = pearsonr(X_arr[:, d], y_arr)
        else:
            r = 0.0
        corrs.append(r)
    return corrs


def generate_final_query(func_id, X_all, y, pca_obj, scaler, pc_corrs, dim_corrs):
    """Generate the FINAL query for a function using RL-informed GP-BO.

    Strategy for the last round:
    - 95% exploitation, 5% safety exploration
    - Hybrid EI + UCB acquisition
    - PCA-constrained DE bounds from R12 analysis
    - Q-value weighting to prioritise regions with recent improvement momentum
    """
    n_dims = FUNCTION_DIMS[func_id]
    X_with_output = X_all[:len(y)]
    X_arr = np.array(X_with_output)
    y_arr = np.array(y)

    gp = fit_gp_model(X_arr, y_arr, n_dims)
    y_best = np.min(y_arr)
    best_idx = np.argmin(y_arr)
    best_x = X_arr[best_idx]

    q_values, running_best = compute_q_values(y)

    top_k = min(3, len(y_arr))
    top_indices = np.argsort(y_arr)[:top_k]
    q_weights = np.array([q_values[i] + 1.0 for i in top_indices])
    q_weights /= q_weights.sum()
    elite_centre = np.average(X_arr[top_indices], axis=0, weights=q_weights)

    print(f"\n    Q-values (last 5): {[f'{q:.4f}' for q in q_values[-5:]]}")
    print(f"    Running best: {running_best[-1]:.6f}")
    print(f"    Elite centre (Q-weighted): [{', '.join(f'{v:.4f}' for v in elite_centre)}]")

    n_total = max(30000, 5000 * n_dims)

    n_tight = int(n_total * 0.50)
    tight_r = 0.04
    cands_tight = best_x + np.random.uniform(-tight_r, tight_r,
                                              size=(n_tight, n_dims))

    n_elite = int(n_total * 0.25)
    cands_elite = elite_centre + np.random.uniform(-0.06, 0.06,
                                                    size=(n_elite, n_dims))

    n_pc = int(n_total * 0.15)
    most_corr_pc = pc_corrs[0][0]
    pc_direction = pca_obj.components_[most_corr_pc]
    pc_dir_orig = pc_direction / (scaler.scale_ + 1e-12)
    pc_dir_orig /= np.linalg.norm(pc_dir_orig) + 1e-12
    corr_sign = pc_corrs[0][1]
    search_dir = -np.sign(corr_sign) * pc_dir_orig

    cands_pc = np.tile(best_x, (n_pc, 1))
    alphas = np.random.uniform(-0.15, 0.15, size=(n_pc, 1))
    cands_pc += alphas * search_dir + np.random.normal(0, 0.03, size=(n_pc, n_dims))

    n_corr = int(n_total * 0.05)
    cands_corr = np.tile(best_x, (n_corr, 1))
    for d in range(n_dims):
        r = dim_corrs[d]
        if abs(r) > 0.3:
            direction = -np.sign(r) * 0.05
            cands_corr[:, d] += np.random.uniform(direction - 0.03,
                                                   direction + 0.03, n_corr)
        else:
            cands_corr[:, d] += np.random.uniform(-0.05, 0.05, n_corr)

    n_global = n_total - n_tight - n_elite - n_pc - n_corr
    cands_global = np.random.uniform(0, 1, size=(n_global, n_dims))

    X_cand = np.clip(
        np.vstack([cands_tight, cands_elite, cands_pc, cands_corr, cands_global]),
        0.000001, 0.999999,
    )

    xi_final = 0.0002
    kappa_final = 0.2
    acq_vals = hybrid_acquisition(X_cand, gp, y_best,
                                  xi=xi_final, kappa=kappa_final,
                                  w_ei=0.75, w_ucb=0.25)
    top_idx = np.argmax(acq_vals)
    x0 = X_cand[top_idx]

    bounds = [(0.000001, 0.999999)] * n_dims
    result = differential_evolution(
        lambda x: -hybrid_acquisition(
            x.reshape(1, -1), gp, y_best,
            xi=xi_final, kappa=kappa_final, w_ei=0.75, w_ucb=0.25
        )[0],
        bounds=bounds, seed=42, maxiter=500, x0=x0, tol=1e-14,
    )
    next_query = np.clip(result.x, 0.000001, 0.999999)

    mu_pred, sigma_pred = gp.predict(next_query.reshape(1, -1), return_std=True)
    dist_to_best = np.linalg.norm(next_query - best_x)

    reasoning = (
        f"RL-informed final GP-BO (hybrid EI+UCB, xi={xi_final}, kappa={kappa_final}). "
        f"Q-weighted elite centre. PC{most_corr_pc} r={corr_sign:+.3f}. "
        f"GP: mu={mu_pred[0]:.4f}, sigma={sigma_pred[0]:.4f}. "
        f"Dist-to-best: {dist_to_best:.4f}."
    )
    return next_query, reasoning, mu_pred[0], sigma_pred[0]


def main():
    np.random.seed(42)

    print("\n" + "=" * 70)
    print("GENERATE ROUND 13 (FINAL) - BBO CAPSTONE (Module 24)")
    print("Focus: RL-Informed Final Exploitation")
    print("  UCB-style acquisition, Q-value region scoring,")
    print("  policy adaptation, 95% exploitation for final submission")
    print("=" * 70)

    # -- Phase 1: Audit all 11 rounds of results -------------------------
    print("\n" + "-" * 70)
    print("PHASE 1: 11-ROUND RESULTS AUDIT + Q-VALUE ANALYSIS")
    print("-" * 70)

    for func_id in range(1, 9):
        y = ALL_OUTPUTS[func_id]
        q_vals, running_best = compute_q_values(y)
        best_idx = int(np.argmin(y))
        momentum = q_vals[-1]
        print(f"  F{func_id} ({FUNCTION_DIMS[func_id]}D): "
              f"best={min(y):>14.6f} (R{best_idx+1})  "
              f"Q-momentum={momentum:.4f}  "
              f"recent_trend={'improving' if y[-1] <= running_best[-2] else 'regressing'}")

    # -- Phase 2: PCA + dimension correlations ----------------------------
    print("\n" + "-" * 70)
    print("PHASE 2: PCA + DIMENSION CORRELATION ANALYSIS")
    print("-" * 70)

    pca_results = {}
    dim_corr_results = {}
    for func_id in range(1, 9):
        n_dims = FUNCTION_DIMS[func_id]
        X_with_output = ALL_QUERIES[func_id][:len(ALL_OUTPUTS[func_id])]
        y = ALL_OUTPUTS[func_id]

        pca_obj, scaler, X_pca, pc_corrs = pca_analysis(func_id, X_with_output, y)
        dim_corrs = dimension_correlations(X_with_output, y, n_dims)

        pca_results[func_id] = (pca_obj, scaler, X_pca, pc_corrs)
        dim_corr_results[func_id] = dim_corrs

        top_pc = pc_corrs[0]
        print(f"  F{func_id} ({n_dims}D): top PC{top_pc[0]} r={top_pc[1]:+.3f} (p={top_pc[2]:.4f})")
        corr_str = " ".join(f"x{d+1}({r:+.2f})" for d, r in enumerate(dim_corrs))
        print(f"    Dim corrs: {corr_str}")

    # -- Phase 3: Generate Round 13 queries --------------------------------
    print("\n" + "-" * 70)
    print("PHASE 3: RL-INFORMED FINAL QUERY GENERATION (ROUND 13)")
    print("-" * 70)

    round13_queries = {}
    round13_reasoning = {}

    for func_id in range(1, 9):
        n_dims = FUNCTION_DIMS[func_id]
        X = ALL_QUERIES[func_id]
        y = ALL_OUTPUTS[func_id]
        pca_obj, scaler, X_pca, pc_corrs = pca_results[func_id]
        dim_corrs = dim_corr_results[func_id]

        print(f"\n  Function {func_id} ({n_dims}D):")
        print(f"    Best so far: {min(y):.10f}")

        if func_id == 1:
            # F1 is solved — all outputs ~0. Final confirmation query.
            best_x = np.array([0.55, 0.50])
            perturbation = np.random.normal(0, 0.005, size=n_dims)
            next_query = np.clip(best_x + perturbation, 0.000001, 0.999999)
            reasoning = (
                "F1 converged (all ~0). Final confirmation query near (0.55, 0.50) "
                "with minimal perturbation. RL analogue: greedy policy on converged Q-table."
            )
        elif func_id == 4:
            # F4: [1,0,1,0,1] corner confirmed by PCA (r=-0.962).
            # Final round: ultra-tight exploitation at the exact corner.
            X_arr = np.array(X[:len(y)])
            y_arr = np.array(y)
            gp = fit_gp_model(X_arr, y_arr, n_dims)
            y_best = min(y)

            n_cand = 40000
            corner = np.array([0.999999, 0.000001, 0.999999, 0.000001, 0.999999])
            cands = np.tile(corner, (n_cand, 1))
            cands += np.random.normal(0, 0.03, size=cands.shape)
            X_cand = np.clip(cands, 0.000001, 0.999999)

            acq = hybrid_acquisition(X_cand, gp, y_best, xi=0.0001, kappa=0.15)
            x0 = X_cand[np.argmax(acq)]

            bounds_f4 = [
                (0.90, 0.999999), (0.000001, 0.10),
                (0.90, 0.999999), (0.000001, 0.10),
                (0.90, 0.999999),
            ]
            x0 = clip_to_bounds(x0, bounds_f4)
            result = differential_evolution(
                lambda x: -hybrid_acquisition(
                    x.reshape(1, -1), gp, y_best, xi=0.0001, kappa=0.15
                )[0],
                bounds=bounds_f4, seed=42, maxiter=500, x0=x0, tol=1e-14,
            )
            next_query = np.clip(result.x, 0.000001, 0.999999)
            mu_pred, sigma_pred = gp.predict(next_query.reshape(1, -1), return_std=True)
            reasoning = (
                f"F4: Final exploitation at [1,0,1,0,1] corner (PCA r=-0.962). "
                f"Ultra-tight DE bounds [0.9,1]x[0,0.1]. Hybrid EI+UCB. "
                f"GP: mu={mu_pred[0]:.4f}, sigma={sigma_pred[0]:.4f}. "
                f"RL analogue: greedy action on highest-Q state."
            )

        elif func_id == 5:
            # F5: Interior optimum near R9 [0.10, 0.73, 0.13, 0.72, 0.50, 0.45].
            # Boundaries catastrophic. Final round: refine around R9 best.
            X_arr = np.array(X[:len(y)])
            y_arr = np.array(y)
            gp = fit_gp_model(X_arr, y_arr, n_dims)
            y_best = min(y)
            best_x = X_arr[np.argmin(y_arr)]  # R9

            n_cand = 40000
            cands_tight = best_x + np.random.uniform(-0.06, 0.06, size=(int(n_cand * 0.7), n_dims))
            cands_tight[:, 0] = np.clip(cands_tight[:, 0], 0.02, 0.25)
            cands_tight[:, 2] = np.clip(cands_tight[:, 2], 0.02, 0.25)

            # Explore the R4 basin (2nd best) — UCB hedge
            r4_x = X_arr[3]
            cands_r4 = r4_x + np.random.uniform(-0.06, 0.06, size=(int(n_cand * 0.2), n_dims))

            cands_interior = np.random.uniform(0.05, 0.80, size=(int(n_cand * 0.1), n_dims))
            cands_interior[:, 0] = np.random.uniform(0.02, 0.25, int(n_cand * 0.1))
            cands_interior[:, 2] = np.random.uniform(0.02, 0.15, int(n_cand * 0.1))

            X_cand = np.clip(
                np.vstack([cands_tight, cands_r4, cands_interior]),
                0.020000, 0.900000,
            )
            acq = hybrid_acquisition(X_cand, gp, y_best, xi=0.001, kappa=0.3)
            x0 = X_cand[np.argmax(acq)]

            bounds_f5 = [
                (0.020000, 0.250000), (0.400000, 0.900000),
                (0.020000, 0.250000), (0.400000, 0.900000),
                (0.300000, 0.700000), (0.250000, 0.600000),
            ]
            x0 = clip_to_bounds(x0, bounds_f5)
            result = differential_evolution(
                lambda x: -hybrid_acquisition(
                    x.reshape(1, -1), gp, y_best, xi=0.001, kappa=0.3
                )[0],
                bounds=bounds_f5, seed=42, maxiter=500, x0=x0, tol=1e-14,
            )
            next_query = np.clip(result.x, 0.020000, 0.900000)
            mu_pred, sigma_pred = gp.predict(next_query.reshape(1, -1), return_std=True)
            reasoning = (
                f"F5: Interior refinement near R9 best (5.93). Low x1/x3 critical "
                f"(PCA r=+0.820). DE bounded to safe interior. "
                f"GP: mu={mu_pred[0]:.2f}, sigma={sigma_pred[0]:.2f}. "
                f"RL analogue: epsilon-greedy with epsilon~0.05."
            )

        elif func_id == 3:
            # F3: PCA PC0 r=-0.882: low x1/x2, high x3. R11 best (-0.179).
            X_arr = np.array(X[:len(y)])
            y_arr = np.array(y)
            gp = fit_gp_model(X_arr, y_arr, n_dims)
            y_best = min(y)
            best_x = X_arr[np.argmin(y_arr)]

            n_cand = 30000
            cands_tight = best_x + np.random.uniform(-0.05, 0.05, size=(int(n_cand * 0.6), n_dims))
            cands_tight[:, 0] = np.random.uniform(0, 0.03, int(n_cand * 0.6))
            cands_tight[:, 1] = np.random.uniform(0, 0.03, int(n_cand * 0.6))

            cands_x4 = np.tile(best_x, (int(n_cand * 0.3), 1))
            cands_x4[:, 0] = np.random.uniform(0, 0.05, int(n_cand * 0.3))
            cands_x4[:, 1] = np.random.uniform(0, 0.05, int(n_cand * 0.3))
            cands_x4[:, 2] = np.random.uniform(0.55, 0.80, int(n_cand * 0.3))
            cands_x4[:, 3] = np.random.uniform(0.0, 1.0, int(n_cand * 0.3))

            cands_global = np.random.uniform(0, 1, size=(int(n_cand * 0.1), n_dims))

            X_cand = np.clip(
                np.vstack([cands_tight, cands_x4, cands_global]),
                0.000001, 0.999999,
            )
            acq = hybrid_acquisition(X_cand, gp, y_best, xi=0.0005, kappa=0.2)
            x0 = X_cand[np.argmax(acq)]

            bounds_f3 = [
                (0.000001, 0.08), (0.000001, 0.08),
                (0.55, 0.85), (0.000001, 0.999999),
            ]
            x0 = clip_to_bounds(x0, bounds_f3)
            result = differential_evolution(
                lambda x: -hybrid_acquisition(
                    x.reshape(1, -1), gp, y_best, xi=0.0005, kappa=0.2
                )[0],
                bounds=bounds_f3, seed=42, maxiter=500, x0=x0, tol=1e-14,
            )
            next_query = np.clip(result.x, 0.000001, 0.999999)
            mu_pred, sigma_pred = gp.predict(next_query.reshape(1, -1), return_std=True)
            reasoning = (
                f"F3: PCA PC0 r=-0.882 confirms low-x1/x2, high-x3 basin. "
                f"Final tight DE [0,0.08]x[0,0.08]x[0.55,0.85]. "
                f"GP: mu={mu_pred[0]:.4f}, sigma={sigma_pred[0]:.4f}."
            )

        elif func_id == 6:
            # F6: Hardest function — best still from R2 (-2.54). 7D.
            # Try a different strategy: explore the R2 basin AND the R8/R11 basin
            X_arr = np.array(X[:len(y)])
            y_arr = np.array(y)
            gp = fit_gp_model(X_arr, y_arr, n_dims)
            y_best = min(y)
            best_x = X_arr[np.argmin(y_arr)]  # R2

            # R2 basin: [0.67, 0.99, 0.47, 0.12, 0.92, 0.95, 0.28]
            # R8 basin: [0.49, 0.36, 0.33, 0.21, 0.92, 0.54, 0.22]
            # R11 basin: [0.84, 0.60, 0.45, 0.12, 0.32, 0.59, 0.52]

            n_cand = 40000
            cands_r2 = best_x + np.random.uniform(-0.08, 0.08, size=(int(n_cand * 0.5), n_dims))

            r8_x = X_arr[7]
            cands_r8 = r8_x + np.random.uniform(-0.10, 0.10, size=(int(n_cand * 0.2), n_dims))

            r11_x = X_arr[10]
            cands_r11 = r11_x + np.random.uniform(-0.10, 0.10, size=(int(n_cand * 0.15), n_dims))

            cands_global = np.random.uniform(0, 1, size=(int(n_cand * 0.15), n_dims))

            X_cand = np.clip(
                np.vstack([cands_r2, cands_r8, cands_r11, cands_global]),
                0.000001, 0.999999,
            )
            acq = hybrid_acquisition(X_cand, gp, y_best, xi=0.002, kappa=0.5,
                                     w_ei=0.6, w_ucb=0.4)
            x0 = X_cand[np.argmax(acq)]

            bounds = [(0.000001, 0.999999)] * n_dims
            result = differential_evolution(
                lambda x: -hybrid_acquisition(
                    x.reshape(1, -1), gp, y_best, xi=0.002, kappa=0.5,
                    w_ei=0.6, w_ucb=0.4
                )[0],
                bounds=bounds, seed=42, maxiter=500, x0=x0, tol=1e-14,
            )
            next_query = np.clip(result.x, 0.000001, 0.999999)
            mu_pred, sigma_pred = gp.predict(next_query.reshape(1, -1), return_std=True)
            reasoning = (
                f"F6: Hardest function (7D, best R2=-2.54). Multi-basin search: "
                f"R2 (50%), R8 (20%), R11 (15%), global (15%). Higher UCB weight "
                f"(0.4) for exploration. GP: mu={mu_pred[0]:.4f}, sigma={sigma_pred[0]:.4f}. "
                f"RL analogue: soft-max policy over multiple high-Q states."
            )

        elif func_id == 2:
            # F2: Best R8 (-0.064) at [0.117, 0.562, 0.744]. x3 has r=-0.53.
            # R1 (-0.058) at [0.156, 0.599, 0.732] — same basin.
            # Final: tight exploitation around the R8/R1 basin.
            X_arr = np.array(X[:len(y)])
            y_arr = np.array(y)
            gp = fit_gp_model(X_arr, y_arr, n_dims)
            y_best = min(y)
            best_x = X_arr[np.argmin(y_arr)]  # R8

            n_cand = 30000
            cands_tight = best_x + np.random.uniform(-0.06, 0.06, size=(int(n_cand * 0.6), n_dims))

            r1_x = X_arr[0]
            cands_r1 = r1_x + np.random.uniform(-0.06, 0.06, size=(int(n_cand * 0.2), n_dims))

            cands_scan = np.tile(best_x, (int(n_cand * 0.15), 1))
            cands_scan[:, 0] = np.random.uniform(0.0, 0.25, int(n_cand * 0.15))
            cands_scan[:, 1] = np.random.uniform(0.40, 0.75, int(n_cand * 0.15))
            cands_scan[:, 2] = np.random.uniform(0.65, 0.90, int(n_cand * 0.15))

            cands_global = np.random.uniform(0, 1, size=(int(n_cand * 0.05), n_dims))

            X_cand = np.clip(
                np.vstack([cands_tight, cands_r1, cands_scan, cands_global]),
                0.000001, 0.999999,
            )
            acq = hybrid_acquisition(X_cand, gp, y_best, xi=0.0005, kappa=0.3)
            x0 = X_cand[np.argmax(acq)]

            bounds_f2 = [
                (0.000001, 0.300000), (0.350000, 0.750000), (0.600000, 0.900000),
            ]
            x0 = clip_to_bounds(x0, bounds_f2)
            result = differential_evolution(
                lambda x: -hybrid_acquisition(
                    x.reshape(1, -1), gp, y_best, xi=0.0005, kappa=0.3
                )[0],
                bounds=bounds_f2, seed=42, maxiter=500, x0=x0, tol=1e-14,
            )
            next_query = np.clip(result.x, 0.000001, 0.999999)
            mu_pred, sigma_pred = gp.predict(next_query.reshape(1, -1), return_std=True)
            reasoning = (
                f"F2: Tight exploitation around R8/R1 basin (~0.12, ~0.57, ~0.74). "
                f"x3 (r=-0.53) high. DE bounded [0,0.3]x[0.35,0.75]x[0.6,0.9]. "
                f"GP: mu={mu_pred[0]:.4f}, sigma={sigma_pred[0]:.4f}."
            )

        elif func_id == 7:
            # F7: R10 best (2.7e-6) at [1,0,1,1,0.84,0,0,0]. PCA PC1 r=+0.795.
            # x8 (r=+0.92) strongest: low x8 = good. x1 (r=-0.68): high = good.
            # Final: ultra-tight exploitation at boundary pattern.
            X_arr = np.array(X[:len(y)])
            y_arr = np.array(y)
            gp = fit_gp_model(X_arr, y_arr, n_dims)
            y_best = min(y)
            best_x = X_arr[np.argmin(y_arr)]  # R10

            n_cand = 40000
            cands_tight = best_x + np.random.normal(0, 0.03, size=(int(n_cand * 0.6), n_dims))

            r12_x = np.array(ALL_QUERIES[7][11])
            cands_r12 = r12_x + np.random.normal(0, 0.03, size=(int(n_cand * 0.25), n_dims))

            r8_x = X_arr[7]
            cands_r8 = r8_x + np.random.normal(0, 0.05, size=(int(n_cand * 0.15), n_dims))

            X_cand = np.clip(
                np.vstack([cands_tight, cands_r12, cands_r8]),
                0.000001, 0.999999,
            )
            acq = hybrid_acquisition(X_cand, gp, y_best, xi=0.0001, kappa=0.15)
            x0 = X_cand[np.argmax(acq)]

            bounds_f7 = [
                (0.85, 0.999999), (0.000001, 0.15),
                (0.85, 0.999999), (0.85, 0.999999),
                (0.000001, 0.999999), (0.000001, 0.15),
                (0.000001, 0.15), (0.000001, 0.10),
            ]
            x0 = clip_to_bounds(x0, bounds_f7)
            result = differential_evolution(
                lambda x: -hybrid_acquisition(
                    x.reshape(1, -1), gp, y_best, xi=0.0001, kappa=0.15
                )[0],
                bounds=bounds_f7, seed=42, maxiter=500, x0=x0, tol=1e-14,
            )
            next_query = np.clip(result.x, 0.000001, 0.999999)
            mu_pred, sigma_pred = gp.predict(next_query.reshape(1, -1), return_std=True)
            reasoning = (
                f"F7: Final exploitation at [1,0,1,1,*,0,0,0] boundary pattern. "
                f"x8 (r=+0.92) critical: keep near 0. Tight DE bounds. "
                f"GP: mu={mu_pred[0]:.6f}, sigma={sigma_pred[0]:.6f}. "
                f"RL analogue: greedy on highest-Q state with x5 as free variable."
            )

        elif func_id == 8:
            # F8: R11 best (3.08) at [1,1,1,1,1,1,1,0,0]. R10 (4.28) at [1,0,1,0,*,1,1,1,1].
            # Strong correlations: x1(-0.83), x3(-0.80), x7(-0.73), x6(-0.66).
            # All high except x8(+0.16), x9(+0.25). Final: refine the R11 pattern.
            X_arr = np.array(X[:len(y)])
            y_arr = np.array(y)
            gp = fit_gp_model(X_arr, y_arr, n_dims)
            y_best = min(y)
            best_x = X_arr[np.argmin(y_arr)]  # R11

            n_cand = 40000
            cands_tight = best_x + np.random.normal(0, 0.03, size=(int(n_cand * 0.5), n_dims))

            r12_x = np.array(ALL_QUERIES[8][11])
            cands_r12 = r12_x + np.random.normal(0, 0.04, size=(int(n_cand * 0.2), n_dims))

            r10_x = X_arr[9]
            cands_r10 = r10_x + np.random.normal(0, 0.04, size=(int(n_cand * 0.15), n_dims))

            pattern_11_10_mix = (best_x + r10_x) / 2
            cands_mix = pattern_11_10_mix + np.random.normal(0, 0.05, size=(int(n_cand * 0.15), n_dims))

            X_cand = np.clip(
                np.vstack([cands_tight, cands_r12, cands_r10, cands_mix]),
                0.000001, 0.999999,
            )
            acq = hybrid_acquisition(X_cand, gp, y_best, xi=0.0005, kappa=0.2)
            x0 = X_cand[np.argmax(acq)]

            bounds_f8 = [
                (0.85, 0.999999), (0.000001, 0.999999),
                (0.85, 0.999999), (0.000001, 0.999999),
                (0.000001, 0.999999), (0.85, 0.999999),
                (0.85, 0.999999), (0.000001, 0.999999),
                (0.000001, 0.999999),
            ]
            x0 = clip_to_bounds(x0, bounds_f8)
            result = differential_evolution(
                lambda x: -hybrid_acquisition(
                    x.reshape(1, -1), gp, y_best, xi=0.0005, kappa=0.2
                )[0],
                bounds=bounds_f8, seed=42, maxiter=500, x0=x0, tol=1e-14,
            )
            next_query = np.clip(result.x, 0.000001, 0.999999)
            mu_pred, sigma_pred = gp.predict(next_query.reshape(1, -1), return_std=True)
            reasoning = (
                f"F8: Refining R11 [1,1,1,1,1,1,1,0,0] pattern. "
                f"x1(r=-0.83), x3(r=-0.80), x7(r=-0.73) all high. "
                f"x2,x4,x5,x8,x9 scanned. "
                f"GP: mu={mu_pred[0]:.4f}, sigma={sigma_pred[0]:.4f}."
            )

        else:
            next_query, reasoning, _, _ = generate_final_query(
                func_id, X, y, pca_obj, scaler, pc_corrs, dim_corrs,
            )

        formatted = format_query(next_query)
        round13_queries[func_id] = formatted
        round13_reasoning[func_id] = reasoning

        print(f"    R13 query:   {formatted}")
        print(f"    Reasoning:   {reasoning}")

    # -- Phase 4: Save portal submission -----------------------------------
    print("\n" + "-" * 70)
    print("PHASE 4: PORTAL SUBMISSION FILE")
    print("-" * 70)

    submission_path = Path(__file__).parent / "round_13_portal_submission.txt"
    with open(submission_path, "w") as f:
        f.write("ROUND 13 (FINAL) QUERIES - Module 24\n")
        f.write("=" * 60 + "\n\n")
        f.write("Copy and paste these queries into the capstone portal:\n\n")
        for func_id in range(1, 9):
            f.write(f"Function {func_id}: {round13_queries[func_id]}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("FORMAT: Each value 0.XXXXXX (6 decimal places)\n")
        f.write("=" * 60 + "\n\n")
        f.write("RL-INFORMED REASONING (per function):\n")
        f.write("-" * 60 + "\n\n")
        for func_id in range(1, 9):
            n_dims = FUNCTION_DIMS[func_id]
            best_val = min(ALL_OUTPUTS[func_id])
            f.write(f"Function {func_id} ({n_dims}D):\n")
            f.write(f"  Best result (11 rounds): {best_val:.10f}\n")
            f.write(f"  Strategy: {round13_reasoning[func_id]}\n\n")

    print(f"  Saved to {submission_path}")

    # -- Phase 5: Save inputs_13.txt ----------------------------------------
    results_path = Path(__file__).parent.parent.parent / "results"

    inputs13_entries = []
    for func_id in range(1, 9):
        vals = [float(v) for v in round13_queries[func_id].split("-")]
        inputs13_entries.append(
            f"array([{', '.join(f'{v:.6f}' for v in vals)}])"
        )

    inputs13_path = results_path / "inputs_13.txt"
    inputs12_path = results_path / "inputs_12.txt"
    with open(inputs12_path, "r") as f:
        prev_content = f.read()
    with open(inputs13_path, "w") as f:
        f.write(prev_content)
        f.write("[" + ", ".join(inputs13_entries) + "]\n")
    print(f"  Saved inputs_13.txt to {inputs13_path}")

    # -- Phase 6: Update query_history.json ----------------------------------
    history_path = Path(__file__).parent.parent / "module17" / "query_history.json"
    history = {}
    for func_id in range(1, 9):
        key = f"function_{func_id}"
        entries = []
        for rnd in range(len(ALL_OUTPUTS[func_id])):
            entries.append({
                "round": rnd + 1,
                "query": ALL_QUERIES[func_id][rnd],
                "result": ALL_OUTPUTS[func_id][rnd],
            })
        r12_query = ALL_QUERIES[func_id][11]
        entries.append({"round": 12, "query": r12_query, "result": None})

        r13_vals = [float(v) for v in round13_queries[func_id].split("-")]
        entries.append({"round": 13, "query": r13_vals, "result": None})

        history[key] = entries

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Updated query_history.json at {history_path}")

    # -- Summary ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ROUND 13 (FINAL) SUMMARY")
    print("=" * 70)
    print("\nQueries ready for portal submission:\n")
    for func_id in range(1, 9):
        print(f"  Function {func_id}: {round13_queries[func_id]}")
    print(f"\nPortal submission: {submission_path}")
    print(f"Inputs file:      {inputs13_path}")
    print(f"Query history:    {history_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
