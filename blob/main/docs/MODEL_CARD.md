# Model Card: BBO Capstone Optimisation Approach

## 1. Overview

| Field | Value |
|-------|-------|
| **Name** | GP-BO Sequential Optimiser |
| **Type** | Bayesian Optimisation with Gaussian Process surrogate |
| **Version** | 3.0 (final, Module 21) |
| **Framework** | Python 3.12, scikit-learn, SciPy |
| **Repository** | This GitHub repository (`notebooks/module21/generate_round10_from_results.py`) |

The approach uses a Gaussian Process (GP) surrogate model with a Matern 2.5 kernel to approximate each unknown black-box function, then selects the next query point by maximising an Expected Improvement (EI) acquisition function. The acquisition optimisation uses a two-stage process: random candidate screening followed by SciPy's differential evolution for local refinement.

## 2. Intended Use

**Suitable tasks:**
- Sequential optimisation of expensive-to-evaluate black-box functions
- Low-budget optimisation (fewer than 20 evaluations per function)
- Continuous optimisation over bounded hypercubes [0, 1]^n
- Functions from 2 to 9 dimensions

**Use cases to avoid:**
- High-dimensional optimisation (n > 15) where GP scaling becomes prohibitive
- Functions with known discontinuities, discrete variables, or combinatorial structure
- Real-time optimisation requiring sub-second decision latency (GP fitting takes seconds)
- Problems where function evaluations are cheap and plentiful (direct search or evolutionary methods would be more efficient)

## 3. Details: Strategy Across 10 Rounds

The optimisation strategy evolved through three distinct phases:

### Phase 1: Exploration (Rounds 1--2)
- **Round 1:** Random sampling using NumPy to establish initial baselines across all 8 functions.
- **Round 2:** Structured random queries designed for broader coverage of the search space.

### Phase 2: Surrogate-Guided Exploration (Rounds 3--5)
- **Round 3:** SVM-informed queries using support vector boundaries from rounds 1--2 to identify promising regions.
- **Round 4:** Neural network surrogate with gradient-based refinement to propose queries in predicted low-output regions.
- **Round 5:** Architectural refinement with focused perturbations around the best-known points from rounds 1--4.

### Phase 3: Bayesian Optimisation (Rounds 6--10)
- **Rounds 6--7:** GP-BO with moderate exploration (xi = 0.007--0.01) to balance exploring new regions and exploiting known good areas.
- **Rounds 8--9:** Increased exploration for higher-dimensional functions (boundary probing), while lower-dimensional functions shifted to exploitation.
- **Round 10:** Strong exploitation bias (xi = 0.001 for F1--F4, xi = 0.003 for F5--F8). The GP posterior was mature enough in lower dimensions to focus on refinement. Higher-dimensional functions received a slightly larger exploration bonus due to the severe under-sampling of their search spaces.

**Key technical details (final version):**

| Component | Configuration |
|-----------|--------------|
| GP Kernel | ConstantKernel * Matern(nu=2.5, ARD) |
| Kernel bounds | Length-scale: [0.001, 1000]; Constant: [0.0001, 10000] |
| GP alpha | 1e-6 (observation noise) |
| Normalise y | True |
| Optimizer restarts | 15 |
| Candidate pool | max(10000, 2000 * n_dims) random samples |
| Refinement | SciPy differential_evolution, maxiter=200, tol=1e-10 |
| Random seed | 42 (fully deterministic) |

## 4. Performance

### Results Summary (Best Values After 9 Rounds)

| Function | Dims | Best Output | Best Round | Trend |
|----------|------|-------------|------------|-------|
| F1 | 2 | ~0 (1e-209) | 2 | Converged by round 2 |
| F2 | 3 | -0.0635 | 8 | Improving; best at boundary |
| F3 | 4 | -0.0832 | 1 | Round 1 still best; 47% convergence gap |
| F4 | 5 | -21.3856 | 1 | Round 1 still best; corner optimum |
| F5 | 6 | 5.9314 | 9 | Major improvement in round 9 |
| F6 | 7 | -2.5415 | 2 | Round 2 still best; 30% convergence gap |
| F7 | 8 | 0.000421 | 8 | Near-zero; boundary optimum |
| F8 | 9 | 7.2348 | 8 | Gradual improvement |

### Metrics Used
- **Primary metric:** Raw function output (minimisation objective).
- **Convergence gap:** Percentage difference between the best value in the last 3 rounds versus the overall best. A gap of 0% means recent rounds are matching or beating the historical best.
- **Improvement rate:** Ratio of best-in-round to overall best across rounds, tracking how quickly the algorithm narrows in on the optimum.

### Key Observations
- Low-dimensional functions (F1, F2) converged rapidly within the first few rounds.
- Several high-dimensional functions (F3, F4, F6) have their best values from early random rounds, suggesting the GP has not yet found better regions than the initial random exploration visited by chance.
- F5 and F7 showed significant late-stage improvements (rounds 8--9), demonstrating the value of boundary exploration in later rounds.

## 5. Assumptions and Limitations

### Assumptions
1. **Smoothness:** The functions are assumed to be smooth and well-modelled by a Matern 2.5 kernel. If any function has sharp discontinuities, the GP posterior will be unreliable.
2. **Stationarity:** The GP assumes stationary covariance (uniform smoothness across the domain). If a function is smooth in one region and rough in another, the single kernel cannot adapt.
3. **Minimisation:** All functions are treated as minimisation problems, consistent with the observed pattern of negative outputs for most functions.
4. **Independent dimensions:** The Matern ARD kernel learns one length-scale per dimension but assumes no explicit interactions between dimensions.

### Limitations
1. **Curse of dimensionality:** With only 10 observations in up to 9 dimensions, the GP posterior is highly uncertain across most of the search space. The ratio of observations to search volume is approximately 10 / 10^9 for F8.
2. **Boundary bias:** The GP's high uncertainty near unvisited boundaries can artificially inflate EI at extreme points, leading to queries at 0 or 1 that may not correspond to true optima.
3. **No early stopping:** The approach does not detect when a function is effectively solved (like F1), wasting a query on confirmation rather than reallocating budget.
4. **Single acquisition function:** EI was used exclusively from round 6 onward. A portfolio approach (combining EI, UCB, and Thompson sampling) could improve robustness.
5. **No transfer learning:** Each function is optimised independently, despite potential structural similarities between functions of adjacent dimensionality.

## 6. Ethical Considerations

### Transparency and Reproducibility
- **Full reproducibility:** Every query is deterministically generated from a fixed seed, documented kernel configuration, and versioned code. Any researcher can clone the repository and reproduce the exact queries.
- **Audit trail:** The `query_history.json` file provides a complete record of every input submitted and every output received, enabling post-hoc analysis of decision quality.
- **Reasoning logs:** Each round's portal submission file includes per-function reasoning (GP predicted mean, uncertainty, and acquisition function parameters), making the decision-making process inspectable.

### Real-World Adaptation
In real-world applications (e.g., hyperparameter tuning, drug design, engineering optimisation), transparent documentation of the optimisation process is essential for:
- **Regulatory compliance:** Demonstrating that decisions were made systematically rather than arbitrarily.
- **Peer review:** Allowing colleagues to validate, critique, and build on the approach.
- **Failure analysis:** When the optimiser produces poor results, the audit trail allows diagnosis of whether the issue was in the model, the acquisition function, or the data.

### Limitations of Transparency
The current documentation captures the "what" and "how" of each decision but is less explicit about the "why" for hyperparameter choices (e.g., why xi=0.001 rather than 0.002). Future iterations could include sensitivity analyses showing how results change under different hyperparameter settings.
