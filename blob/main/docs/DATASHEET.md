# Datasheet: BBO Capstone Project Dataset

## 1. Motivation

**Why was this dataset created?**
This dataset was created as part of the Imperial College Professional Certificate in Machine Learning and Artificial Intelligence capstone project. The task is Black-Box Optimisation (BBO): finding the input vectors that minimise eight unknown functions accessible only through a query portal. Each function accepts a vector in [0, 1]^n and returns a scalar output; the functions range from 2 to 9 dimensions. The dataset captures every query submitted and every output received across 10 iterative rounds of optimisation.

**Who created it and on whose behalf?**
The dataset was created by the student (Devang Shah) for academic assessment. The unknown functions are hosted by Imperial College London through a capstone project portal.

**What tasks does it support?**
The dataset supports surrogate-model-based optimisation, Gaussian Process regression, acquisition function design, and the study of sequential decision-making under uncertainty. It also serves as a case study for transparency and reproducibility in ML experimentation.

## 2. Composition

**What does the dataset contain?**
The dataset consists of 80 observation records (10 rounds x 8 functions). Each record contains:

| Field | Description |
|-------|-------------|
| `round` | Integer 1--10, the submission round |
| `function_id` | Integer 1--8, identifying the unknown function |
| `query` | A list of floats in [0, 1], length equal to the function's dimensionality |
| `result` | A single float returned by the portal |

**Function dimensionalities:**

| Function | Dimensions | Best Result (after 9 rounds) |
|----------|------------|------------------------------|
| F1 | 2 | ~0 (effectively solved) |
| F2 | 3 | -0.0635 |
| F3 | 4 | -0.0832 |
| F4 | 5 | -21.3856 |
| F5 | 6 | 5.9314 |
| F6 | 7 | -2.5415 |
| F7 | 8 | 0.000421 |
| F8 | 9 | 7.2348 |

**Size and format:**
- 80 total observations stored in `query_history.json` (JSON format).
- Raw portal inputs/outputs are in `results/inputs_N.txt` and `results/outputs_N.txt` (N = 1 to 10).
- Portal submission strings are in `notebooks/moduleXX/round_N_portal_submission.txt`.

**Are there any gaps?**
- Rounds 7--8 initially returned error values (0.1 and 0.0) from the portal due to dimension mismatches in the submitted format. The corrected outputs were obtained in the `outputs_9.txt` file, which retrospectively contains the true values for all 9 rounds.
- The raw `inputs_N.txt` files from the portal truncate the last dimension for some functions in rounds 7--9. The authoritative submitted queries are recorded in the portal submission text files.

## 3. Collection Process

**How were queries generated?**
Queries were generated through an evolving strategy across 10 rounds:

| Rounds | Strategy | Description |
|--------|----------|-------------|
| 1 | Random sampling | Initial exploration with NumPy random vectors |
| 2 | Structured random | Broader coverage with deliberate spread |
| 3 | SVM-informed | Used support vector boundaries to guide queries |
| 4 | Neural network + gradient | Gradient-based refinement from neural surrogate |
| 5 | Architectural refinement | Focused perturbations around best-known points |
| 6--10 | GP Bayesian Optimisation | Gaussian Process with Expected Improvement acquisition |

**What tools were used?**
- Python 3.12, NumPy, scikit-learn (GaussianProcessRegressor), SciPy (differential_evolution)
- GP kernel: Matern 2.5 with automatic relevance determination
- Acquisition: Expected Improvement with xi values ranging from 0.001 (exploitation) to 0.01 (exploration)
- All code used fixed random seeds (seed=42) for reproducibility

**Over what time frame?**
Data collection spanned approximately 10 weeks (one round per module), from December 2025 to March 2026.

## 4. Preprocessing and Uses

**Have you applied any transformations?**
- All query values are clipped to [0.000001, 0.999999] to satisfy the portal format (each value must start with 0 and have 6 decimal places).
- Output values are used as-is from the portal; no normalisation is applied to outputs before GP fitting. The GP's `normalize_y=True` parameter handles internal standardisation.
- The `query_history.json` file was rebuilt from authoritative sources (`outputs_9.txt` and portal submission files) in Module 21 to correct earlier rounds where error values (0.0, 0.1) were recorded.

**What are the intended uses?**
- Training surrogate models for black-box function optimisation
- Benchmarking sequential optimisation strategies under extreme budget constraints (10 queries per function)
- Studying the curse of dimensionality in Bayesian optimisation

**What are inappropriate uses?**
- The dataset should not be used to draw conclusions about the true global optima of these functions, as the search space is severely under-sampled (10 points in up to 9 dimensions).
- The outputs should not be treated as ground truth for the functions' full landscapes; they represent sparse point evaluations only.

## 5. Distribution and Maintenance

**Where is the dataset available?**
The dataset is publicly available in this GitHub repository:
- Primary file: `notebooks/module17/query_history.json`
- Raw data: `results/inputs_*.txt` and `results/outputs_*.txt`
- Portal submissions: `notebooks/moduleXX/round_N_portal_submission.txt`

**What are the terms of use?**
This dataset is shared for academic and educational purposes under the terms of the Imperial College capstone project. It may be referenced or reused for non-commercial educational work with attribution.

**Who maintains it?**
The dataset is maintained by Devang Shah as part of the capstone project repository. No further updates are planned after the 10th round submission.
