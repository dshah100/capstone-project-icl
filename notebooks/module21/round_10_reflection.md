# Reflection on Transparency and Interpretability (Round 10, 19 Data Points)

---

## 1. What reasoning guided your submission for this tenth round?

My tenth-round strategy uses Gaussian Process Bayesian Optimisation (GP-BO) with Expected Improvement (EI) tuned for strong exploitation (xi=0.001 for functions 1-4, xi=0.003 for functions 5-8). After nine rounds, the GP surrogate has enough data to form reliable posteriors in lower dimensions, so the emphasis shifts from exploration to exploitation.

Patterns from previous rounds directly shaped each submission. **Function 1 (2D)** is solved: all outputs are near zero around (0.55, 0.50), so I submitted a small perturbation to confirm the basin. **Functions 3-4** showed best results at extreme corner regions in round 1, so the GP steered round 10 toward boundary points to probe deeper optima. **Function 5 (6D)** improved dramatically from 251.2 (round 8) to 5.93 (round 9), so the GP aggressively explores that newly discovered basin. **Functions 7-8** have best values near search-space boundaries (many coordinates at 0 or 1), so the GP proposes further boundary-heavy queries. **Function 6 (7D)** has not improved on its round 2 best (-2.54) despite focused queries in rounds 5-7; round 10 queries near the round 2 region with partial variation.

## 2. How transparent is your decision-making process?

The process is fully reproducible. All code uses a fixed random seed (42), deterministic GP fitting (Matern 2.5 kernel, 15 restarts), and deterministic differential evolution for acquisition optimisation. A researcher with the same codebase, query_history.json, and outputs_9.txt could re-run `generate_round10_from_results.py` and produce identical queries. The reasoning for each function is logged alongside the GP's predicted mean and uncertainty at the proposed point. To fully reproduce the strategy, a researcher would need: (a) the complete input/output history, (b) the acquisition function and its hyperparameters, (c) the GP kernel configuration, and (d) the seed values. All are explicitly declared in the code.

## 3. What assumptions are you making?

A key assumption is that each function is reasonably smooth and well-approximated by a GP with a Matern 2.5 kernel. This assumes no sharp discontinuities, highly oscillatory behaviour, or discrete jumps. If a function is piecewise or fractal, the GP will produce misleading confidence intervals and the EI criterion will be poorly calibrated. This assumption particularly limits results for higher-dimensional functions (5-9D), where 9 data points cover an exponentially small fraction of the search space. Additionally, I assume minimisation is the objective, consistent with the negative outputs observed for most functions.

## 4. Where do you see gaps or potential biases?

The most significant gap is severe under-sampling of corner regions in higher dimensions. With 9 queries in 9 dimensions, we have explored roughly 9 out of 512 corners. The GP has observed strong performance at the few corners visited (Functions 4, 7, 8), but better corners may exist unseen. There is also a central-tendency bias from rounds 3-7, where many queries clustered around the centre of [0,1]^n, leaving boundaries under-explored until rounds 8-9. Additionally, functions with larger output magnitudes (F4, F5, F8) dominate the GP's attention in terms of absolute improvement, while subtle improvements in functions like F3 (range 0.06) may be overlooked.

## 5. What is one significant limitation of your approach?

The most significant limitation is the curse of dimensionality combined with a fixed budget of one query per function per round. With only 10 data points in 9 dimensions (Function 8), the GP posterior remains highly uncertain across most of the domain. The Matern kernel assumes a single length-scale per dimension, but different dimensions may have vastly different sensitivities. Our 10 observations are insufficient to estimate these reliably, causing the GP to over-smooth some dimensions while under-smoothing others. This leads to extreme boundary queries, reflecting the model's inability to distinguish true boundary optima from artefacts of high posterior uncertainty near unvisited corners. A more robust approach would incorporate dimension-reduction techniques or multi-fidelity evaluations.
