# Reflection on Scaling and Emergent Behaviours (Round 9, 18 Data Points)

*Keep under 700 words for the discussion board.*

---

## 1. Scaling Laws and Query Choices

How do scaling laws influence your current query choices? Do you see diminishing returns or steady improvements?

Scaling laws are now the dominant factor in my query strategy. Across eight rounds of Bayesian optimisation, I observe a clear dimensionality-dependent scaling pattern: low-dimensional functions (2-3D) converged rapidly within the first two rounds, while higher-dimensional functions (5-9D) exhibit diminishing returns with each additional query. Function 1 (2D) reached near-zero loss by round 2 (from ~1.6e-117 to ~4.4e-209), whereas Function 5 (6D) dropped from 147.6 to 11.7 but remains far from its minimum. This mirrors the curse of dimensionality -- the volume of the search space grows exponentially, so each data point covers proportionally less ground in higher dimensions. For Round 9, I have shifted the acquisition function to favour exploitation (xi=0.005), concentrating queries near the best-known regions rather than broadly exploring. With only one query per function per round, diminishing returns are real: Functions 4 and 6 show improvements of less than 5% between recent rounds, suggesting we are approaching the practical limit of what 9 data points can resolve in a 5-7D space.

---

## 2. Emergent Behaviours

Where might emergent behaviours alter your expectations, and how are you preparing for them?

Several functions have exhibited emergent-like behaviour -- sudden, disproportionate improvements that were not predicted by the gradual GP surrogate trend. Function 7 (8D) jumped from a loss of 0.384 in Round 1 to 0.002 in Round 2, a 170x improvement from a single exploratory query that happened to land near a narrow basin. Function 5 (6D) showed the opposite: after improving to 11.7 by Round 4, Rounds 5-6 returned worse values (40.1, 15.4), suggesting the landscape contains local traps or sharp discontinuities. These phase-transition-like jumps resemble the emergent capabilities seen in large language models, where a quantitative increase in scale (data) produces a qualitative shift in performance. I am preparing by maintaining a dual strategy: the GP's Expected Improvement acquisition naturally trades off exploitation against exploration, but I also monitor the improvement trajectory per function. Where the trajectory is flat, I allow the GP to explore more aggressively (higher kappa in future rounds); where an emergent jump was observed, I tighten the search radius to exploit the discovered basin before it is lost to surrogate model drift.

---

## 3. Cost, Robustness and Performance Trade-offs

What trade-offs between cost, robustness and performance are shaping your strategy now?

With only one query per function per round, every observation is expensive. The primary trade-off is between robustness (ensuring we have not over-fitted the GP to a local minimum) and raw performance (driving the best-known value lower). For well-converged functions like Function 1 and Function 7, the cost of further queries yields negligible performance gain -- their budgets would be better spent on Functions 4, 5 and 8 which still have large residual losses (3.06, 11.7, and 7.59 respectively). However, the capstone format allocates one query per function, so I cannot reallocate budget across functions. Instead, I shape robustness through the acquisition parameters: low xi (0.005) keeps Round 9 queries close to proven regions, sacrificing potential breakthroughs for reliable incremental gains. The GP convergence warnings in high dimensions (length-scale parameters hitting bounds) also signal that the surrogate model itself is becoming less trustworthy as dimensionality increases, reinforcing the need for robust rather than aggressive query placement.

---

## 4. Predictable Optimisation vs Emergent Risk

How do you balance predictable optimisation with the risk of sudden but uneven emergent capabilities?

The tension between predictable optimisation and emergent risk is most visible when comparing Function 6's steady descent (1.52 -> 1.42 -> 1.14 -> 0.91 -> 0.90) with Function 7's erratic jumps (0.38 -> 0.002 -> 0.17 -> 0.28 -> 0.31). Predictable functions reward a pure exploitation strategy, while emergent functions punish it by rapidly shifting the optimal region. My balance point is the Expected Improvement criterion itself: by construction, EI gives nonzero weight to points where the GP uncertainty is high, even when a low-uncertainty exploitation point exists nearby. This provides a built-in hedge against emergent risk. For Round 9 specifically, I chose a conservative xi of 0.005 because, with 8 rounds of data, the GP posterior is more informed and the risk of missing a new basin is lower than in early rounds. The scaling analysis confirms that most functions have entered a diminishing-returns regime, making predictable exploitation the higher-expected-value strategy at this stage. If Round 9 reveals another emergent jump, I would increase xi in Round 10 to widen the search net accordingly.
