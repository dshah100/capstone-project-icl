# Reflection on My Hyperparameter Tuning Strategy (Round 7, 16 Data Points)

## 1. Which hyperparameters I chose to tune — and why I prioritised them

By the seventh iteration, I focused on tuning the hyperparameters that historically showed the strongest influence on the objective value. These included parameters affecting:

**Learning dynamics** — the exploration parameter `xi` (EI) and `kappa` (UCB), which control exploration versus exploitation.

**Model complexity** — the acquisition function choice (`ei` vs `ucb`) and round-adaptive scaling (`xi * 0.8` in Round 7), which reduces exploration as data grows.

**Regularisation** — the GP’s `alpha` and kernel length scale bounds, which influence surrogate fit.

I prioritised these because they consistently shifted where the next query was proposed. With only limited queries allowed, the aim was to target parameters most likely to influence the curvature of the search space, rather than spreading queries thin across low-impact dimensions.

## 2. How hyperparameter tuning changed my query strategy compared to earlier rounds

In early rounds, my approach resembled exploratory probing: submitting queries that covered broad regions of the search space just to understand general behaviour. As tuning progressed, especially after accumulating 16 data points, my strategy became more structured and evidence-driven.

Hyperparameter tuning shifted my approach from:

- **Wide exploration** → **selective refinement** (round-adaptive scaling reduces `xi` and `kappa`)
- **Guess-based adjustments** → **inference-driven adjustments** (GP surrogate guides queries)
- **Single-parameter moves** → **coordinated multi-parameter tuning** (EI vs UCB across `xi` and `kappa`)

My queries became more intentionally positioned—focusing on promising basins rather than random sweeps.

## 3. Tuning methods I applied and the trade-offs I noticed

Across iterations, I used a blend of tuning methods:

**Manual adjustments** — Used at the very beginning for coarse intuition (e.g., `xi = 0.01` as default).  
*Trade-off:* fast but uninformed; high variance in outcomes.

**Grid search** — `hyperparameter_tuning_comparison` compared `xi` in [0.005, 0.01, 0.02, 0.05] and `kappa` in [1.5, 2.0, 3.0].  
*Trade-off:* good coverage but each comparison requires a new GP fit.

**Bayesian optimisation** — The framework uses a GP surrogate: EI or UCB balances exploration and exploitation. As data grew, this reasoning guided decisions.  
*Trade-off:* more deliberate, but requires sufficient data.

This combination allowed efficient learning without overcommitting to one method.

## 4. Limitations that became clearer as the dataset reached 16 points

With 16 data points, several limitations surfaced:

**a. Model instability or noise** — Some hyperparameters showed inconsistent effects across functions. Functions with very different scales (e.g., Function 1 near-zero vs Function 5 large) made a single strategy harder to generalise.

**b. Diminishing returns** — The more refined the search, the smaller the gains per query.

**c. Overfitting to early regions** — Early points influenced exploration direction; there was a risk of anchoring around initial patterns.

**d. Interaction effects** — `xi` with EI vs `kappa` with UCB behaved differently; non-linear interactions challenged simple tuning.

These insights helped temper expectations and adjust future planning.

## 5. Applying hyperparameter tuning techniques to larger datasets or more complex models

As future rounds introduce more data—or future ML/AI projects involve more complex models—I would adapt tuning techniques in several ways:

- Adopt more Bayesian-style strategies as data increases.
- Narrow search bounds based on earlier patterns.
- Incorporate multi-parameter joint tuning when interaction effects appear.
- Allocate early rounds to exploration, later to exploitation.
- Move toward automated optimisation (Optuna, Hyperband) when manual tuning becomes infeasible.

This prepares me for larger optimisation challenges.

## 6. How tuning in a black-box set-up prepares me for real-world ML/AI practice

The black-box nature of the BBO challenge closely mirrors real ML scenarios where:

- The objective is unclear or noisy
- Internal model structure is inaccessible
- Only outputs are observable
- Data is limited and expensive to collect
- Decisions must be made under uncertainty

This environment forces me to:

- Think probabilistically, not deterministically
- Recognise the value of each data point
- Balance risk and reward when selecting experiments
- Avoid overfitting to partial information
- Develop systematic, iterative tuning strategies
- Remain flexible as patterns emerge

It cultivates the mindset of an ML practitioner who must operate with incomplete information, imperfect signals, and constrained evaluation budgets—exactly what happens in real industrial and research settings.
