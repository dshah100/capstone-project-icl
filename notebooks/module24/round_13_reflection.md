# Reflection: RL-Informed Final Round (Round 13, 22+ Data Points)

---

## 1. How has your understanding of the exploration–exploitation trade-off evolved with increasing data?

My understanding shifted from an abstract concept to a quantifiable, data-driven decision framework. In rounds 1-4, with fewer than five observations per function, I had no reliable signal to exploit, so exploration dominated at 80:20. The MAB analogy is direct: each function dimension is an "arm," and with limited pulls I needed wide coverage.

As data accumulated through GP-BO (rounds 5-8), I adopted Expected Improvement, which naturally encodes the trade-off through its xi parameter. I reduced xi from 0.05 to 0.001 — analogous to decaying epsilon in epsilon-greedy policies. The critical inflection came in round 10: correlation analysis revealed boundary queries produced four simultaneous breakthroughs. This was a calculated risk — "exploring" a structural hypothesis rather than blindly sampling. It paid off for F3, F4, F7, F8, but catastrophically for F5 (3,730 vs best of 5.93).

For the final round, I set the balance at 95:5. This mirrors the greedy policy of a mature RL agent: when the Q-table is well-estimated, commit to the highest-value state. My hybrid EI+UCB acquisition provides a small UCB hedge (kappa=0.2) — just enough uncertainty awareness to avoid local minima.

---

## 2. How did the nature of feedback influence your optimisation process?

The portal's feedback — a single scalar per function per round — is structurally identical to RL's reward signal. Like an agent receiving sparse, delayed reward, I never observe gradients; I only see consequences.

I implemented a Q-value analogue: for each function, a discounted cumulative improvement score (gamma=0.9) weights recent gains more heavily, exactly as temporal-difference learning updates Q-values. Functions with high Q-momentum (F8: 1.21, F4: 0.41) justify tighter exploitation. Functions with zero momentum (F2, F6) signal policy stall — the RL response is to increase the exploration bonus, which I applied to F6 (kappa=0.5 vs default 0.2).

F5's catastrophe in R11 is reward shaping gone wrong: the R10 boundary success created a biased reward expectation that generalised poorly. Like an agent encountering deceptive reward, the correct response was policy correction — constrain the search to the proven interior basin, use PCA to identify responsible dimensions — not policy abandonment.

---

## 3. Does your process resemble model-free learning or model-based planning?

My process is predominantly model-based planning. The GP surrogate is an explicit world model: it predicts output (reward) for any candidate input (action) with uncertainty quantification. This parallels AlphaGo Zero's Monte Carlo Tree Search using a learned dynamics model. Each round, my "planning" step evaluates thousands of candidates through the GP, selecting the highest acquisition value.

However, rounds 1-4 resembled model-free learning: random sampling and SVM classifiers acted like trial-and-error exploration. The transition mirrors AlphaGo Zero's progression from random self-play to strategic planning as the value network matures.

The self-play analogy extends further: AlphaGo generates training data by playing itself; I generate GP training data by querying unknown functions. The key difference is my hard budget constraint (one query per function per round), making each "game" expensive and strategic planning essential.

---

## 4. How could RL strategies enhance exploration design in real-world optimisation?

Three RL strategies have direct applicability. First, UCB-based acquisition provides a principled exploration bonus that decays with data density — directly applicable to hyperparameter tuning where budgets are limited. Second, Q-value-inspired region scoring prioritises which search areas to revisit; in drug discovery, this would accelerate hit-to-lead optimisation by focusing synthesis on improving compound families. Third, MAB frameworks enable adaptive experimental design: allocating resources to the most promising arms reduces convergence time by 30-50% compared to grid search, as demonstrated in neural architecture search literature. The core RL insight — that optimal strategies must balance immediate payoff against information gain — is the fundamental challenge in any sequential decision-making process under budget constraints.
