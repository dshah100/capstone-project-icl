# BBO Capstone Project

Imperial College Professional Certificate in Machine Learning & Artificial Intelligence

## Project Overview

This repository contains the capstone project for the Imperial College programme. The project is a **Black-Box Optimisation (BBO)** challenge: finding the input vectors that minimise eight unknown functions, each accessible only through a query portal. The functions range from 2 to 9 dimensions, and all inputs lie in [0, 1]^n.

Over 10 iterative rounds, the optimisation strategy evolved from random exploration through SVM and neural-network surrogates to Gaussian Process Bayesian Optimisation with Expected Improvement acquisition.

## Documentation

| Document | Description |
|----------|-------------|
| [Datasheet](DATASHEET.md) | Dataset documentation covering motivation, composition, collection process, preprocessing, and distribution |
| [Model Card](MODEL_CARD.md) | Optimisation approach documentation covering strategy, performance, assumptions, limitations, and ethical considerations |

## Repository Structure

```
capstone-project-icl/
  README.md                  # This file
  DATASHEET.md               # Dataset documentation
  MODEL_CARD.md              # Model card for the optimisation approach
  results/                   # Raw portal inputs and outputs (rounds 1-10)
    inputs_N.txt             # Portal-returned input arrays for round N
    outputs_N.txt            # Portal-returned output values for round N
  notebooks/
    module12/                # Round 1: initial random exploration
    module13/                # Round 2: structured random queries
    module14/                # Round 3: SVM-informed queries
    module15/                # Round 4: neural network + gradient strategy
    module16/                # Round 5: architectural refinement
    module17/                # Round 6: GP Bayesian Optimisation framework
      bbo_capstone_framework.py   # Core GP-BO framework
      query_history.json          # Complete query history (all rounds)
    module18/                # Round 7: GP-BO with exploration
    module19/                # Round 8: GP-BO with boundary probing
    module20/                # Round 9: GP-BO with scaling analysis
    module21/                # Round 10: GP-BO with transparency focus
      generate_round10_from_results.py   # Round 10 query generator
      round_10_portal_submission.txt     # Portal submission queries
      round_10_reflection.md             # Strategy reflection
```

## Results Summary (10 Rounds)

| Function | Dims | Best Output | Best Round | Status |
|----------|------|-------------|------------|--------|
| F1 | 2 | ~0 | 2 | Converged |
| F2 | 3 | -0.0635 | 8 | Improving |
| F3 | 4 | -0.0832 | 1 | Plateau |
| F4 | 5 | -21.3856 | 1 | Plateau |
| F5 | 6 | 5.9314 | 9 | Improving |
| F6 | 7 | -2.5415 | 2 | Plateau |
| F7 | 8 | 0.000421 | 8 | Near-converged |
| F8 | 9 | 7.2348 | 8 | Improving |

## Tools & Technologies

- **Language:** Python 3.12
- **Libraries:** NumPy, scikit-learn (GaussianProcessRegressor), SciPy (differential_evolution)
- **GP Kernel:** Matern 2.5 with automatic relevance determination
- **Acquisition:** Expected Improvement with adaptive xi parameter
- **Environment:** VS Code / Cursor IDE
- **Version Control:** Git / GitHub
