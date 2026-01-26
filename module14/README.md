# capstone-project-icl
Capstone Project

Capstone Project Repository
# Black-Box Optimisation: Insurance Pricing Challenge

## Section 1: Project Overview
The Black-Box Optimisation (BBO) capstone project involves finding the optimal parameters for eight unknown insurance pricing functions. The objective is to navigate high-dimensional spaces—ranging from 2D to 9D—where the underlying mathematical structure and noise levels are entirely hidden.

In real-world Machine Learning, this mirrors scenarios where data is expensive or slow to acquire, such as tuning complex industrial systems or financial risk models. This project builds critical skills in iterative strategy refinement, uncertainty-aware modeling, and efficient search under incomplete knowledge.

## Section 2: Inputs and Outputs
The system operates through a structured query-and-response loop:
* **Inputs**: A coordinate string in the format `x1-x2-x3-...-xn`.
* **Constraints**: Each `xi` must begin with `0` and be specified to exactly six decimal places (e.g., `0.123456-0.654321`).
* **Dimensions**: Eight distinct functions with dimensionality $D \in \{2, 3, 4, 5, 6, 7, 8, 9\}$.
* **Outputs**: A response value (performance signal) indicating the objective score of the queried coordinate.

## Section 3: Challenge Objectives
The primary goal is to **maximise** the response value for each of the eight functions.
* **Efficiency**: Success must be achieved within a strictly limited number of query rounds.
* **Robustness**: The approach must account for unknown function structures and potential noise (heteroskedasticity) while managing the "curse of dimensionality".

## Section 4: Technical Approach
My strategy has evolved iteratively over three submission rounds to balance exploration of the unknown space with exploitation of high-performing regions.

### Round 1 & 2: Broad Exploration and Space-Filling
Initially, I utilized broad exploratory sampling to establish a baseline and identify promising regions. For the second iteration, I moved to a **structured space-filling strategy**. I applied diagnostics using linear and logistic regression to test if local behavior appeared smooth or thresholded, which guided whether to continue exploring or start refining.

### Round 3: SVM-Informed Exploitation
As the dataset grew, I integrated **Support Vector Machines (SVMs)** to refine the search process:
* **Soft-Margin SVMs**: Used to classify regions into high- vs. low-performing areas while accounting for noisy observations.
* **Kernel SVMs**: Specifically RBF kernels were used to capture non-linear decision boundaries on complex response surfaces.

### Balancing Logic
My current approach leans toward **exploitation** by sampling near consistently strong regions. However, I retain a portion of queries for untested areas (exploration) to prevent premature convergence on local optima. This allows for identifying irrelevant dimensions and focusing resources on the most sensitive features.