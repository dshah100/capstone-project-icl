# BBO Capstone Project - Round 6 Submission Guide

## 📋 Overview

This guide helps you complete **Stage 2: Round 6** of the Black-Box Optimization capstone project, including:
- **Part 1:** Generate and submit Round 6 queries (8 functions)
- **Part 2:** Reflect on BBO strategy with CNN parallels (< 700 words)

## 🚀 Quick Start (Choose Your Path)

### Path A: I Have Portal Results from Rounds 1-5 ✓

If you've already submitted queries and received results from the portal:

```bash
# Step 1: Run the quick generator
python quick_round6_generator.py

# Step 2: Enter your portal results when prompted
# Step 3: Get your Round 6 queries automatically
# Step 4: Review round_6_portal_submission.txt
```

### Path B: I Need to Understand the Framework First 📚

If you're new or want to understand the methodology:

```bash
# Step 1: Review the main framework
# Open: bbo_capstone_framework.py

# Step 2: Create data entry template
python bbo_capstone_framework.py

# Step 3: Fill query_history_template.json with portal results
# Step 4: Use the framework to generate queries
```

### Path C: I Don't Have Previous Results Yet ⚠️

If you haven't completed Rounds 1-5:

```bash
# Step 1: Run with mock data to understand the process
python quick_round6_generator.py
# Choose "n" when asked for portal results

# Step 2: Submit the initial queries from 1st.log
# Step 3: Get portal results and update query_history.json
# Step 4: Re-run for accurate Round 6 queries
```

## 📁 File Structure

```
capstone-project-icl/
│
├── 1st.log                           # Your initial Round 1 queries
├── pricinginsurance.py               # Original pricing model generator
├── pricinginsurance2D.py             # 2D Bayesian Optimization example
├── pricinginsurance3D.py             # 3D Bayesian Optimization example
│
├── bbo_capstone_framework.py         # ⭐ Main BBO framework (comprehensive)
├── quick_round6_generator.py         # ⭐ Quick query generator (easy to use)
│
├── query_history.json                # Your accumulated portal results
├── round_6_queries_[timestamp].txt   # Generated Round 6 queries
├── round_6_portal_submission.txt     # Copy-paste ready for portal
│
├── round_6_reflection.md             # ⭐ Reflection template (Part 2)
├── README_ROUND6.md                  # This file
│
└── function_*_progress.png           # Optimization visualizations
```

## 🔧 Detailed Instructions

### Part 1: Generate Round 6 Queries

#### Step 1: Prepare Your Data

You need the **distance results** from the capstone portal for your previous queries.

**Option A: Use quick_round6_generator.py** (Recommended)

```bash
python quick_round6_generator.py
```

This will:
1. Guide you through entering portal results
2. Save data to `query_history.json`
3. Generate Round 6 queries automatically
4. Create visualizations of your progress

**Option B: Manual JSON Entry**

Create/edit `query_history.json`:

```json
{
  "function_1": [
    {
      "round": 1,
      "query": [0.374546, 0.950714],
      "result": 0.850000
    },
    {
      "round": 2,
      "query": [0.123456, 0.789012],
      "result": 0.432100
    }
  ],
  "function_2": [
    ...
  ]
}
```

#### Step 2: Generate Queries

**Using Python:**

```python
from bbo_capstone_framework import BBOCapstoneManager

# Initialize
manager = BBOCapstoneManager()

# Load your history
manager.load_query_history_from_file('query_history.json')

# Generate Round 6 queries
queries = manager.generate_round_queries(
    round_num=6,
    acquisition='ei',  # Expected Improvement
    xi=0.01            # Exploration parameter
)

# Save for portal submission
manager.save_queries_to_file(queries, round_num=6)

# Visualize progress
for func_id in range(1, 9):
    manager.visualize_progress(func_id)
```

#### Step 3: Submit to Portal

1. Open `round_6_portal_submission.txt`
2. Copy each query in the format: `x1-x2-x3-...-xn`
3. Go to your capstone portal (link from Module 12 email)
4. Paste queries for each function
5. Submit and record results

**Query Format Requirements:**
- Format: `x1-x2-x3-...-xn`
- Each value: `0.XXXXXX` (6 decimal places)
- Example 2D: `0.123456-0.654321`
- Example 3D: `0.111111-0.222222-0.333333`

### Part 2: Write Your Reflection

#### Step 1: Open the Template

```bash
# Edit this file:
round_6_reflection.md
```

#### Step 2: Complete Each Section

The template includes 5 prompts with guidance:

1. **Progressive Feature Extraction** - How hierarchical learning influenced your strategy
2. **Breakthrough Parallels** - LeNet evolution vs. your BBO improvements
3. **Trade-offs** - Balancing exploration/exploitation like CNN depth/overfitting
4. **Learning Components** - CNN concepts (convolution, pooling, etc.) in BBO
5. **Real-World Deployment** - Edge AI lessons for BBO success metrics

**Tips:**
- Use specific examples from YOUR data (functions, distances, rounds)
- Connect each CNN concept to concrete BBO decisions you made
- Keep total under 700 words
- Be honest about challenges and learnings

#### Step 3: Submit to Discussion Board

1. Post your reflection (< 700 words)
2. Engage with at least 2-3 peers' posts
3. Look for different BBO strategies to discuss

## 🧠 Understanding the BBO Strategy

### What is Bayesian Optimization?

Your framework uses **Gaussian Process + Expected Improvement**:

1. **Gaussian Process (GP)**: Probabilistic model that learns the objective function
   - Inputs: Your previous queries (X)
   - Outputs: Portal results/distances (y)
   - Prediction: Mean (expected distance) + Uncertainty (σ)

2. **Expected Improvement (EI)**: Acquisition function that balances:
   - **Exploitation**: Query near best known point
   - **Exploration**: Query in uncertain regions
   - Formula: `EI = (y_best - μ) * Φ(Z) + σ * φ(Z)`

3. **Iteration**: Each round:
   - Fit GP to all previous observations
   - Calculate EI across candidate points
   - Select point with maximum EI
   - Query portal, get result, add to dataset

### Why This Strategy?

**Advantages:**
- Efficient with limited queries (only 6 rounds!)
- Handles high dimensions (up to 9D for Function 8)
- Provides uncertainty estimates
- Principled exploration-exploitation balance

**Key Parameters:**

| Parameter | Value | Effect |
|-----------|-------|--------|
| `xi` | 0.01-0.02 | Higher = more exploration |
| `kappa` | 2.0 | UCB exploration (alternative to EI) |
| `kernel` | Matern(ν=2.5) | Smoothness assumption |
| `alpha` | 1e-6 | Noise level / regularization |

## 📊 Interpreting Your Results

### Progress Visualization

The `function_X_progress.png` plots show:
- **Blue circles**: Distance at each query round
- **Red squares**: Best distance so far
- **Trend**: Should generally decrease (getting closer to optimum)

### Good Signs:
✓ Best distance improving each round
✓ Queries converging toward a region
✓ Uncertainty decreasing in promising areas

### Warning Signs:
⚠️ No improvement after Round 3-4 (stuck in local minimum)
⚠️ Queries jumping around randomly (GP not learning)
⚠️ Very slow convergence (need more exploration)

### If Stuck:
1. Increase `xi` parameter (more exploration)
2. Try UCB acquisition instead of EI
3. Check if queries are too similar (numerical precision issue)
4. Review portal results for errors

## 🔬 Advanced: Customizing Your Strategy

### Experiment with Acquisition Functions

```python
# Expected Improvement (default)
queries_ei = manager.generate_round_queries(
    round_num=6,
    acquisition='ei',
    xi=0.01  # Lower = exploit, Higher = explore
)

# Upper Confidence Bound (alternative)
queries_ucb = manager.generate_round_queries(
    round_num=6,
    acquisition='ucb',
    kappa=2.0  # Lower = exploit, Higher = explore
)
```

### Adapt Strategy by Function

```python
# More exploration for high-dimensional functions
for func_id in [7, 8]:  # 8D and 9D functions
    optimizer = manager.optimizers[func_id]
    next_query = optimizer.propose_next_query(
        acquisition='ei',
        xi=0.02  # Double exploration
    )

# More exploitation for low-dimensional functions
for func_id in [1, 2]:  # 2D and 3D functions
    optimizer = manager.optimizers[func_id]
    next_query = optimizer.propose_next_query(
        acquisition='ei',
        xi=0.005  # Half exploration
    )
```

### Kernel Selection

```python
# Try different kernels
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic

# Smoother assumption (good for continuous functions)
kernel = RBF(length_scale=1.0)

# Less smooth (good for functions with discontinuities)
kernel = Matern(nu=1.5, length_scale=1.0)

# Very flexible
kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)
```

## ❓ FAQ

### Q: I don't have results from previous rounds. Can I still generate Round 6 queries?

**A:** Not accurately. You need actual portal results to train the Gaussian Process. However, you can:
1. Use mock data to understand the process
2. Submit your Round 1 queries (from `1st.log`)
3. Work through Rounds 2-5 quickly
4. Then generate accurate Round 6 queries

### Q: Should I use EI or UCB?

**A:** Expected Improvement (EI) is generally better for:
- Limited query budgets (like our 6 rounds)
- Noisy functions
- When you want principled exploration-exploitation balance

UCB is good for:
- Quick decisions
- When you have strong prior about the function
- More aggressive exploration (higher kappa)

### Q: How do I know if my queries are good?

**A:** Check:
1. **Diversity**: Early rounds should spread across space
2. **Convergence**: Later rounds should cluster near best region
3. **Improvement**: Distance should generally decrease
4. **GP confidence**: High uncertainty should decrease over rounds

### Q: My Function X isn't improving. What should I do?

**A:** Diagnose:
1. Plot the observations (visualize_progress)
2. Check if queries are too similar (precision issue)
3. Increase exploration parameter (higher xi or kappa)
4. Verify portal results are correct (no entry errors)

### Q: The reflection asks for specific examples. What should I include?

**A:** For each prompt, reference:
- Specific function IDs and dimensions
- Actual distance values from your results
- Rounds where you changed strategy
- Concrete decisions you made (e.g., "increased xi from 0.01 to 0.02")
- Observations from your progress plots

## 📚 Additional Resources

### Understanding Gaussian Processes
- [A Visual Exploration of GP](https://distill.pub/2019/visual-exploration-gaussian-processes/)
- [GP Regression Tutorial](https://scikit-learn.org/stable/modules/gaussian_process.html)

### Bayesian Optimization
- [A Tutorial on Bayesian Optimization](https://arxiv.org/abs/1807.02811)
- [Bayesian Optimization Book](https://bayesoptbook.com/)

### CNNs (for reflection context)
- LeNet, AlexNet, VGG, ResNet architectures
- Edge AI deployment challenges
- Convolution, pooling, activation functions

## ✅ Checklist

### Part 1: Query Submission
- [ ] Collected portal results from Rounds 1-5
- [ ] Created/updated `query_history.json`
- [ ] Generated Round 6 queries
- [ ] Verified query format (6 decimal places, 0.XXXXXX)
- [ ] Submitted queries to portal
- [ ] Recorded results for future rounds

### Part 2: Reflection
- [ ] Completed all 5 prompts
- [ ] Used specific examples from MY data
- [ ] Connected CNN concepts to BBO strategy
- [ ] Kept under 700 words
- [ ] Posted to discussion board
- [ ] Engaged with peer posts

### Bonus
- [ ] Generated progress visualizations
- [ ] Analyzed which functions improved most
- [ ] Identified patterns in successful strategies
- [ ] Prepared questions for next module

## 🎯 Success Metrics

Evaluate your Round 6 performance:

1. **Query Efficiency**: (Initial distance - Current distance) / 6
2. **Improvement Rate**: Compare Round 6 to Round 5
3. **Convergence**: Are your best queries clustering?
4. **Uncertainty**: Is your GP becoming more confident?
5. **Ranking**: How do you compare to peers (from discussion board)?

## 💡 Tips for Success

1. **Don't rush**: Understanding the strategy is more valuable than quick submission
2. **Visualize**: Always plot your progress before submitting new queries
3. **Document**: Keep notes on why you made each decision
4. **Engage**: Learn from peers' different strategies
5. **Iterate**: Use Round 7+ to refine based on Round 6 insights

## 📝 Reflection Writing Tips

### Do:
✓ Use concrete numbers from your data
✓ Explain your reasoning for decisions
✓ Connect CNN concepts to specific BBO actions
✓ Discuss both successes and challenges
✓ Show evolution of your thinking

### Don't:
✗ Generic statements without examples
✗ Just describing CNNs without BBO connection
✗ Claiming perfect strategy (be honest!)
✗ Ignoring the 700-word limit
✗ Copying template text verbatim

## 🆘 Getting Help

If you're stuck:

1. **Check visualizations**: Often reveal the issue
2. **Review query_history.json**: Ensure data is correct
3. **Try different parameters**: Adjust xi or kappa
4. **Simplify first**: Get 2D functions working first
5. **Ask peers**: Discussion board is valuable
6. **Document**: Write down what you tried

## 🎓 Learning Objectives Alignment

This activity addresses:
- ✅ **Modify optimization strategy**: By adjusting acquisition functions
- ✅ **Interpret appropriateness**: Reflecting on when BBO suits problems
- ✅ **CNN parallels**: Connecting hierarchical learning to BBO refinement
- ✅ **Performance improvement**: Demonstrating convergence over rounds

---

**Good luck with Round 6! Remember: The journey of understanding your optimization strategy is more valuable than just finding the minimum distance.**

**Questions?** Document them in your reflection or bring them to the discussion board!
