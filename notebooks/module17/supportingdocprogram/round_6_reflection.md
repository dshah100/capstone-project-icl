# Round 6 Reflection: CNN Insights Applied to Black-Box Optimization

**Student:** [Your Name]  
**Date:** February 6, 2026  
**Round:** 6 (15 data points accumulated)

---

## Part 1: Progressive Feature Extraction

**Prompt:** *CNNs build up features from edges and textures to full objects. How did this idea of progressive feature extraction influence the way you thought about refining your BBO strategy?*

### My Response:

The concept of progressive feature extraction from CNNs profoundly influenced my BBO strategy refinement. Just as CNNs build understanding hierarchically—starting with simple edges, progressing to textures, then complex shapes, and finally full objects—my optimization strategy evolved through distinct phases:

**Early Rounds (Rounds 1-2): Edge Detection Phase**
- Initial queries explored the search space broadly, like a CNN's first convolutional layer detecting basic edges
- These queries provided rough "gradients" indicating promising directions in the high-dimensional space
- For example, in Function X [replace with your function], the initial queries revealed [specific insight from your data]

**Middle Rounds (Rounds 3-4): Texture Recognition Phase**
- With accumulated data, I began recognizing "patterns" or "textures" in the objective function landscape
- The Gaussian Process model started identifying regions with consistent behavior (smooth valleys, sharp peaks)
- This parallels how intermediate CNN layers combine edges into textures—I combined point observations into regional understanding

**Recent Rounds (Rounds 5-6): Object Recognition Phase**
- Now with 15 data points, I can "see" the global structure more clearly
- My Expected Improvement acquisition function balances exploration (finding new features) with exploitation (refining known good regions)
- This mirrors how deep CNN layers integrate all previous features to recognize complete objects

**Key Insight:** Just as you cannot train a CNN to recognize faces using only edge detectors, I cannot find the optimum using only local information. The progressive accumulation of observations across rounds builds a hierarchical understanding of the objective function, enabling increasingly sophisticated query decisions.

[Add specific examples from your functions and how your strategy evolved]

---

## Part 2: Breakthrough Parallels - LeNet to Modern CNNs

**Prompt:** *LeNet and later CNNs redefined what is possible in computer vision. What parallels do you see between those breakthroughs and the incremental improvements you make in your BBO capstone project?*

### My Response:

The evolution from LeNet to modern CNNs offers striking parallels to my BBO journey:

**Architectural Evolution vs. Strategy Evolution**
- LeNet introduced the fundamental convolution-pooling architecture
- My Round 1 established the fundamental explore-exploit framework using Gaussian Processes
- Just as AlexNet improved on LeNet with deeper networks and ReLU activations, my later rounds refined the acquisition function and exploration parameters

**Incremental Gains Through Better Understanding**
- ImageNet breakthrough: 15.3% to 10.8% error (AlexNet) to 3.6% (ResNet)—not revolutionary jumps, but steady progress
- My BBO project: Function X improved from [distance in Round 1] to [distance in Round 6]
- Both demonstrate that systematic, incremental refinement based on accumulated knowledge leads to significant cumulative improvement

**Transfer Learning Parallel**
- Modern CNNs leverage pre-trained weights from large datasets
- In Round 6, I'm leveraging "pre-trained" knowledge from my 15 previous observations
- My Gaussian Process model encodes learned patterns, similar to how transfer learning encodes learned visual features

**Dimensionality Challenge**
- LeNet worked on 32×32 images (1,024 dimensions)
- My Functions 7-8 operate in 8-9 dimensional spaces
- Modern CNNs handle ImageNet's millions of parameters through clever architecture
- My BBO strategy handles high dimensionality through Gaussian Process kernels that capture correlations

**Key Parallel:** Both CNN evolution and my BBO progress show that breakthrough performance comes not from a single "trick" but from systematically combining multiple incremental improvements: better models (deeper CNNs / refined GP kernels), better data (ImageNet / accumulated queries), and better optimization (Adam / adaptive acquisition functions).

[Add specific performance improvements from your data]

---

## Part 3: Balancing Depth, Cost, and Overfitting

**Prompt:** *Training CNNs often involves balancing depth, computational costs and overfitting risks. Did you face similar trade-offs when choosing whether to explore widely or exploit promising regions in your queries?*

### My Response:

Yes, I encountered remarkably similar trade-offs in my BBO strategy:

### Trade-off 1: Model Complexity vs. Overfitting

**CNNs:**
- Deeper networks capture more complex patterns but risk overfitting to training data
- Regularization (dropout, batch norm) prevents overfitting
- Must validate on held-out test set

**My BBO:**
- More complex GP kernels (Matern vs. RBF) capture more intricate function landscapes
- Risk: With only 15 observations, an overly complex kernel might overfit to noise
- My solution: Use Matern kernel with ν=2.5 (moderate smoothness assumption) and normalize targets
- Like CNN dropout, my GP's noise parameter (alpha=1e-6) provides regularization

**Concrete Example:** [Describe a case where you adjusted kernel complexity based on observed behavior]

### Trade-off 2: Exploration vs. Exploitation

**CNNs:**
- Must explore different architectures (ResNet, DenseNet, Transformers)
- But also exploit known good designs (VGG backbone)
- Learning rate schedules balance exploration (high LR) vs. exploitation (low LR)

**My BBO:**
- Must explore unseen regions to avoid local minima
- But also exploit promising regions identified by previous queries
- My Expected Improvement acquisition function's ξ parameter controls this balance

**My Strategy Evolution:**
- **Rounds 1-2:** High exploration (ξ = 0.02) to map the landscape broadly
- **Rounds 3-4:** Balanced approach (ξ = 0.01) as patterns emerged
- **Rounds 5-6:** Moderate exploration with refined exploitation in promising regions

**Impact:** For Function X, early exploration prevented premature convergence to a local minimum, while later exploitation refined the best region found.

### Trade-off 3: Computational Budget

**CNNs:**
- Training deeper networks requires more GPU hours
- Must decide: train one very deep model or ensemble of shallower models?

**My BBO:**
- Each query "costs" one evaluation (limited budget, like compute hours)
- Must decide: spread queries across search space or concentrate in best region?
- Using Bayesian optimization maximizes information gained per expensive evaluation

**Decision Framework:**
- Function X (2D): Easier to explore broadly with limited queries
- Function X (9D): High dimensionality requires strategic query placement
- Adapted strategy: Higher exploration in high-D functions to avoid missing promising regions

[Add specific examples of how you adjusted your strategy based on these trade-offs]

---

## Part 4: Learning from Data - Convolutions, Pooling, Activations, Loss

**Prompt:** *Convolutions, pooling, activations and loss functions influence how CNNs learn from data. Which of these concepts helped you think differently about how your optimisation model learns from your accumulated data?*

### My Response:

Each CNN component offered valuable insights for my BBO strategy:

### Convolution: Local Pattern Detection

**CNN Concept:**
- Convolutional filters detect local patterns (edges, textures)
- Shared weights across spatial positions enable translation invariance
- Multiple filters capture different patterns

**My BBO Application:**
- My Gaussian Process kernel acts like a "convolution" operator, identifying local smoothness patterns
- The kernel's length scale determines the "receptive field"—how far correlations extend
- Multiple kernel components (Constant × Matern) capture different aspects: overall scale and local smoothness

**Concrete Insight:** In Function X, I noticed that [specific observation]. This led me to adjust the kernel length scale, similar to how you'd adjust CNN filter sizes for different image resolutions.

### Pooling: Dimensionality Reduction

**CNN Concept:**
- Max/average pooling reduces spatial dimensions while retaining important features
- Creates translation invariance and reduces computation
- Must balance information preservation vs. efficiency

**My BBO Application:**
- In high-dimensional functions (7D, 8D, 9D), I cannot afford to densely sample the space
- My GP effectively performs "pooling" by interpolating between sparse observations
- The model learns which dimensions matter most (via length scale parameters for each dimension)

**Strategy:** For Function X (XD), my GP learned that dimensions [X, Y] vary more smoothly than dimension [Z], similar to how CNNs learn which features to pool more aggressively.

### Activation Functions: Non-linearity

**CNN Concept:**
- ReLU, Sigmoid, Tanh introduce non-linearity
- Enable learning complex decision boundaries
- ReLU's simplicity (max(0, x)) aids optimization and prevents vanishing gradients

**My BBO Application:**
- My objective functions are highly non-linear (unknown black boxes)
- The GP's kernel introduces non-linearity through the Matern function
- Expected Improvement acquisition function's use of Normal CDF/PDF adds another layer of non-linearity in decision-making

**Insight:** Just as ReLU works well despite simplicity, my relatively simple Matern kernel with ν=2.5 proved effective across all 8 functions. Over-complicating the model (like exotic activation functions) didn't improve performance given limited data.

### Loss Function: Learning Objective

**CNN Concept:**
- Cross-entropy loss for classification guides learning
- Loss landscape determines optimization difficulty
- Focal loss addresses class imbalance

**My BBO Application:**
- My "loss function" is the distance to the unknown optimum (measured by the portal)
- Unlike CNNs where I can compute loss gradients, I only get the loss value (black-box)
- This makes each query extremely valuable—no backpropagation shortcuts!

**Key Realization:** CNNs can evaluate millions of training examples with backprop. My BBO can only afford 6 queries per function. This scarcity fundamentally changes the optimization strategy—I must maximize information gain per query, leading to Bayesian optimization rather than gradient descent.

[Add specific examples from your optimization runs]

---

## Part 5: Real-World Deployment - Edge AI and BBO Success Metrics

**Prompt:** *The interview with Andrea Dunbar highlighted the trade-offs of deploying CNNs in edge AI systems. How might reflecting on real-world deployment challenges help you decide how to benchmark success in your own BBO capstone project?*

### My Response:

Andrea Dunbar's insights on edge AI deployment reveal crucial parallels for evaluating my BBO success:

### Resource Constraints

**Edge AI Challenge:**
- Limited compute: Must compress CNNs (pruning, quantization) to fit on mobile/IoT devices
- Accuracy vs. efficiency trade-off: 95% accuracy at 200ms vs. 90% accuracy at 50ms
- Battery life considerations affect model complexity choices

**My BBO Parallel:**
- Limited query budget: Only 6 rounds to find optimum across 8 functions
- Accuracy vs. efficiency trade-off: Perfect optimum vs. "good enough" solution within budget
- Must strategically allocate queries across functions (like allocating compute across model components)

**Success Metric Implication:** Instead of just measuring "distance to optimum," I should benchmark:
1. **Query efficiency:** Distance improvement per query (like accuracy per FLOP in edge AI)
2. **Relative ranking:** How do my solutions compare to classmates using same query budget?
3. **Convergence rate:** Am I improving faster in later rounds (learning efficiency)?

### Latency and Real-Time Requirements

**Edge AI Challenge:**
- Inference must be fast enough for real-time applications (autonomous vehicles, AR)
- May sacrifice some accuracy for speed
- Need confidence estimates for safety-critical decisions

**My BBO Parallel:**
- In real-world insurance pricing (the domain context), decisions have time constraints
- A pricing model that takes months to optimize isn't practical
- My 6-round constraint simulates real-world time limits

**Revised Success Metric:** Measure "solution quality at Round 6" rather than "can I eventually find the optimum given infinite queries." This reflects real deployment constraints where you must act on best available solution, not wait for perfect solution.

### Robustness and Uncertainty Quantification

**Edge AI Challenge:**
- Models must handle distribution shift (new environments, lighting conditions)
- Uncertainty estimates critical for safety (should the car's vision system flag low confidence?)
- Andrea emphasized monitoring model performance post-deployment

**My BBO Parallel:**
- My GP provides uncertainty estimates (standard deviation σ)
- High σ regions indicate where model is unsure—similar to CNN confidence scores
- Expected Improvement uses σ to balance confidence in exploitation vs. exploring uncertain areas

**Enhanced Success Metrics:**
1. **Uncertainty reduction:** How much has my GP's uncertainty decreased across the search space?
2. **Robustness:** Do my best solutions cluster in a stable region (robust optimum) or jump around (unstable)?
3. **Confidence:** What's my GP's predicted distance at my best query? How does it compare to actual distance?

### Practical Impact

**Edge AI:**
- Success isn't just accuracy—it's deployment success
- User experience, power consumption, cost all matter
- Model must work in the wild, not just on test sets

**My BBO:**
- In real insurance pricing, "success" means:
  - Competitive premiums (not too high)
  - Profitable portfolios (not too low)
  - Fast enough for quote generation
  - Explainable to regulators

**Final Success Benchmark Framework:**

| Metric Category | Traditional Metric | Real-World Metric (Edge AI Inspired) |
|----------------|-------------------|--------------------------------------|
| **Accuracy** | Minimum distance found | Distance vs. query budget efficiency |
| **Efficiency** | Final round performance | Convergence rate across rounds |
| **Reliability** | Best single query | Consistency of top 3 queries |
| **Uncertainty** | N/A | GP uncertainty at final solution |
| **Comparative** | Absolute distance | Ranking vs. peers with same budget |

**Concrete Example:** For Function X, instead of just reporting "final distance = 0.XXX," I benchmark:
- Query efficiency: (Initial distance - Final distance) / 6 queries = X improvement per query
- Convergence: Distance in Round 6 vs. Round 3 (accelerating or plateauing?)
- Robustness: Standard deviation of my 3 best queries (clustered or scattered?)
- Uncertainty: My GP's σ at the final query (confident or guessing?)

This multi-faceted evaluation, inspired by real-world deployment constraints, provides a more honest assessment than a single distance metric.

---

## Conclusion

Reflecting on CNN principles has fundamentally improved my BBO strategy:

1. **Progressive refinement** (like hierarchical features) replaced random search
2. **Incremental improvements** (like LeNet → ResNet) compound into significant gains
3. **Explicit trade-off management** (depth/cost/overfitting) guides each decision
4. **Component-level understanding** (convolution/pooling/activation/loss) informs model design
5. **Real-world constraints** (like edge AI deployment) define meaningful success metrics

**Key Takeaway:** The most valuable lesson from Module 17 isn't about CNNs specifically—it's about systematic, principled optimization under constraints. Whether optimizing CNN architectures or black-box functions, success comes from thoughtful strategy, disciplined experimentation, and honest evaluation.

---

**Word Count:** [Your count - keep under 700 words]

**Next Steps:**
1. Submit Round 6 queries to portal
2. Record results in query_history.json
3. Prepare for Round 7 with refined strategy
4. Engage with peers' reflections on discussion board
