# Reflection: Clustering Lens on BBO Optimisation (Round 11, 20 Data Points)

---

## 1. How have patterns in your past queries influenced your latest choices?

Round 10 delivered breakthroughs on four functions: F3 (-0.129, up from -0.083), F4 (-39.9, nearly doubling the previous best), F7 (2.7e-06, 158x improvement), and F8 (4.28, down from 7.23). All came from boundary-heavy queries at corners of [0,1]^n. This shaped round 11: I applied K-Means (k=3) to each function's 10 queries, identified the cluster with lowest mean output, and biased 60% of GP candidate sampling toward that cluster's centroid. For F4, the winning cluster (R2, R9, R10) -- all corner-dominant -- had mean -26.2, far better than the central cluster's -4.2, leading the GP to propose the pure corner [1, 0, 1, 0, 1]. Conversely, for F5, boundaries were catastrophic (R10 scored 372.7), so the cluster analysis correctly identified R2/R9 (mean 51.7) as the best starting region.

## 2. Have you identified any clusters or recurring regions?

Yes, clear clusters emerge across most functions. **F3** has a "low-x1/x2, high-x4" corner cluster (R1, R9, R10) with mean output -0.081, outperforming the central cluster (R3-R7, mean -0.032) by 2.5x. **F4** shows a "corner" cluster (R2, R9, R10) at mean -26.2 versus a "centre" cluster (R3-R8) at mean -4.2. **F7** has a tight "boundary" cluster (R2, R6, R7, R8) with mean 0.002, all sharing the pattern of low x2/x3 and high x5. **F6** is distinctive: its best point (R2, -2.54) sits alone as a singleton cluster, far from the central cluster (R4-R8, mean -1.0) -- no other query has approached R2's region of high x2 (~0.99), high x5/x6, and low x4/x7. For F2, the best cluster groups R1, R3, and R8 (mean -0.018), characterised by low x1 and high x3. These groupings are not arbitrary; they reflect genuine basins of attraction in the underlying functions.

## 3. Which strategies have proven less effective?

Pure boundary exploration fails for certain functions. F5's round 10 query ([0.31, 0.49, 1.0, 0.60, 0.85, 0.66]) scored 372.7, the worst result in its history. Similarly, F2's round 10 extreme corner ([1.0, 0.0, 0.71]) scored +0.07, regressing from -0.063. The lesson: boundary strategies work spectacularly for functions whose optima genuinely lie at corners (F4, F7, F8) but catastrophically for those whose optima lie in interior regions (F5). For round 11, I adjusted by using the cluster analysis as a filter: the GP still proposes candidates, but 60% are drawn from within the best cluster's boundary rather than the full [0,1]^n space. This prevents the GP from being seduced by high-uncertainty boundary regions for functions where boundaries have proven poor.

## 4. How do your refinements parallel clustering algorithms?

The parallel is direct. Just as K-Means separates data into groups by minimising within-cluster variance, my strategy separates query history into "good" and "poor" basins and allocates future queries disproportionately to the good basins. The "noise" in a clustering context corresponds to exploratory queries that landed in poor regions (e.g., F5's R8 and R10); these are assigned to outlier clusters and effectively deprioritised. The "signal" is the tight grouping of good-performing queries -- for F7, rounds 2/6/7/8 form a compact cluster (avg intra-cluster distance 0.38) with consistently low outputs (~0.002), analogous to a dense cluster with high cohesion and clear separation from the diffuse central cluster (distance 0.69). My boundary-tightening strategy mirrors how iterative K-Means refines centroids: each round's best-cluster centroid shifts toward the true optimum as new data narrows the region.

## 5. What trends might appear if results were plotted?

Plotting query coordinates against outputs would reveal clear spatial structure. For F4, a PCA projection would show two separated groups: corner points (outputs near -20 to -40) and central points (near -3 to -6). For F7, the best points form a tight boundary cluster in the low-x2/x3 region. For F8, rounds 8 and 10 appear as outliers far from the central mass but with the best outputs, suggesting the optimum lies in a sparsely explored boundary region. These visualisations would inform round 12 by revealing whether the best cluster is tightening (convergence) or whether new outliers keep outperforming (undiscovered basins).
