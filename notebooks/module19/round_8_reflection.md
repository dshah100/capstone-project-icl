# Reflection on LLM-Centred BBO Strategy (Round 8, 17 Data Points)

*Keep under 700 words for the discussion board.*

---

## 1. Prompt Patterns

I used a hybrid approach: a GP surrogate for query generation, with structured prompts when consulting an LLM for strategy decisions (e.g., whether to favour EI vs UCB, or how aggressively to exploit). For those auxiliary tasks, I found **structured prompts** (explicit sections: context, task, output format) reduced variance compared to ad-hoc questions. A simplified prompt like "What should I do next?" produced inconsistent advice; a structured prompt with "Given 17 observations across 8 functions, recommend acquisition strategy and rationale" yielded more coherent, actionable outputs. The structure acted like few-shot scaffolding—guiding the model to reason stepwise rather than leap to conclusions.

---

## 2. Decoding Settings

I treated GP hyperparameters as analogues to LLM decoding: **xi** (EI exploration) and **kappa** (UCB) play roles similar to **temperature**—lower values favour exploitation (coherent, near-best regions), higher values favour exploration (diverse, uncertain regions). For Round 8 I used xi=0.007 and kappa=1.7, equivalent to low temperature: more deterministic, exploitation-heavy. When I experimented with LLM-assisted strategy, I used **temperature ≈ 0.3–0.5** and **top-p ≈ 0.9** to balance coherence with slight diversity in recommendations. **Max-tokens** was set to 500 to avoid runaway outputs. These choices pushed queries toward refinement rather than random exploration—appropriate given 17 data points and limited remaining budget.

---

## 3. Tokenization Effects

I did not observe token boundaries or truncation directly affecting query generation, because the GP outputs fixed-format strings (e.g., `0.822682-0.538761`). To check for tokenisation artefacts, I compared how an LLM handled edge cases: `0.000001` vs `0.0`, and `0.999999` vs `1.0`. Some models tokenised these differently; I standardised to six decimal places to minimise ambiguity. With 17 data points, the context length stayed well under typical limits, so truncation was not an issue. I also verified that boundary values (0.0, 1.0) in proposed queries did not trigger unexpected behaviour in downstream parsing.

---

## 4. Limitations with 17 Data Points

With 17 data points, several limitations emerged. **Prompt overfitting**: When prompting an LLM with full history, it sometimes anchored on early, salient points and underweighted recent ones. **Attention dilution**: As context grew, the model’s focus spread; the most informative observations did not always receive proportional weight. **Diminishing returns**: Adding more points to the prompt did not always improve recommendations—beyond a certain length, marginal value dropped. The GP surrogate, by contrast, treats all observations equally through the kernel; its limitations were more about numerical instability (e.g., length-scale bounds) than context length.

---

## 5. Hallucination Reduction Strategies

To reduce hallucinations, I used: (1) **Tighter instructions**—explicitly stating "base recommendations only on the provided data" and "do not invent values." (2) **Output format constraints**—requiring responses in a fixed structure (e.g., "Strategy: X. Rationale: Y.") to reduce free-form fabrication. (3) **Retrieval of prior information**—when consulting an LLM, I included only the most relevant observations (e.g., best-so-far and recent queries) rather than the full 17-point dump. For the GP pipeline, the surrogate is inherently grounded in observed data; the main risk was numerical artefacts (e.g., optimiser proposing values outside [0,1]), which I mitigated with explicit bounds.

---

## 6. Scaling Strategies

For larger data sets or more complex LLMs, I would: (1) **Summarise** long histories instead of passing raw points—e.g., "Best region: X. Recent trend: Y." (2) **Chunk** by function or round, and aggregate insights. (3) **Retrieve** top-k most informative points (e.g., by uncertainty or improvement) rather than all. (4) **Adjust decoding**—lower temperature as data grows to favour exploitation; consider chain-of-thought for complex reasoning. (5) **Constrain output**—JSON or structured formats to reduce parsing errors at scale.

---

## 7. Practitioner's Balance

These design choices reinforced a practitioner mindset: **exploration vs exploitation** (temperature/xi), **risk** (hallucination reduction, format constraints), and **computational constraints** (token limits, summarisation). In a black-box setting with incomplete information, I learned to treat prompts and decoding as levers—not guarantees. Structured prompts reduced uncertainty but did not eliminate it; I still maintained residual exploration because attention is finite and tokenisation can distort edge cases. The balance is iterative: tighten when confident, loosen when uncertain, and always validate outputs against observed data.
