# Circuit Archaeology: A Deep Dive into Transformer Circuits

**Repository Name:** `circuit-archaeology`  
**Full Title:** *Circuit Archaeology: Dissecting Transformer Attention Mechanisms*  
**Scope:** 8 research-grade notebooks consolidating 17 core topics into deep, experiment-rich investigations.

---

## Repository Structure

```
circuit-archaeology/
├── README.md                          # Project manifesto, setup, results summary
├── requirements.txt                   # PyTorch, TransformerLens, einops, plotly, etc.
├── setup.py                           # Installable package for utilities
├── .gitignore
├── Makefile                           # `make test`, `make figures`, `make blog`
├── notebooks/
│   ├── 01_Residual_Stream_Geometry.ipynb
│   ├── 02_QK_Circuits_and_Bilinear_Attention.ipynb
│   ├── 03_OV_Circuits_and_Direct_Attribution.ipynb
│   ├── 04_Path_Expansion_and_Composition.ipynb
│   ├── 05_SVD_Composition_Geometry.ipynb
│   ├── 06_Skip_Trigrams_and_Induction_Heads.ipynb
│   ├── 07_Circuit_Discovery_and_Systematic_Ablations.ipynb
│   └── 08_Extensions_Open_Questions_and_Synthesis.ipynb
├── src/
│   ├── __init__.py
│   ├── circuit_utils.py               # Path expansion, circuit extraction
│   ├── ablation.py                    # Zero, mean, resample, path ablations
│   ├── composition.py                 # Q/K/V composition matrices, SVD helpers
│   ├── induction.py                   # Skip-trigram detection, induction scoring
│   ├── visualization.py               # Attention heatmaps, circuit graphs, logit lens
│   └── residual_analysis.py           # Subspace decomposition, norm tracking
├── tests/
│   ├── test_ablation.py               # Sanity checks: ablation sum, identity paths
│   ├── test_composition.py            # Matrix shape checks, orthogonality tests
│   └── test_circuits.py               # Reproduce known small-model results
├── figures/
│   ├── nb01/                          # Auto-saved high-res PNGs/SVGs per notebook
│   ├── nb02/
│   └── ...
├── blog/
│   ├── 01_excavating_the_residual_stream.md
│   ├── 02_the_bilinear_heart.md
│   ├── 03_mapping_value_pathways.md
│   └── 04_beyond_the_paper.md
├── reports/
│   └── findings_summary.md            # Living document of novel observations
└── data/
    └── .gitkeep
```

---

## Notebook 1: Residual Stream Geometry and Subspace Cartography

### Learning Objective
Develop a rigorous, geometric understanding of the residual stream as a high-dimensional computational workspace. Establish that the residual stream is not an undifferentiated blob but a structured object with layer-dependent subspace specialization, norm dynamics, and vocabulary-projectable structure.

### Theory Section
- **Residual stream recurrence:** $x^{(l)} = x^{(l-1)} + f^{(l)}(x^{(l-1)})$ where $f^{(l)}$ is the $l$-th layer (attention + MLP). Derive that the final output is a sum of all layer contributions.
- **Subspace decomposition:** Treat the residual stream at each layer as a matrix $X^{(l)} \in \mathbb{R}^{T \times d_{model}}$ (token positions × model dimension). Apply SVD: $X^{(l)} = U^{(l)} \Sigma^{(l)} (V^{(l)})^T$.
- **Logit lens mathematical basis:** The unembedding matrix $W_U \in \mathbb{R}^{d_{model} \times |V|}$ projects residual stream vectors onto vocabulary space. For any layer $l$, the "logit lens" is $\text{logits}^{(l)} = x^{(l)} W_U$.
- **Norm growth:** Derive that $\|x^{(l)}\|$ grows with depth due to cumulative additions. Discuss whether this growth is beneficial (signal amplification) or pathological (representation collapse).
- **Orthogonality hypothesis:** Early layers may write to subspaces roughly orthogonal to later layers, suggesting a "pipeline" architecture within the residual stream.

### Experiments
- **Experiment 1.1: Norm Trajectory Analysis**
  - *What gets ablated:* Nothing (observational). Compute $\|x^{(l)}_t\|_2$ for each token position $t$ and layer $l$.
  - *Metrics:* Mean norm, norm variance, max/median ratio across positions.
  - *Hypothesis:* Norms grow monotonically and super-linearly in early layers, then stabilize.
  - *Expected outcome:* Plot shows monotonic increase; layer 6 norm ≈ 1.5× layer 1 norm in a 12-layer model.
  - *Sanity check:* Random initialization should show flat norm profile.

- **Experiment 1.2: Subspace Overlap Between Layers**
  - *What gets ablated:* Subspaces. Compute principal angles between $V^{(l)}$ and $V^{(l+k)}$ for $k \in \{1,2,3\}$.
  - *Metrics:* Cosine of principal angles, subspace overlap score $\|V^{(l)T} V^{(l+k)}\|_F$.
  - *Hypothesis:* Adjacent layers have higher subspace overlap than distant layers.
  - *Expected outcome:* Heatmap shows diagonal dominance; overlap decays with $|l_1 - l_2|$.

- **Experiment 1.3: SVD Variance Compression**
  - *What gets ablated:* Truncate SVD to rank-$k$ reconstruction $\hat{X}^{(l)}_k = U^{(l)}_k \Sigma^{(l)}_k (V^{(l)}_k)^T$.
  - *Metrics:* Relative reconstruction error $\|X - \hat{X}_k\|_F / \|X\|_F$, perplexity when feeding $\hat{X}_k$ to downstream layers.
  - *Hypothesis:* Later layers are more compressible (lower effective dimensionality).
  - *Expected outcome:* Later layers achieve <5% reconstruction error with $k \ll d_{model}$.

- **Experiment 1.4: Logit Lens Sanity Check**
  - *What gets ablated:* Project residual stream to vocabulary at each layer.
  - *Metrics:* Top-1 accuracy, entropy of probability distribution, rank of true token.
  - *Hypothesis:* Logit lens quality improves monotonically with depth.
  - *Expected outcome:* Accuracy increases from ~5% (random) at layer 0 to ~60%+ at final layer.

- **Experiment 1.5: Negative Control — Random Subspaces**
  - *What gets ablated:* Replace $V^{(l)}$ with random orthonormal matrices.
  - *Metrics:* Same as 1.2.
  - *Hypothesis:* Random subspaces show no layer-dependent structure.
  - *Expected outcome:* Flat overlap heatmap; serves as baseline for 1.2.

### Visualizations
- **Figure 1.1:** Multi-line plot of residual stream norm vs. layer, colored by token type (punctuation, content, position).
- **Figure 1.2:** Semilog plot of singular value spectra for layers {0, 3, 6, 9, 11}. Reveals whether later layers have sharper spectral decay.
- **Figure 1.3:** Heatmap of subspace overlap between all layer pairs. Should reveal block-diagonal or diagonal structure.
- **Figure 1.4:** Logit lens heatmap: rows = layers, columns = token positions, color = top-1 token probability.

### Research Questions
1. Do residual stream norms grow monotonically, and is this growth task-dependent or architecture-dependent?
2. Are early-layer subspaces more "general" (high variance across many singular values) than late-layer subspaces (concentrated)?
3. How much of the residual stream's information is contained in the top 10% of singular vectors, and does this fraction increase with depth?
4. Can we identify "bottleneck dimensions" that are critical across all layers?
5. Does the logit lens peak before the final layer, suggesting early "decision" formation?

### Deliverables
- **Table 1.1:** Variance explained by top-$k$ components ($k \in \{1, 10, 50, 100, 256\}$) for each layer.
- **Figure 1.1:** Residual stream norm trajectory.
- **Figure 1.2:** Singular value decay curves.
- **Finding:** Quantified statement on subspace overlap decay rate.

---

## Notebook 2: QK Circuits and Bilinear Attention Mechanisms

### Learning Objective
Master the QK circuit as a bilinear form that governs attention patterns. Understand that $W_Q W_K^T$ is not merely a weight product but a structured, often low-rank operator that encodes positional and semantic preferences.

### Theory Section
- **Bilinear attention derivation:** $\text{Attention}(Q, K) = \text{softmax}\left(\frac{X W_Q W_K^T X^T}{\sqrt{d_{head}}}\right)$. The QK circuit is the bilinear form $B = W_Q W_K^T \in \mathbb{R}^{d_{head} \times d_{head}}$.
- **Rank structure:** $\text{rank}(B) \leq \min(\text{rank}(W_Q), \text{rank}(W_K))$. In practice, effective rank is often much lower than $d_{head}$.
- **Positional vs. semantic decomposition:** The QK circuit can be decomposed into $B = B_{pos} + B_{sem} + B_{noise}$ using Fourier analysis or token-shuffling.
- **Temperature and scale:** The $\sqrt{d_{head}}$ denominator controls the sharpness of the softmax. Derive how scaling affects attention entropy.

### Experiments
- **Experiment 2.1: QK Circuit Rank Analysis**
  - *What gets ablated:* Truncate SVD of $B = W_Q W_K^T$ to rank-$k$.
  - *Metrics:* Effective rank $r_{eff} = \exp(-\sum p_i \log p_i)$ where $p_i = \sigma_i^2 / \sum \sigma_j^2$; spectral gap $\sigma_1 / \sigma_2$.
  - *Hypothesis:* QK circuits have low effective rank ($<< 20\%$ of $d_{head}$) because they implement specific positional or semantic matching.
  - *Expected outcome:* Sharp singular value decay; effective rank $\ll d_{head}$ for most heads.
  - *Ablations:* Reconstruct attention patterns using only top-$k$ singular components; measure correlation with full attention.

- **Experiment 2.2: Bilinear Temperature Scaling**
  - *What gets ablated:* Replace $\sqrt{d_{head}}$ with temperature $T \in \{0.1, 0.5, 1.0, 2.0, 5.0\}$.
  - *Metrics:* Attention entropy $H = -\sum_j a_{ij} \log a_{ij}$; top-1 attention weight; next-token accuracy.
  - *Hypothesis:* Lower $T$ sharpens attention, increasing sparsity but potentially losing multi-token context.
  - *Expected outcome:* Entropy decreases monotonically with $1/T$; accuracy peaks at $T=1$ and degrades at extremes.

- **Experiment 2.3: Positional vs. Semantic QK Decomposition**
  - *What gets ablated:* Compare attention patterns on (a) original sequence, (b) token-shuffled sequence (destroy position but preserve token identity), (c) position-shuffled sequence (destroy token identity but preserve position).
  - *Metrics:* Attention pattern correlation (Frobenius inner product), KL divergence between distributions.
  - *Hypothesis:* Heads cluster into "positional" (high correlation on shuffled tokens) and "semantic" (high correlation on original).
  - *Expected outcome:* 2D scatter plot shows clustering; some heads are pure positional (diagonal), others pure semantic (content-matching).

- **Experiment 2.4: QK Circuit Counterfactual — Orthogonal Replacement**
  - *What gets ablated:* Replace $W_Q$ with a random orthogonal matrix of same shape.
  - *Metrics:* Attention entropy, next-token loss, pattern correlation with original.
  - *Hypothesis:* Random QK destroys structured attention, increasing entropy.
  - *Expected outcome:* Entropy approaches uniform; loss increases significantly.

- **Experiment 2.5: Negative Control — Random Bilinear Forms**
  - *What gets ablated:* Generate random $B_{rand}$ with same singular values as real $B$ but random singular vectors.
  - *Metrics:* Same as 2.1.
  - *Hypothesis:* Random $B$ has full effective rank and no interpretable attention patterns.
  - *Expected outcome:* Flat singular value spectrum; no sharp attention peaks.

### Visualizations
- **Figure 2.1:** Attention heatmaps for 6 representative heads (2 positional, 2 semantic, 2 mixed).
- **Figure 2.2:** Semilog plot of singular values of $B$ for all heads in a layer. Color by head.
- **Figure 2.3:** 2D scatter of heads: x-axis = positional correlation, y-axis = semantic correlation. Reveals head taxonomy.
- **Figure 2.4:** Attention entropy vs. temperature $T$ for multiple heads.

### Research Questions
1. What is the distribution of effective ranks across all heads in a layer? Is it bimodal (low-rank vs. full-rank)?
2. Do QK circuits attend to specific token types (e.g., nouns, verbs, punctuation) or purely positional offsets?
3. How does the spectral gap of $B$ correlate with the "interpretability" of a head's attention pattern?
4. Can we reconstruct attention patterns from only the top 3 singular components of $B$?
5. Are positional QK circuits more common in early layers and semantic QK circuits in late layers?

### Deliverables
- **Figure 2.2:** QK circuit rank analysis (singular value spectra).
- **Figure 2.3:** Head taxonomy scatter plot.
- **Table 2.1:** Effective rank and spectral gap for all heads in layers 0, 5, 10.
- **Finding:** Quantified positional vs. semantic head classification.

---

## Notebook 3: OV Circuits and Direct Logit Attribution

### Learning Objective
Understand the OV circuit as the mechanism that translates attended information into vocabulary predictions. Master direct logit attribution as a rigorous decomposition of the final output into per-head contributions.

### Theory Section
- **OV circuit definition:** $W_{OV} = W_V W_O \in \mathbb{R}^{d_{model} \times d_{model}}$. The OV circuit maps the residual stream to a modified residual stream via the attended values.
- **Direct logit attribution:** For head $h$ at layer $l$, its contribution to the logits for token $t$ is:
  $$\text{logit}_h^{(l)}(t) = \text{Attn}^{(l)}_h \cdot (x^{(l-1)} W_V^{(l,h)} W_O^{(l,h)}) W_U$$
- **Head output decomposition:** The total logits are the sum of all head contributions plus the embedding/unembedding path and MLP paths.
- **Copying vs. anti-copying:** Derive that an OV circuit can implement "copying" (attending to token $A$ increases logit for $A$) or "anti-copying" (increases logit for $B \neq A$).

### Experiments
- **Experiment 3.1: Direct Logit Attribution Reproduction**
  - *What gets ablated:* Compute per-head direct logit attribution for a set of test prompts.
  - *Metrics:* L2 norm of logit contribution vector, top-1 predicted token from head alone, rank of true next token.
  - *Hypothesis:* A small number of heads contribute disproportionately to the correct next-token prediction.
  - *Expected outcome:* Pareto distribution: 20% of heads account for 80% of attribution mass.
  - *Sanity check:* Sum of all head contributions + embedding path ≈ final logits (within 1%).

- **Experiment 3.2: OV Circuit Top-Token Analysis**
  - *What gets ablated:* For each head, compute the top-k tokens most activated by the OV circuit when attending to a specific source token.
  - *Metrics:* Token identity, logit value, semantic category of top tokens.
  - *Hypothesis:* OV circuits specialize (e.g., punctuation heads, noun heads, verb heads).
  - *Expected outcome:* Clustering of heads by semantic function; some heads are "copy heads" (top token = source token).

- **Experiment 3.3: OV Circuit Ablations**
  - *What gets ablated:* Zero out $W_V$, $W_O$, or both for each head individually.
  - *Metrics:* Increase in cross-entropy loss, change in top-1 accuracy, KL divergence from full model.
  - *Hypothesis:* Ablating heads with high direct attribution causes larger loss increases.
  - *Expected outcome:* Strong correlation between direct attribution magnitude and ablation impact.

- **Experiment 3.4: OV Circuit Swap (Counterfactual)**
  - *What gets ablated:* Swap $W_{OV}$ between two heads $h_1$ and $h_2$.
  - *Metrics:* Loss, accuracy, attention pattern (does the swap change QK or just OV?).
  - *Hypothesis:* Swapping OV circuits changes the "meaning" of the head's output but preserves attention patterns.
  - *Expected outcome:* Attention patterns remain similar; logits change dramatically.

- **Experiment 3.5: Negative Control — Random OV**
  - *What gets ablated:* Replace $W_{OV}$ with random Gaussian matrix (scaled appropriately).
  - *Metrics:* Same as 3.3.
  - *Hypothesis:* Random OV circuits add noise rather than structured signal.
  - *Expected outcome:* Loss increases uniformly; no head-specific specialization.

### Visualizations
- **Figure 3.1:** Direct logit attribution bar chart: x-axis = heads, y-axis = L2 norm of contribution. Color by layer.
- **Figure 3.2:** Logit lens heatmap across layers (rows = layers, cols = positions, color = probability of true next token).
- **Figure 3.3:** OV circuit "token projection": For head $h$, show top-10 tokens in vocabulary space when attending to [specific token].
- **Figure 3.4:** Scatter plot: x = direct attribution norm, y = ablation loss increase. Should show correlation.

### Research Questions
1. Which heads have the strongest direct logit attribution, and do they cluster in specific layers?
2. Do OV circuits implement "copying" more often than "anti-copying"? Quantify the ratio.
3. How much of the final logit is explained by the top 5 heads vs. the bottom 20 heads?
4. Can we predict a head's ablation impact from its direct attribution alone?
5. Do MLP layers or attention heads contribute more to the final logit for specific token types?

### Deliverables
- **Figure 3.1:** Direct logit attribution by head.
- **Figure 3.4:** Attribution-ablation correlation.
- **Table 3.1:** Top-5 most important heads by layer, with their primary semantic function.
- **Finding:** Quantified copy vs. anti-copy ratio across the model.

---

## Notebook 4: Path Expansion and Composition Fundamentals

### Learning Objective
Decompose the transformer output into a sum over computational paths, distinguishing direct paths from composed paths. Understand Query, Key, and Value composition as distinct mathematical operations with different geometric properties.

### Theory Section
- **Path expansion:** The final residual stream $x^{(L)}$ can be expanded as:
  $$x^{(L)} = x^{(0)} + \sum_{l} \text{Attn}^{(l)} + \sum_{l} \text{MLP}^{(l)} + \sum_{l_1 < l_2} \text{Attn}^{(l_2)} \circ \text{Attn}^{(l_1)} + \dots$$
  where paths are sequences of operations.
- **Composition types:**
  - **Query composition:** $W_{Q}^{(l_2)}$ receives input from head $h_1$ at layer $l_1 < l_2$. Effective matrix: $W_{O}^{(l_1, h_1)} W_{Q}^{(l_2, h_2)}$.
  - **Key composition:** $W_{K}^{(l_2)}$ receives input from head $h_1$. Effective matrix: $W_{O}^{(l_1, h_1)} W_{K}^{(l_2, h_2)}$.
  - **Value composition:** $W_{V}^{(l_2)}$ receives input from head $h_1$. Effective matrix: $W_{O}^{(l_1, h_1)} W_{V}^{(l_2, h_2)}$.
- **Path strength metric:** Define $S(h_1 \to h_2) = \|W_{O}^{(h_1)} W_{Q/K/V}^{(h_2)}\|_F$ as a measure of composition strength.

### Experiments
- **Experiment 4.1: Path Expansion Decomposition**
  - *What gets ablated:* Compute contributions of all paths up to depth 3 (direct, 1-hop, 2-hop).
  - *Metrics:* L2 norm of path contribution, fraction of total output variance explained.
  - *Hypothesis:* Direct paths dominate, but 1-hop composed paths contribute significantly (>20% of variance).
  - *Expected outcome:* Direct > 1-hop > 2-hop in magnitude; 1-hop still non-negligible.

- **Experiment 4.2: Composition Type Prevalence**
  - *What gets ablated:* Compute $S(h_1 \to h_2)$ for Q, K, and V composition separately.
  - *Metrics:* Mean strength, max strength, fraction of head pairs with strength > threshold.
  - *Hypothesis:* Query composition is most prevalent because queries directly determine "what to look for."
  - *Expected outcome:* Q-composition > V-composition > K-composition in aggregate strength.

- **Experiment 4.3: Composed Path Ablations**
  - *What gets ablated:* For a strong composition path $h_1 \to h_2$, zero out $W_O^{(h_1)}$ and measure impact on $h_2$'s output.
  - *Metrics:* Change in $h_2$'s attention pattern, change in $h_2$'s direct logit attribution.
  - *Hypothesis:* Ablating the donor head disrupts the recipient head's behavior more than ablating an unrelated head.
  - *Expected outcome:* Targeted ablation causes larger disruption than random ablation.

- **Experiment 4.4: Counterfactual — Artificial Composition Boost**
  - *What gets ablated:* Scale the composition matrix $W_O^{(h_1)} W_Q^{(h_2)}$ by $\alpha \in \{0, 0.5, 1.0, 2.0, 5.0\}$.
  - *Metrics:* Recipient head's attention entropy, next-token accuracy.
  - *Hypothesis:* Boosting composition strengthens the recipient's "preference" for the donor's output.
  - *Expected outcome:* Attention becomes more concentrated on positions where the donor was active.

- **Experiment 4.5: Sanity Check — Direct Path Isolation**
  - *What gets ablated:* Mask out all composed paths, keeping only direct paths (no head output feeds into another head's Q/K/V).
  - *Metrics:* Final loss, accuracy.
  - *Hypothesis:* Direct paths alone cannot maintain full model performance.
  - *Expected outcome:* Significant performance drop, confirming that composition is essential.

### Visualizations
- **Figure 4.1:** Stacked bar chart of path contributions: direct, 1-hop Q, 1-hop K, 1-hop V, 2-hop.
- **Figure 4.2:** Three heatmaps (Q, K, V composition) showing $S(h_1, h_2)$ for all head pairs across layers.
- **Figure 4.3:** Network graph of composition: nodes = heads, edges = composition strength > threshold, color = composition type.
- **Figure 4.4:** Line plot: accuracy vs. composition scaling factor $\alpha$.

### Research Questions
1. Which composition type (Q, K, V) is most prevalent, and does this vary by layer depth?
2. Do composition strengths increase with donor-recipient layer distance, or are nearby layers more likely to compose?
3. Can we identify "hub" heads that donate to many recipients, and "sink" heads that receive from many donors?
4. What fraction of the model's performance is explained by paths of depth ≤ 2 vs. depth > 2?
5. Do Q-composition and K-composition tend to co-occur (same donor-recipient pair), or are they independent?

### Deliverables
- **Figure 4.2:** Composition strength heatmaps (Q/K/V).
- **Figure 4.3:** Composition network graph.
- **Table 4.1:** Top 10 strongest composition paths, with type and layer locations.
- **Finding:** Quantified prevalence ranking of Q/K/V composition.

---

## Notebook 5: SVD Analysis of Composition Geometry

### Learning Objective
Apply rigorous linear algebra to understand the geometric structure of composition. Move beyond scalar strength metrics to analyze the subspaces that enable or inhibit composition between heads.

### Theory Section
- **SVD of composition matrices:** For Q-composition $C_Q = W_O^{(h_1)} W_Q^{(h_2)} \in \mathbb{R}^{d_{model} \times d_{head}}$, compute SVD: $C_Q = U \Sigma V^T$.
- **Subspace alignment:** Composition is effective when the output subspace of $h_1$ (spanned by columns of $W_O^{(h_1)}$) aligns with the input subspace of $h_2$'s query (rows of $W_Q^{(h_2)}$). Measure via principal angles.
- **Effective rank and bottleneck dimensions:** Low-rank composition matrices imply that only a few directions in the residual stream are used for cross-head communication.
- **Geometric interpretation:** Composition as a sequence of operations: project residual stream to donor's output subspace → rotate/scale via singular values → project to recipient's input subspace.

### Experiments
- **Experiment 5.1: Composition Matrix Rank Analysis**
  - *What gets ablated:* Compute SVD for all $C_Q, C_K, C_V$ matrices across all layer pairs.
  - *Metrics:* Effective rank, singular value entropy, fraction of variance in top-5 singular values.
  - *Hypothesis:* Composition matrices are extremely low-rank ($r_{eff} < 10$) because heads communicate via narrow "channels."
  - *Expected outcome:* Sharp singular value decay; most composition matrices have $r_{eff} \ll \min(d_{model}, d_{head})$.

- **Experiment 5.2: Subspace Alignment and Composition Strength**
  - *What gets ablated:* Compute principal angles between $\text{Im}(W_O^{(h_1)})$ and $\text{Im}(W_Q^{(h_2)})$.
  - *Metrics:* Minimum principal angle $\theta_{min}$, composition strength $S(h_1, h_2)$.
  - *Hypothesis:* Smaller $\theta_{min}$ (better alignment) correlates with stronger composition.
  - *Expected outcome:* Scatter plot shows negative correlation: $\cos(\theta_{min})$ vs. $S$.

- **Experiment 5.3: Truncated Composition Reconstruction**
  - *What gets ablated:* Reconstruct recipient head behavior using only top-$k$ singular components of $C_Q$.
  - *Metrics:* Cosine similarity between full and truncated recipient output, next-token accuracy.
  - *Hypothesis:* Top 3-5 singular components capture most of the composed behavior.
  - *Expected outcome:* Accuracy plateaus after $k \approx 5$.

- **Experiment 5.4: Layer-wise Composition Geometry**
  - *What gets ablated:* Compare subspace alignment statistics for (early→early), (early→late), (late→early), (late→late) compositions.
  - *Metrics:* Mean alignment, mean rank, mean composition strength.
  - *Hypothesis:* Early→late compositions have better alignment (early features feed into late processing).
  - *Expected outcome:* Early→late shows highest alignment; late→early shows lowest.

- **Experiment 5.5: Negative Control — Random Composition**
  - *What gets ablated:* Generate random composition matrices with same dimensions, compare rank and alignment.
  - *Metrics:* Same as 5.1.
  - *Hypothesis:* Random matrices have full rank and no alignment structure.
  - *Expected outcome:* Flat singular spectra; no correlation with subspace alignment.

### Visualizations
- **Figure 5.1:** Singular value spectra for 20 representative composition matrices (semilog plot).
- **Figure 5.2:** Scatter plot: x = minimum principal angle, y = composition strength. Color by composition type.
- **Figure 5.3:** Heatmap of effective rank for all layer-pair composition matrices.
- **Figure 5.4:** 3D scatter plot: x = donor layer, y = recipient layer, z = mean alignment. Reveals geometric structure.

### Research Questions
1. What is the rank distribution of composition matrices, and does it differ between Q, K, and V?
2. Do heads compose more strongly with nearby layers, and is this due to subspace alignment or other factors?
3. Is there a geometric signature (e.g., specific singular vectors) that identifies "effective" composition pairs?
4. How much of composition strength is explained by subspace alignment vs. singular value magnitudes?
5. Are there "universal" singular vectors that appear across many composition matrices?

### Deliverables
- **Figure 5.1:** Composition SVD spectra.
- **Figure 5.2:** Alignment vs. strength scatter.
- **Table 5.1:** Effective rank statistics (mean, median, std) by composition type and layer pair category.
- **Finding:** Quantified relationship between subspace alignment and composition strength.

---

## Notebook 6: Skip Trigrams and Induction Heads

### Learning Objective
Investigate the canonical example of a transformer circuit: the induction head. Understand how skip trigrams [A][B]...[A] → [B] emerge from the interaction of previous-token heads and induction heads via QK and OV composition.

### Theory Section
- **Skip trigram definition:** In a repeated sequence, the model predicts the token that followed the previous occurrence of the current token. Formally: given sequence $[A][B]\dots[A]$, predict $[B]$.
- **Induction head mechanism:** An induction head attends to the token that previously followed the current token (the "previous token" of the current token's earlier occurrence). This requires:
  1. A **previous token head** (copies token $t-1$ to position $t$).
  2. An **induction head** whose QK circuit matches [current token] with [previous token] and whose OV circuit writes the token that followed the match.
- **Mathematical characterization:** The induction head's QK circuit implements a "query = current token, key = previous token" matching. The OV circuit implements "value = token after key, output = predict that token."

### Experiments
- **Experiment 6.1: Skip Trigram Detection**
  - *What gets ablated:* Generate repeated random token sequences (e.g., $[A][B][C][A][B][?]$). Detect positions where attention peaks at the earlier occurrence of the current token.
  - *Metrics:* Induction score = attention weight at the "correct" previous position; accuracy of predicting the repeated token.
  - *Hypothesis:* Specific heads show high induction score (>0.5) on repeated sequences.
  - *Expected outcome:* 1-2 heads per layer show strong induction; score ≈ 0 on non-repeated sequences.

- **Experiment 6.2: Previous Token Head Dependency**
  - *What gets ablated:* Identify previous token heads (attend strongly to position $t-1$). Ablate them individually and in combination.
  - *Metrics:* Induction score of downstream induction heads, next-token accuracy on repeated sequences.
  - *Hypothesis:* Induction heads depend on previous token heads to provide the "key" information.
  - *Expected outcome:* Ablating previous token heads reduces induction score by >50%.

- **Experiment 6.3: Induction Head Ablations**
  - *What gets ablated:* Zero out each induction head individually.
  - *Metrics:* Accuracy on repeated sequences, perplexity on natural text.
  - *Hypothesis:* Induction heads are critical for in-context learning but less important for standard text.
  - *Expected outcome:* Large accuracy drop on repeated sequences; modest perplexity increase on natural text.

- **Experiment 6.4: Counterfactual — Non-Repeated Sequences**
  - *What gets ablated:* Test induction heads on sequences with no repeated tokens.
  - *Metrics:* Attention entropy, induction score, top-1 prediction.
  - *Hypothesis:* Induction heads show no special behavior without repetition.
  - *Expected outcome:* Attention is uniform or follows standard patterns; no skip-trigram peaks.

- **Experiment 6.5: Generalization to Unseen Pairs**
  - *What gets ablated:* Train on sequences with pairs from a subset of tokens; test on unseen pairs.
  - *Metrics:* Induction score, accuracy.
  - *Hypothesis:* Induction heads generalize to unseen token pairs because they implement an abstract "copy next token" algorithm.
  - *Expected outcome:* High accuracy on unseen pairs, confirming algorithmic rather than memorized behavior.

- **Experiment 6.6: Negative Control — Random Attention**
  - *What gets ablated:* Replace induction head attention with uniform random.
  - *Metrics:* Same as 6.1.
  - *Hypothesis:* Random attention destroys skip-trigram behavior.
  - *Expected outcome:* Induction score drops to baseline.

### Visualizations
- **Figure 6.1:** Attention heatmap for an induction head on a repeated sequence. Should show diagonal-offset peaks (attending to previous occurrence).
- **Figure 6.2:** Induction score bar chart for all heads. Color by layer.
- **Figure 6.3:** Accuracy on repeated sequences before/after ablating previous token heads and induction heads.
- **Figure 6.4:** Sequence diagram showing the skip-trigram mechanism: [A][B]...[A] → attention to first [B] → predict [B].

### Research Questions
1. Are induction heads always paired with previous token heads, or can they operate independently?
2. Does induction strength vary with the distance between repeated tokens (short-range vs. long-range)?
3. Can induction heads generalize to unseen token pairs, confirming algorithmic implementation?
4. How many induction heads does a 12-layer model typically have, and do they form a "circuit cluster"?
5. Do induction heads emerge in all transformer sizes, or only above a certain scale?

### Deliverables
- **Figure 6.1:** Skip trigram attention heatmap.
- **Figure 6.2:** Induction score distribution.
- **Table 6.1:** Identified previous token heads and induction heads by layer and index.
- **Finding:** Quantified generalization performance on unseen token pairs.

---

## Notebook 7: Circuit Discovery and Systematic Ablations

### Learning Objective
Treat the model as a graph and apply principled ablation strategies to discover minimal faithful circuits. Reproduce key figures from the Transformer Circuits paper and validate them with alternative metrics.

### Theory Section
- **Circuit discovery as optimization:** Find the minimal set of components (heads, MLPs) such that the model's behavior on a task is preserved. Formally: $\min_{S \subseteq \text{Components}} |S|$ s.t. $L(f_S) \leq (1+\epsilon) L(f_{full})$.
- **Ablation taxonomy:**
  - **Zero ablation:** Set head output to zero.
  - **Mean ablation:** Replace head output with its mean over a distribution.
  - **Resample ablation:** Replace head output with output on a different input.
- **Faithfulness metrics:** $F(S) = 1 - \frac{L(f_S) - L(f_{full})}{L(f_{random}) - L(f_{full})}$, where $f_{random}$ is a random subset of same size.
- **Key paper figures:** Identify which figures from "A Mathematical Framework for Transformer Circuits" are reproducible with available tools (e.g., attention pattern decomposition, composition strength).

### Experiments
- **Experiment 7.1: Systematic Head Importance Ranking**
  - *What gets ablated:* Zero ablation of each attention head individually (144 ablations for 12 layers × 12 heads).
  - *Metrics:* Increase in cross-entropy loss ($\Delta L$), decrease in accuracy, KL divergence from full model.
  - *Hypothesis:* A small subset of heads is individually critical; most heads are redundant.
  - *Expected outcome:* Pareto distribution: top 10 heads cause 60% of total possible damage.

- **Experiment 7.2: Iterative Circuit Pruning**
  - *What gets ablated:* Start with full model. Iteratively remove the least important head (by $\Delta L$). Stop when performance drops below 90% of full.
  - *Metrics:* Faithfulness $F(S)$, sparsity $|S|/N_{total}$.
  - *Hypothesis:* A sparse circuit (20-30% of heads) can maintain 90% performance.
  - *Expected outcome:* Faithfulness curve shows plateau; minimal circuit size identified.

- **Experiment 7.3: Reproduce Key Paper Figures**
  - *What gets ablated:* Select 2-3 key figures from the paper (e.g., composition strength heatmap, attention pattern decomposition).
  - *Metrics:* Visual similarity, quantitative correlation with paper values.
  - *Hypothesis:* The paper's claims are reproducible in the open-source model.
  - *Expected outcome:* Side-by-side comparison showing strong agreement.

- **Experiment 7.4: Negative Control — Random Circuit Pruning**
  - *What gets ablated:* Remove random heads (same number as in 7.2).
  - *Metrics:* Same as 7.2.
  - *Hypothesis:* Random pruning degrades performance faster than importance-based pruning.
  - *Expected outcome:* Random pruning curve lies below importance-based curve.

- **Experiment 7.5: Alternative Experiment — MLP vs. Attention Trade-off**
  - *What gets ablated:* Prune attention heads while keeping MLPs, and vice versa.
  - *Metrics:* Faithfulness curves for both.
  - *Hypothesis:* Attention heads and MLPs have redundant capabilities; pruning one can be partially compensated by the other.
  - *Expected outcome:* Pruning both simultaneously causes steeper drop than pruning either alone.

### Visualizations
- **Figure 7.1:** Head importance heatmap: rows = layers, columns = heads, color = $\Delta L$.
- **Figure 7.2:** Faithfulness vs. sparsity curve. Two lines: importance-based pruning vs. random pruning.
- **Figure 7.3:** Discovered circuit graph: nodes = retained heads, edges = composition > threshold.
- **Figure 7.4:** Side-by-side reproduction of a key paper figure.

### Research Questions
1. What is the minimal faithful circuit for next-token prediction, and how sparse can it be while maintaining 90% performance?
2. Do attention heads have redundant functionality, or is each head uniquely important for specific tokens?
3. How does the importance ranking vary across different input distributions (code vs. natural language)?
4. Can we discover circuits that are faithful for specific tasks but not globally?
5. Do MLP layers compensate for attention head ablations, suggesting a distributed architecture?

### Deliverables
- **Figure 7.1:** Head importance heatmap.
- **Figure 7.2:** Faithfulness-sparsity curve.
- **Table 7.1:** Minimal circuit components (head list) for 90% and 95% faithfulness.
- **Finding:** Quantified redundancy level and MLP-attention trade-off.

---

## Notebook 8: Extensions, Open Questions, and Synthesis

### Learning Objective
Move beyond reproduction to independent investigation. Identify where the Transformer Circuits framework breaks down, design novel experiments, and document negative results. Synthesize findings into a research agenda.

### Theory Section
- **Limitations of the linear framework:** Real transformers have LayerNorm, nonlinear MLPs, and attention softmax. The bilinear framework is an approximation.
- **Multi-token circuits:** The paper focuses on single-token predictions. Extend to multi-token dependencies (e.g., subject-verb agreement across long distances).
- **Cross-layer composition:** The standard framework assumes composition is direct (layer $l_1$ → $l_2$). Investigate indirect composition via the residual stream.
- **Open questions:** 
  - How do MLPs fit into the circuit framework?
  - What is the role of superposition (polysemanticity) in circuits?
  - Can circuits be dynamically reconfigured based on context?

### Experiments
- **Extension 8.1: Multi-Token Induction**
  - *What gets ablated:* Test whether induction heads can handle multi-token patterns (e.g., [A][B][C]...[A][B] → [C]).
  - *Metrics:* Accuracy, attention pattern analysis.
  - *Hypothesis:* Standard induction heads fail on multi-token patterns; deeper models may have "meta-induction" heads.
  - *Expected outcome:* Accuracy drops sharply for 3+ token patterns; potential discovery of longer-range heads.

- **Extension 8.2: Failure Case Analysis — Where Circuits Break**
  - *What gets ablated:* Identify inputs where the circuit model makes confident but wrong predictions. Analyze which heads are active.
  - *Metrics:* Confidence, error rate, head attribution on failure cases.
  - *Hypothesis:* Circuit framework fails when multiple circuits interfere or when the input is out-of-distribution.
  - *Expected outcome:* Specific failure modes (e.g., ambiguous pronoun references) where head attributions conflict.

- **Extension 8.3: Context-Dependent Circuit Switching**
  - *What gets ablated:* Compare circuit activation on two different syntactic structures (e.g., active vs. passive voice).
  - *Metrics:* Head activation patterns, circuit overlap between contexts.
  - *Hypothesis:* Different contexts recruit different sub-circuits.
  - *Expected outcome:* Low overlap between head activation sets across contexts; evidence for dynamic routing.

- **Extension 8.4: Superposition and Circuit Overlap**
  - *What gets ablated:* Analyze whether the same residual stream dimensions participate in multiple circuits.
  - *Metrics:* Subspace overlap between different task-specific circuits.
  - *Hypothesis:* Superposition allows dimension reuse, causing circuit overlap.
  - *Expected outcome:* Significant overlap (>30% shared dimensions) between seemingly unrelated circuits.

- **Extension 8.5: Novel Observation Hunt**
  - *What gets ablated:* Free-form exploration. Test 3-5 hypotheses not in the paper.
  - *Metrics:* Various, depending on hypothesis.
  - *Hypothesis:* Multiple (e.g., "Do heads have circadian-like activation patterns across layers?").
  - *Expected outcome:* At least 2-3 genuinely novel observations, even if they are negative results.

### Visualizations
- **Figure 8.1:** Multi-token induction accuracy vs. pattern length.
- **Figure 8.2:** Failure case analysis: attention patterns on incorrect predictions.
- **Figure 8.3:** Circuit overlap matrix: rows = tasks, columns = tasks, color = subspace overlap.
- **Figure 8.4:** "Surprise" observation: a novel pattern or unexpected result.

### Research Questions
1. Where does the bilinear attention framework break down, and what nonlinear effects are most significant?
2. Are there "meta-circuits" — heads that control which other heads are active?
3. How much do task-specific circuits overlap, and does this explain superposition?
4. Can induction generalize to arbitrary pattern lengths, or is there a hard limit?
5. What are the most important missing pieces in the current circuit framework?

### Deliverables
- **Figure 8.1:** Multi-token induction results.
- **Figure 8.3:** Circuit overlap matrix.
- **Table 8.1:** Open questions ranked by tractability (1-10) and potential impact (1-10).
- **Finding:** At least 3 novel observations with statistical validation.

---

## Aggressive Weekend Timeline

**Total Available Time:** ~20 hours (Sat 13:00–Sun 22:00, minus sleep/meals).

| Time Block | Notebook | Activities | Est. Duration |
|---|---|---|---|
| **Sat 13:00–15:30** | NB1 | Setup repo, ARENA recap, residual stream geometry, norm analysis | 2.5h |
| **Sat 15:30–18:00** | NB2 | QK circuit theory, bilinear attention, rank analysis | 2.5h |
| **Sat 19:00–21:30** | NB3 | OV circuits, direct logit attribution, ablations | 2.5h |
| **Sat 21:30–23:00** | NB4 | Path expansion theory, composition fundamentals | 1.5h |
| **Sun 09:00–11:30** | NB4–5 | Finish path expansion, SVD composition geometry | 2.5h |
| **Sun 11:30–14:00** | NB6 | Skip trigrams, induction heads, previous token analysis | 2.5h |
| **Sun 14:00–16:30** | NB7 | Circuit discovery, systematic ablations, paper reproductions | 2.5h |
| **Sun 16:30–19:00** | NB8 | Extensions, open questions, novel observations | 2.5h |
| **Sun 19:00–20:30** | Blog | Draft blog posts, extract figures | 1.5h |
| **Sun 20:30–22:00** | Polish | README, tests, repo cleanup, GitHub push | 1.5h |

**Critical Path:** If falling behind, skip NB5 (SVD geometry) and merge its key figures into NB4. If still behind, make NB8 a "mini-notebook" with just 2 extensions.

---

## Blog Series Design

**Series Title:** *Excavating Attention: A Weekend in Transformer Circuits*

### Post 1: Excavating the Residual Stream
- **Core message:** The residual stream is not a black box; it has geometric structure that evolves predictably across layers.
- **Figures:** NB1 Figures 1.1, 1.2, 1.4.
- **Story arc:** Start with the mystery of what happens inside the residual stream → SVD reveals subspace specialization → logit lens shows early decision formation.
- **Key insight:** "By layer 6, the model has already made up its mind about the next token, but it spends the next 6 layers refining the confidence."

### Post 2: The Bilinear Heart of Attention
- **Core message:** Attention patterns are governed by low-rank bilinear forms that mix positional and semantic signals.
- **Figures:** NB2 Figures 2.2, 2.3.
- **Story arc:** Attention seems magical → QK circuit decomposition reveals it's just matrix multiplication → rank analysis shows it's simpler than it looks → head taxonomy emerges.
- **Key insight:** "Most attention heads are operating in a 5-dimensional subspace, not 64."

### Post 3: When Heads Compose
- **Core message:** Transformers are not layers in series; they are graphs where heads talk to each other through composition.
- **Figures:** NB4–5 Figures 4.2, 4.3, 5.2.
- **Story arc:** Individual heads are interesting → but the real magic is in the connections → SVD reveals geometric "tunnels" between heads → some heads are universal donors, others are universal recipients.
- **Key insight:** "Query composition is 3× more common than key composition, suggesting transformers are primarily 'query-driven' systems."

### Post 4: Beyond the Paper
- **Core message:** Reproducing research is the starting point; the real value is in finding what the original authors missed.
- **Figures:** NB6–8 Figures 6.1, 7.1, 8.3.
- **Story arc:** Reproducing induction heads is satisfying → but they break on multi-token patterns → circuit overlap reveals superposition → the framework has gaps, and that's exciting.
- **Key insight:** "The circuit model works beautifully until it doesn't — and the failure points are where the next breakthroughs hide."

---

## Stretch Goal Analysis

### Three Notebooks Most Likely to Generate Genuinely Novel Insights

1. **Notebook 5 (SVD Composition Geometry)**
   - *Why:* The geometric analysis of composition via subspace alignment and SVD is underexplored in the original paper. You are likely to discover that composition matrices have unexpectedly low rank or that specific singular vectors recur across heads — observations that could hint at "universal" communication channels in transformers.

2. **Notebook 8 (Extensions and Open Questions)**
   - *Why:* By definition, this notebook is designed for exploration. The multi-token induction failure analysis and context-dependent circuit switching experiments are not in the original paper. Even negative results (e.g., "circuits do not switch cleanly between contexts") are novel and publishable as observations.

3. **Notebook 7 (Circuit Discovery and Ablations)**
   - *Why:* Iterative pruning to find minimal circuits often reveals unexpected redundancies. You may discover that a much smaller circuit than expected maintains high performance, or that certain "unimportant" heads are actually critical for specific out-of-distribution inputs.

### Three Notebooks Most Likely to Impress Mechanistic Interpretability Researchers

1. **Notebook 7 (Circuit Discovery and Ablations)**
   - *Why:* Systematic ablation studies with faithfulness metrics and paper reproductions demonstrate methodological rigor. Researchers value reproducibility and principled circuit discovery more than flashy visualizations. The faithfulness-sparsity curve is a hallmark of serious MI work.

2. **Notebook 5 (SVD Composition Geometry)**
   - *Why:* Applying heavy linear algebra (SVD, principal angles, effective rank) to composition shows mathematical maturity. The subspace alignment analysis connects mechanistic interpretability to classical numerical linear algebra, which appeals to researchers with formal backgrounds.

3. **Notebook 2 (QK Circuits and Bilinear Attention)**
   - *Why:* The positional vs. semantic decomposition and QK rank analysis are foundational to the field. Doing this rigorously with multiple controls (temperature, orthogonal replacement, random baseline) shows you understand the difference between correlation and mechanism.

### Three Notebooks That Would Teach the Deepest Intuition

1. **Notebook 4 (Path Expansion and Composition)**
   - *Why:* Path expansion is the conceptual bridge between "transformers are black boxes" and "transformers are sums of interpretable paths." Once you internalize that the output is a sum over direct and composed paths, you never look at a transformer layer the same way again.

2. **Notebook 1 (Residual Stream Geometry)**
   - *Why:* The residual stream is the substrate on which everything else operates. Understanding its norm dynamics, subspace structure, and logit lens behavior provides the geometric intuition necessary for all subsequent analysis. Without this, circuits are just matrices.

3. **Notebook 6 (Skip Trigrams and Induction Heads)**
   - *Why:* Induction heads are the "poster child" of mechanistic interpretability because they are a complete, interpretable circuit with a clear algorithmic story. Understanding them deeply — including their dependence on previous token heads and their failure modes — provides the template for how to think about all other circuits.

---

**Final Recommendation:** Start with Notebook 1 to build your tooling and geometric intuition, then jump to Notebook 6 (induction heads) if you need a motivational "win" early on. Save Notebook 5 for when you are mentally fresh (Sunday morning) because the linear algebra is demanding but high-reward. Notebook 8 should be treated as a "bonus" — if you only have time for 7 notebooks, skip the multi-token extension in NB8 and keep the novel observation hunt.
