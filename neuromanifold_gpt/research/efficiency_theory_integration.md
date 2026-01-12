# Theoretical Efficiency Extensions

## 1. Knot-Theoretic Gating (Sparse Topology)
**Goal:** Reduce Attention Complexity from $O(N^2)$ to $O(N \cdot K)$ where $K$ is "entangled" neighbors.

**Concept:**
Instead of attending to all tokens or just local ones, attend only to tokens whose manifold trajectories are *topologically linked*.

**Metric: Discrete Linking Number**
For two curves $\gamma_1, \gamma_2$ in $\mathbb{R}^3$ (projected from manifold):
$$ \text{Link}(\gamma_1, \gamma_2) = \frac{1}{4\pi} \oint_{\gamma_1} \oint_{\gamma_2} \frac{\mathbf{r}_1 - \mathbf{r}_2}{|\mathbf{r}_1 - \mathbf{r}_2|^3} \cdot (d\mathbf{r}_1 \times d\mathbf{r}_2) $$

**Approximation for Tokens:**
Tokens form discrete points. We can approximate "entanglement" by checking if the convex hulls of their k-nearest neighbor graphs interlock.

**Implementation Strategy:**
1. Project SDRs to 3D subspace via PCA/Manifold.
2. Compute fast approximate linking number (or crossing number).
3. **Hard Gate:** If Link < $\epsilon$, Attention_{ij} = 0.
4. **Result:** Physically sparse attention matrix based on semantic entanglement.

## 2. Number Partitioning for ODE Stability
**Goal:** Speed up FHN solver by increasing timestep $\Delta t$.

**Problem:**
FHN dynamics become unstable if input current $I_{ext}$ is too large (stiff equation). This forces small $\Delta t$.
$I_{ext}$ is the sum of spectral inputs. Random sums have high variance (Central Limit Theorem).

**Solution: Balanced Partitioning**
Use **Karmarkar-Karp** differencing heuristic to split inputs $x_1, ..., x_n$ into $M$ groups such that sums $S_1 \approx ... \approx S_M$.

**Algorithm:**
1. Sort inputs by magnitude.
2. Place largest input in smallest-sum bin.
3. Repeat.

**Benefit:**
- Inputs to each FHN head are bounded and balanced.
- We can use a **coarse solver** (larger $\Delta t$) because we guaranteed no "shocks" (large $I_{ext}$).
- Parallelize the $M$ groups.

**Status:**
- [ ] Research efficient GPU implementation of Karmarkar-Karp (or greedy approximation).
- [ ] Apply to `SpectralDecomposition` output before `FHNAttention`.
