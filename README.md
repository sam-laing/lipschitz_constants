# Lipschitz tracking

Consider an MLP whose hidden linear layers are parameterized by matrices
{ W₁ᵏ, W₂ᵏ, …, W_pᵏ } at each time step k ∈ ℕ.
For simplicity, suppose our network is 
$$
f(x) := W_p \sigma(W_{p-1}  \sigma(W_{p-2}(... \sigma(W_1 x))+ b_{p-2})+b_{p-1}) + b_p
$$
The central premise of Muon and its comtemporaries, along with Mirror Descent, is the introduction of a bespoke norm in the proximal GD definition:

We train with GD, SignGD, and Muon, and aim to track inferred Lipschitz constants with respect to various norms.

We denote the vectorized weights at step k by
$$
\mathbf{W}^k := \operatorname{vec}\!\big(W_1^k, \ldots, W_p^k\big).
$$

We are interested in both:
- the full-weight Lipschitz constants (on the vectorized parameter space), and
- the layer-wise Lipschitz constants.


We design an experiment to validate the assertion that different norms matter by tracking multiple Lipschitz constants and examining how they relate to the optimizer. In particular, Muon is motivated by equipping the spectral norm on the weight space.

The true global Lipschitz constant is intractable to estimate even for small problems. Instead, we consider the Lipschitz constant along the trajectory of the weight updates.

Definition (Dual norm). Given a norm ρ: V → ℝ_{\ge 0} on an inner product space (V, ⟨·,·⟩), the dual norm ρ* is
$$
\rho^*(\varphi) := \sup\{ \langle \varphi, \psi \rangle \;\mid\; \psi \in V,\ \rho(\psi) \le 1 \}.
$$

More explicitly, we track, for each layer i ∈ {1, …, p}, iteration k ∈ ℕ, and norm ρ:
$$
\ell_{\rho}^{\,k,i}
\;:=\;
\frac{\rho^*\!\left(\nabla_{W_i} L\!\left(W_{i}^{k+1} ; X^k, y^k\right)
- \nabla_{W_i} L\!\left(W_{i}^{k} ; X^k, y^k\right)\right)}
{\rho\!\left( W_i^{k+1} - W_i^{k} \right)}.
$$
On a pratical level, this means 

We also compute the same quantity over the vectorized full weights, replacing \(W_i^k\) with \(\mathbf{W}^k\).

Some of the norms we track for ρ include:
- ℓ₁ (“1”)
- ℓ₂ (“2”)
- ℓ∞ (“inf”)
- Spectral/operator norm (“spec”)
