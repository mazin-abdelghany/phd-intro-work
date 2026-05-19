Mathematically, this code solves a **sequential integration problem** to find the boundaries of a **non-homogeneous Markov chain**.

In a group sequential clinical trial, the test statistics at each interim analysis $k$ (where $k = 1, 2, \dots, N$) form a multivariate normal distribution with an **independent increments structure**. Instead of solving an incredibly expensive $N$-dimensional multivariate normal integral, this code uses the **Markov property** to break the problem down into a sequence of simple 1D integrals propagated across a numerical grid.

Here is the exact mathematical formulation of what each section of your code is doing.

---

### 1. The Underlying Stochastic Process

Let $Z_k$ be the standard normal test statistic observed at interim analysis $k$. Under a given hypothesis (Null or Alternative), $Z_k$ has a mean (or drift) $\mu_k$.

Because the trial accumulates data over time, the relationship between the score at the previous look ($Z_{k-1}$) and the current look ($Z_k$) can be written as a conditional linear equation:

$$Z_k \mid (Z_{k-1} = z_{k-1}) \sim \mathcal{N}\left(\mu_k + \rho_k(z_{k-1} - \mu_{k-1}), \, 1 - \rho_k^2\right)$$

Where:

* **$\rho_k$ (Correlation):** $\rho_k = \sqrt{\frac{I_{k-1}}{I_k}}$, the ratio of cumulative information (sample sizes).
* **$\sigma_{k|k-1}$ (`cond_scale`):** $\sqrt{1 - \rho_k^2}$, the conditional standard deviation.
* **$\mathbb{E}[Z_k \mid Z_{k-1}=z_{k-1}]$ (`cond_means`):** $\mu_k + \rho_k(z_{k-1} - \mu_{k-1})$, the expected location of $Z_k$ given it was at $z_{k-1}$.

---

### 2. Analysis 1: The Initial Probability Density

At the first look, the probability density function (PDF) of $Z_1$ is simply a standard normal curve shifted by its mean $\mu_1$:

$$f_1(z) = \phi(z - \mu_1) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}(z - \mu_1)^2}$$

The trial stops early at Analysis 1 if $Z_1$ crosses the stopping boundaries $(l_1, u_1)$:

* **Futility Probability:** $P(\text{Fut}_1) = \int_{-\infty}^{l_1} f_1(z)dz = \Phi(l_1 - \mu_1)$  *(Computed via `numba_norm_cdf`)*
* **Efficacy Probability:** $P(\text{Eff}_1) = \int_{u_1}^{\infty} f_1(z)dz = 1 - \Phi(u_1 - \mu_1)$

If the trial continues, the sub-distribution of paths that survive is restricted to the **continuation region** $z \in (l_1, u_1)$. The code samples this continuous function $f_1(z)$ at $G$ discrete points (`grid_points`) to create the vector $\mathbf{d}_1$.

---

### 3. Analyses 2 to $N$: Recursive Transition

For any subsequent analysis $k$, the joint probability density function $f_k(z)$ depends entirely on the density that survived the previous stage, $f_{k-1}(y)$.

By the law of total probability and the Markov property, the recursive density profile is defined by the Chapman-Kolmogorov-style integral:

$$f_k(z) = \int_{l_{k-1}}^{u_{k-1}} f_{k-1}(y) \cdot g(z \mid y) \, dy$$

Where $g(z \mid y)$ is the **transition density matrix** mapping a state $y$ at stage $k-1$ to a state $z$ at stage $k$:

$$g(z \mid y) = \frac{1}{\sqrt{1-\rho_k^2}} \phi\left( \frac{z - \left[\mu_k + \rho_k(y - \mu_{k-1})\right]}{\sqrt{1-\rho_k^2}} \right)$$

#### Discretizing the Transition (Matrix-Vector Multiplication)

To compute this on a computer, the code turn this continuous integral into a discrete matrix operation over a grid of size $G \times G$:

$$\mathbf{d}_k = \mathbf{T}_k \mathbf{d}_{k-1} \cdot \Delta y$$

Where $\mathbf{T}_k$ is a $G \times G$ matrix (`transition_pdf`) where every element $(r, c)$ calculates the probability of leaping from the old grid coordinate $y_c$ to the new grid coordinate $z_r$.

---

### 4. Computing Stopping Probabilities via Numerical Quadrature

The stopping probabilities at stage $k$ are computed by integrating the incoming density vector against the conditional probability of escaping the boundaries at this look:

$$P(\text{Fut}_k) = \int_{l_{k-1}}^{u_{k-1}} f_{k-1}(y) \cdot \Phi\left( \frac{l_k - \mathbb{E}[Z_k \mid y]}{\sqrt{1-\rho_k^2}} \right) dy$$

$$P(\text{Eff}_k) = \int_{l_{k-1}}^{u_{k-1}} f_{k-1}(y) \cdot \left[ 1 - \Phi\left( \frac{u_k - \mathbb{E}[Z_k \mid y]}{\sqrt{1-\rho_k^2}} \right) \right] dy$$

The code evaluates these 1D integrals using the **Composite Trapezoidal Rule** (`fut_arg` and `eff_arg`). For a grid vector $\mathbf{x}$ evaluated over a step size $\Delta z$:

$$\int_{a}^{b} x(z) dz \approx \Delta z \left( \sum_{i=1}^{G} x_i - \frac{x_1 + x_G}{2} \right)$$

This is exactly what the code is doing here:

```python
probs[k] = (np.sum(arg) - 0.5 * (arg[0] + arg[-1])) * dz

```

---

### 5. Final Output Statistics

Once the loop calculates the individual probability arrays across all $N$ stages, the high-level wrapper aggregates them linearly:

* **Type I Error ($\alpha$):** The total probability of crossing an efficacy bound at any stage under the Null Hypothesis ($\mu = \mu_0$):

$$\alpha = \sum_{k=1}^{N} P(\text{Eff}_k \mid \mu_0)$$


* **Statistical Power:** The total probability of crossing an efficacy bound at any stage under the Alternative Hypothesis ($\mu = \mu_1$):

$$\text{Power} = \sum_{k=1}^{N} P(\text{Eff}_k \mid \mu_1)$$


* **Expected Sample Size (ASN):** The average number of patients enrolled before the trial halts, calculated as the expected value of a discrete stopping variable:

$$\mathbb{E}[I] = \sum_{k=1}^{N} \left[ P(\text{Fut}_k \mid \mu_1) + P(\text{Eff}_k \mid \mu_1) \right] \cdot I_k$$