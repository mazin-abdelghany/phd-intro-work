# paul/

This directory contains PDWK's Bayesian optimisation (BO) experiments for
delta-minimax group sequential trial design, developed in collaboration with
Mazin Abdelghany (April 2026).

The goal is to replace the simulated annealing in James Wason's C++ code with
BO, finding delta-minimax designs more efficiently (fewer objective evaluations).

---

## Key methodological ideas

### 1. Reverse parameterisation

Rather than optimising over the original parameter space
`(n, l_1, u_1, ..., l_K, u_K)`, we reparameterise as:

```
θ̃ = (c, Δu_K, Δl_K, ..., Δu_2, Δl_2)
```

where `c = l_K = u_K` is the meeting point and the Δs are non-negative
increments. This guarantees monotonicity and the meeting-point constraint by
construction — no rejection sampling, no monotonicity penalty, no failure region
for structural constraints. See `handlingMonotonicityConstraints.tex` (in the
main repo) for the full write-up.

### 2. Integer n by bisection

Rather than including `n` as a BO parameter, for each proposed boundary set we
find the smallest integer `n` achieving `β' ≤ β*` via bisection over `{2,...,200}`.
This removes `n` from the BO space (5-dim instead of 6-dim) and guarantees `β'`
by construction. Only `α'` needs handling.

### 3. Local search

The BO search box is centred on the deflated triangular design (α* × 49/50,
following James's C++) with half-widths of ±0.3 on each parameter. This keeps
proposals in a region where ~92% are feasible.

### 4. Smooth α' penalty

Objective: `max_δ E[N|δ]/μ + μ·((α'-α*)/α*)² · 1{α'>α*}`, clipped to 2.0.
Since β' is guaranteed by bisection, only this single penalty term is needed.

---

## Scripts

### `simple_local_bo.py`

The main result. Single GPR (Matern52, ARD) trained on all evaluations with the
smooth α' penalty, local search box, integer n by bisection. Includes:

- Warm start option: set `warm_start_params` to the best design from a previous
  run to re-centre the search box
- Random search baseline with trajectory comparison
- Running minimum plot overlaying BO vs random search
- Boundary comparison plot (BO best vs triangular vs warm start)

Run as: `python simple_local_bo.py`

### `no_penalty_bo.py`

Same as `simple_local_bo.py` but with no α' penalty — raw ESS/μ objective
regardless of feasibility. Includes a post-hoc assessment cell that checks what
fraction of the dataset is truly feasible and whether the GPR was misled.
Result: GPR chased infeasible low-ESS designs; worse than random search.

Run as: `python no_penalty_bo.py`

### `feasible_only_bo.py`

GPR trained only on feasible points (infeasible proposals discarded), no VGP,
no penalty. Result: only ~10% of BO proposals were feasible so the GPR was
almost never updated; did not work.

Run as: `python feasible_only_bo.py`

### `gp_surface_bo.py`

Same BO as `no_penalty_bo.py` but saves GP posterior mean and std plots to
`plots/` every 25 iterations. Shows three parameter pairs:
- c vs Δl₃
- c vs Δu₂
- Δu₂ vs Δl₂

Other parameters fixed at the current best point. Useful for understanding what
the GPR is learning and where it is uncertain.

Run as: `python gp_surface_bo.py`
Plots saved to: `plots/checkpoint_NNNN.png`

---

## Results

### `simple_local_bo_results.csv`

530 rows from the best BO run. Columns:
`c, delta_u3, delta_l3, delta_u2, delta_l2, obj_f, feasible`

Best design found:
- Objective: **0.2784** (vs deflated triangular benchmark 0.2824)
- Upper boundaries: [1.9607, 1.9607, 1.7964]
- Lower boundaries: [0.4492, 1.3624, 1.7964]
- n per arm per stage: 25
- α' = 0.0493 (target ≤ 0.05) ✓
- Power = 0.9001 (target ≥ 0.9) ✓

### `no_penalty_bo_results.csv`

530 rows from the no-penalty run. Best overall objective was 0.1987 but
infeasible (α' = 0.1137). Best feasible was 0.2835, worse than random search
(0.2788).

### `plots/`

21 PNG checkpoint plots from `gp_surface_bo.py`, saved every 25 iterations
(iterations 0, 25, 50, ..., 500). Each shows GP posterior mean and std for
three parameter pairs.
