import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Delta-Minimax Group Sequential Design — Feasible-Only BO
    ## Reverse Parameterisation + Integer $n$ + Local Search + GPR on Feasible Points Only

    A clean middle ground between the constrained BO (GPR on feasible + VGP classifier)
    and the simple penalty BO (GPR on all points with penalty).

    Key ideas:

    1. **Reverse parameterisation** — structural constraints automatic, 5 parameters
    2. **Integer $n$ by bisection** — $\beta' \leq \beta^*$ guaranteed by construction
    3. **Local search** — tight box around triangular design
    4. **GPR trained on feasible points only** — clean objective signal, no penalty distortion
    5. **Standard EI** — no VGP, no PoV, no failure region

    Infeasible proposals (those where $\alpha' > \alpha^*$) are simply discarded —
    they do not update the GPR. The GPR sees only clean objective values and can
    fit the landscape accurately. The cost is that ~90% of proposals are wasted,
    but the GPR is numerically stable and the signal it receives is uncontaminated.
    """)
    return


# ============================================================
# Imports
# ============================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"# Imports")
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    return np, pd, plt


@app.cell
def _():
    import gpflow
    import trieste
    from trieste.space import Box
    from trieste.models.gpflow.models import GaussianProcessRegression
    return Box, GaussianProcessRegression, gpflow, trieste


@app.cell
def _():
    from py_group_sequential_designs import generate_boundaries as bd
    from py_group_sequential_designs import simulate as sim
    from py_group_sequential_designs import sample_size as ss
    return bd, sim, ss


# ============================================================
# Trial design settings
# ============================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"# Trial Design Settings")
    return


@app.cell
def _(ss):
    num_analyses  = 3
    target_alpha  = 0.05
    target_power  = 0.9
    delta1        = 1.0
    sigma2        = 3.0

    mu = ss.sample_size_means(
        ratio=1, variance=sigma2, power=target_power,
        alpha=target_alpha, delta=delta1
    )
    print(f"Single-stage sample size mu = {mu:.2f}")
    return delta1, mu, num_analyses, sigma2, target_alpha, target_power


# ============================================================
# Reverse parameterisation (5-dim, no n)
# ============================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reverse Parameterisation

    BO vector: $(c, \Delta u_3, \Delta l_3, \Delta u_2, \Delta l_2)$
    """)
    return


@app.cell
def _(np):
    def reverse_to_boundaries(params, K):
        params  = np.asarray(params).flatten()
        c       = params[0]
        delta_u = params[1::2][::-1]
        delta_l = params[2::2][::-1]
        upper_bounds = np.array([c + np.sum(delta_u[k:]) for k in range(K)])
        lower_bounds = np.array([c - np.sum(delta_l[k:]) for k in range(K)])
        return upper_bounds, lower_bounds


    def boundaries_to_reverse(upper_bounds, lower_bounds):
        upper_bounds = np.asarray(upper_bounds)
        lower_bounds = np.asarray(lower_bounds)
        K            = len(upper_bounds)
        c            = upper_bounds[-1]
        delta_u      = np.diff(upper_bounds[::-1])
        delta_l      = np.diff(lower_bounds)[::-1]
        increments        = np.empty(2 * (K - 1))
        increments[0::2]  = delta_u
        increments[1::2]  = delta_l
        return np.concatenate([[c], increments])

    return boundaries_to_reverse, reverse_to_boundaries


# ============================================================
# Integer n by bisection
# ============================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Integer $n$ by Bisection

    Finds smallest integer $n$ achieving target power.
    Returns `None` if no $n \leq 200$ suffices.
    """)
    return


@app.cell
def _(delta1, num_analyses, sigma2, sim):
    def find_n_integer(upper_bounds, lower_bounds, target_power,
                       n_min=2, n_max=200):
        _, _, power_max, _ = sim.group_sequential_designs(
            n_analyses=num_analyses, upper_bounds=upper_bounds,
            lower_bounds=lower_bounds, n_patients=n_max,
            null_hypothesis=0, alt_hypothesis=delta1, variance=sigma2
        )
        if power_max < target_power:
            return None

        while n_max - n_min > 1:
            n_mid = (n_min + n_max) // 2
            _, _, power_mid, _ = sim.group_sequential_designs(
                n_analyses=num_analyses, upper_bounds=upper_bounds,
                lower_bounds=lower_bounds, n_patients=n_mid,
                null_hypothesis=0, alt_hypothesis=delta1, variance=sigma2
            )
            if power_mid >= target_power:
                n_max = n_mid
            else:
                n_min = n_mid

        return int(n_max)

    return (find_n_integer,)


# ============================================================
# Objective function — returns (x, y_obj, feasible)
# ============================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Objective Function

    Returns the true objective $\max_\delta E[N \mid \delta] / \mu$ if feasible,
    otherwise returns `None` for the objective (infeasible proposals are discarded).

    $\beta' \leq \beta^*$ guaranteed by bisection.
    Feasibility requires only $\alpha' \leq \alpha^*$.
    """)
    return


@app.cell
def _(delta1, find_n_integer, mu, np, num_analyses, reverse_to_boundaries, sigma2, sim, ss, target_alpha, target_power):
    def objective(params, K):
        params = np.asarray(params).flatten()
        upper_bounds, lower_bounds = reverse_to_boundaries(params, K)

        # find integer n achieving target power
        n = find_n_integer(upper_bounds, lower_bounds, target_power)

        if n is None:
            return np.array([params]), None, False

        # evaluate alpha'
        trial = sim.group_sequential_designs(
            n_analyses=K, upper_bounds=upper_bounds, lower_bounds=lower_bounds,
            n_patients=n, null_hypothesis=0, alt_hypothesis=delta1, variance=sigma2
        )
        alpha_prime = trial[1]

        if alpha_prime > target_alpha:
            return np.array([params]), None, False

        # feasible — compute objective
        max_ess = ss.max_ess(
            n_analyses=K, upper_bounds=upper_bounds, lower_bounds=lower_bounds,
            n_patients=n, null_hypothesis=0, variance=sigma2
        )

        y_obj = max_ess / mu
        return np.array([params]), np.array([[y_obj]]), True

    return (objective,)


# ============================================================
# Reference designs and c0
# ============================================================

@app.cell
def _(bd, boundaries_to_reverse, np, num_analyses, objective, target_alpha):
    _alpha_deflated = target_alpha * 49 / 50

    _tri = bd.calculate_triangular_boundaries(
        n_analyses=num_analyses, alpha=_alpha_deflated, delta=1.0, n_patients=20
    )
    tri_params = boundaries_to_reverse(_tri[0], _tri[1])
    c0         = tri_params[0]

    _, _y, _feas = objective(tri_params, num_analyses)
    tri_obj = float(_y[0, 0]) if _y is not None else float('nan')
    print(f"Deflated triangular objective (integer n): {tri_obj:.4f}, feasible: {_feas}")
    print(f"Triangular params (5-dim): {np.round(tri_params, 4)}")
    print(f"Meeting point c0 = {c0:.4f}")
    return c0, tri_obj, tri_params


@app.cell
def _(bd, boundaries_to_reverse, np, num_analyses, objective, target_alpha):
    # non-deflated triangular with integer n — the correct integer-n benchmark
    _tri_true = bd.calculate_triangular_boundaries(
        n_analyses=num_analyses, alpha=target_alpha, delta=1.0, n_patients=20
    )
    tri_params_true = boundaries_to_reverse(_tri_true[0], _tri_true[1])

    _, _y_true, _feas_true = objective(tri_params_true, num_analyses)
    tri_obj_true = float(_y_true[0, 0]) if _y_true is not None else float('nan')
    print(f"Non-deflated triangular objective (integer n): {tri_obj_true:.4f}, feasible: {_feas_true}")
    return (tri_obj_true,)


# ============================================================
# Search space
# ============================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"# Search Space")
    return


@app.cell
def _(mo):
    use_local_search = mo.ui.switch(value=True, label="Use local search space")
    use_local_search
    return (use_local_search,)


@app.cell
def _(Box, c0, np, tri_params, use_local_search):
    _hw_c     = 0.3
    _hw_delta = 0.3

    if use_local_search.value:
        _lower = np.array([
            c0 - _hw_c,
            max(0.0, tri_params[1] - _hw_delta),
            max(0.0, tri_params[2] - _hw_delta),
            max(0.0, tri_params[3] - _hw_delta),
            max(0.0, tri_params[4] - _hw_delta),
        ])
        _upper = np.array([
            c0 + _hw_c,
            tri_params[1] + _hw_delta,
            tri_params[2] + _hw_delta,
            tri_params[3] + _hw_delta,
            tri_params[4] + _hw_delta,
        ])
        _label = "LOCAL"
    else:
        _lower = np.array([c0 - 3.0, 0.0, 0.0, 0.0, 0.0])
        _upper = np.array([c0 + 3.0, 4.0, 4.0, 4.0, 4.0])
        _label = "GLOBAL"

    search_space = Box(lower=_lower, upper=_upper)
    print(f"Search space ({_label}):")
    print(f"  lower: {np.round(search_space.lower.numpy(), 3)}")
    print(f"  upper: {np.round(search_space.upper.numpy(), 3)}")
    return (search_space,)


# ============================================================
# Initialisation — feasible points only
# ============================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Initialisation

    Only feasible points are used to initialise the GPR.
    Infeasible perturbations are discarded silently.

    - Local: perturbations around triangular design only
    - Global: perturbations around all three reference designs
    - Plus random feasible points from pool within search space
    """)
    return


@app.cell
def _(
    bd, boundaries_to_reverse, c0, np, num_analyses,
    objective, search_space, target_alpha, tri_params, use_local_search,
):
    rng_init = np.random.default_rng(seed=42)
    _alpha_deflated = target_alpha * 49 / 50

    _poc = bd.calculate_pocock_boundaries(
        n_analyses=num_analyses, alpha=_alpha_deflated, n_patients=20
    )
    poc_params = boundaries_to_reverse(_poc[0], _poc[1])

    _obf = bd.calculate_of_boundaries(
        n_analyses=num_analyses, alpha=_alpha_deflated, n_patients=20
    )
    obf_params = boundaries_to_reverse(_obf[0], _obf[1])

    _refs   = [tri_params] if use_local_search.value else [poc_params, obf_params, tri_params]
    _n_pert = 15 if use_local_search.value else 5

    init_x, init_y = [], []
    n_tried_pert = 0
    for _ref in _refs:
        for _ in range(_n_pert):
            _scale = np.clip(np.abs(_ref) * 0.10, 0.01, 0.3)
            _p     = _ref + rng_init.normal(scale=_scale, size=len(_ref))
            _p[1:] = np.clip(_p[1:], 0.0, 4.0)
            _x, _y, _feas = objective(_p, num_analyses)
            n_tried_pert += 1
            if _feas:
                init_x.append(_x)
                init_y.append(_y)

    print(f"Perturbation points: {n_tried_pert} tried, {len(init_x)} feasible kept")

    # random pool — keep only feasible
    n_target = 15
    n_found  = 0
    n_tried  = 0
    _lb = search_space.lower.numpy()
    _ub = search_space.upper.numpy()

    for _p in rng_init.uniform(_lb, _ub, size=(1000, len(_lb))):
        _x, _y, _feas = objective(_p, num_analyses)
        n_tried += 1
        if _feas:
            init_x.append(_x)
            init_y.append(_y)
            n_found += 1
        if n_found >= n_target:
            break

    print(f"Random feasible points: {n_found} found from {n_tried} candidates")
    print(f"Total initial feasible points: {len(init_x)}")
    print(f"Objective values: {[round(float(_y[0,0]), 4) for _y in init_y]}")

    return init_x, init_y


@app.cell
def _(init_x, init_y, np, trieste):
    design_matrix = np.concatenate(init_x)
    output_vals   = np.concatenate(init_y)

    initial_data = trieste.data.Dataset(
        query_points = design_matrix,
        observations = output_vals
    )
    print(f"Initial GPR dataset: {design_matrix.shape[0]} feasible points")
    return design_matrix, initial_data, output_vals


# ============================================================
# GP model — trained on feasible points only
# ============================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GP Model

    Single GPR (Matern52, ARD) trained on feasible designs only.
    No VGP, no failure region, no penalty.
    """)
    return


@app.cell
def _(GaussianProcessRegression, design_matrix, gpflow, output_vals):
    _kernel = gpflow.kernels.Matern52(
        lengthscales=[1.0] * design_matrix.shape[1]
    )
    _gpr = gpflow.models.GPR(
        data       = (design_matrix, output_vals),
        kernel     = _kernel,
        likelihood = gpflow.likelihoods.Gaussian()
    )
    gpflow.utilities.print_summary(_gpr, fmt="notebook")
    bayes_opt_model = GaussianProcessRegression(_gpr)
    return (bayes_opt_model,)


# ============================================================
# Ask-tell initialisation
# ============================================================

@app.cell
def _(bayes_opt_model, initial_data, search_space, trieste):
    ask_tell = trieste.ask_tell_optimization.AskTellOptimizer(
        search_space     = search_space,
        datasets         = initial_data,
        models           = bayes_opt_model,
        acquisition_rule = trieste.acquisition.rule.EfficientGlobalOptimization(
            optimizer = trieste.acquisition.optimizer.generate_continuous_optimizer(
                num_optimization_runs = 500
            )
        )
    )
    return (ask_tell,)


# ============================================================
# Bayesian optimisation loop
# ============================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bayesian Optimisation Loop

    Infeasible proposals are evaluated but **not** added to the GPR dataset.
    Only feasible proposals update the GPR.
    """)
    return


@app.cell
def _(ask_tell, num_analyses, np, objective, trieste):
    num_repeats    = 500
    when_to_print  = 50
    n_feasible_bo  = 0
    n_total_bo     = 0

    for _i in range(num_repeats):
        x_new = ask_tell.ask()

        _x, _y, _feas = objective(x_new, num_analyses)
        n_total_bo += 1

        if _feas:
            n_feasible_bo += 1
            # only update GPR if feasible
            ask_tell.tell(trieste.data.Dataset(
                query_points = _x,
                observations = _y
            ))
        else:
            # tell with empty dataset — GPR not updated
            ask_tell.tell(trieste.data.Dataset(
                query_points = np.reshape(np.array([]), (0, _x.shape[1])),
                observations = np.reshape(np.array([]), (0, 1))
            ))

        if (_i + 1) % when_to_print == 0:
            _current_best = float(ask_tell.to_result().try_get_final_dataset().observations.numpy().min())
            print(f"\nLoop {_i+1} completed. "
                  f"Feasible: {n_feasible_bo}/{n_total_bo} "
                  f"({100*n_feasible_bo/n_total_bo:.0f}%). "
                  f"Best obj: {_current_best:.4f}", end="")
        elif (_i > when_to_print) and ((_i + 1) % 5 == 0):
            print(".", end="")

    print(f"\nDone. Feasible BO proposals: {n_feasible_bo}/{num_repeats}")
    return n_feasible_bo, n_total_bo, num_repeats, when_to_print


# ============================================================
# Results
# ============================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"# Results")
    return


@app.cell
def _(ask_tell, find_n_integer, np, num_analyses, reverse_to_boundaries, target_power):
    final_dataset = ask_tell.to_result().try_get_final_dataset()

    best_idx    = int(np.argmin(final_dataset.observations.numpy()))
    best_params = final_dataset.query_points.numpy()[best_idx]
    best_obj    = float(final_dataset.observations.numpy()[best_idx])

    best_upper, best_lower = reverse_to_boundaries(best_params, num_analyses)
    best_n = find_n_integer(best_upper, best_lower, target_power)

    print(f"Best objective value: {best_obj:.4f}")
    print(f"n per arm per stage:  {best_n}")
    print(f"Upper boundaries:     {np.round(best_upper, 4)}")
    print(f"Lower boundaries:     {np.round(best_lower, 4)}")
    return best_lower, best_n, best_obj, best_upper, final_dataset


@app.cell
def _(best_lower, best_n, best_upper, delta1, num_analyses, sigma2, sim, target_alpha, target_power, tri_obj, tri_obj_true):
    _trial = sim.group_sequential_designs(
        n_analyses=num_analyses, upper_bounds=best_upper, lower_bounds=best_lower,
        n_patients=best_n, null_hypothesis=0, alt_hypothesis=delta1, variance=sigma2
    )
    print(f"Achieved alpha':  {_trial[1]:.4f}  (target <= {target_alpha})")
    print(f"Achieved power':  {_trial[2]:.4f}  (target >= {target_power})")
    print()
    print(f"Deflated triangular benchmark (integer n): {tri_obj:.4f}")
    print(f"Non-deflated triangular benchmark (integer n): {tri_obj_true:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## Objective function history (feasible designs only)")
    return


@app.cell
def _(final_dataset, np, plt, tri_obj, tri_obj_true):
    _obs         = final_dataset.observations.numpy().flatten()
    _running_min = np.minimum.accumulate(_obs)

    _fig, _axes = plt.subplots(nrows=2, figsize=(10, 7))

    _axes[0].plot(_obs, alpha=0.6, color="steelblue")
    _axes[0].axhline(y=tri_obj, color="orange", linestyle="--",
                     label=f"Deflated triangular ({tri_obj:.4f})")
    _axes[0].axhline(y=tri_obj_true, color="red", linestyle="--",
                     label=f"Non-deflated triangular ({tri_obj_true:.4f})")
    _axes[0].set_xlabel("Feasible iteration")
    _axes[0].set_ylabel("Objective value")
    _axes[0].set_title("Objective history (feasible designs only)")
    _axes[0].legend()

    _axes[1].plot(_running_min, color="steelblue")
    _axes[1].axhline(y=tri_obj, color="orange", linestyle="--",
                     label=f"Deflated triangular ({tri_obj:.4f})")
    _axes[1].axhline(y=tri_obj_true, color="red", linestyle="--",
                     label=f"Non-deflated triangular ({tri_obj_true:.4f})")
    _axes[1].set_xlabel("Feasible iteration")
    _axes[1].set_ylabel("Best objective so far")
    _axes[1].set_title("Running minimum")
    _axes[1].legend()

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## Best design vs triangular benchmark")
    return


@app.cell
def _(best_lower, best_upper, np, plt):
    _fig, _ax = plt.subplots(figsize=(8, 5))

    _ax.plot([1, 2, 3], best_upper, color="blue", label="BO best (upper)")
    _ax.plot([1, 2, 3], np.concatenate((best_lower[:2], [best_upper[2]])),
             color="blue", linestyle="--", label="BO best (lower)")
    _ax.plot([1, 2, 3], [2.1196, 1.8735, 1.8356], color="red",
             linewidth=2, label="Triangular (upper)")
    _ax.plot([1, 2, 3], [0.0, 1.1241, 1.8356], color="red",
             linewidth=2, linestyle="--", label="Triangular (lower)")

    _ax.set_xlabel("Stage")
    _ax.set_ylabel("Standardised boundary")
    _ax.set_title("Best BO design vs triangular benchmark")
    _ax.legend()
    _fig
    return


# ============================================================
# Save results
# ============================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"# Save Results")
    return


@app.cell
def _(ask_tell, np, pd):
    _data = ask_tell.to_result().try_get_final_dataset()

    _df = pd.DataFrame(
        data    = _data.query_points.numpy(),
        columns = ["c", "delta_u3", "delta_l3", "delta_u2", "delta_l2"]
    )
    _df["obj_f"] = _data.observations.numpy()
    _df.to_csv("/tf/paul/feasible_only_bo_results.csv", index=False)

    print(f"Saved {_df.shape[0]} feasible designs.")
    print(f"Best objective: {_df['obj_f'].min():.4f}")
    print(_df.describe())
    return


if __name__ == "__main__":
    app.run()
