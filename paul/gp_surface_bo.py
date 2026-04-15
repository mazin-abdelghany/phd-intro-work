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
    # GP Surrogate Visualisation
    ## No-Penalty BO with Checkpoint Plots

    Runs the same no-penalty BO but saves visualisations at regular checkpoints.
    At each checkpoint, for three pairs of parameters, plots:

    - **GP posterior mean** — what the surrogate thinks the objective looks like
    - **GP posterior std** — where the surrogate is uncertain
    - **Evaluated points** overlaid, coloured by objective value

    All other parameters are fixed at the current best point found so far.
    This shows how the GPR's understanding of the surface evolves during the run.

    Plots are saved to `/tf/paul/plots/` as PNG files.
    """)
    return


# ============================================================
# Imports
# ============================================================

@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend for script mode
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import os
    os.makedirs("/tf/paul/plots", exist_ok=True)
    return matplotlib, np, os, pd, plt, gridspec


@app.cell
def _():
    import gpflow
    import tensorflow as tf
    import trieste
    from trieste.space import Box
    from trieste.models.gpflow.models import GaussianProcessRegression
    return Box, GaussianProcessRegression, gpflow, tf, trieste


@app.cell
def _():
    from py_group_sequential_designs import generate_boundaries as bd
    from py_group_sequential_designs import simulate as sim
    from py_group_sequential_designs import sample_size as ss
    return bd, sim, ss


# ============================================================
# Trial design settings
# ============================================================

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
# Reverse parameterisation
# ============================================================

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
# Objective function (no penalty)
# ============================================================

@app.cell
def _(delta1, find_n_integer, mu, np, num_analyses, reverse_to_boundaries, sigma2, sim, ss, target_alpha, target_power):
    def objective(params, K):
        params = np.asarray(params).flatten()
        upper_bounds, lower_bounds = reverse_to_boundaries(params, K)
        n = find_n_integer(upper_bounds, lower_bounds, target_power)
        if n is None:
            return np.array([params]), np.array([[2.0]]), False
        max_ess = ss.max_ess(
            n_analyses=K, upper_bounds=upper_bounds, lower_bounds=lower_bounds,
            n_patients=n, null_hypothesis=0, variance=sigma2
        )
        y = float(max_ess / mu)
        trial = sim.group_sequential_designs(
            n_analyses=K, upper_bounds=upper_bounds, lower_bounds=lower_bounds,
            n_patients=n, null_hypothesis=0, alt_hypothesis=delta1, variance=sigma2
        )
        feasible = bool(trial[1] <= target_alpha)
        return np.array([params]), np.array([[y]]), feasible
    return (objective,)


# ============================================================
# Reference designs
# ============================================================

@app.cell
def _(bd, boundaries_to_reverse, find_n_integer, mu, np, num_analyses, reverse_to_boundaries, sim, ss, target_alpha, target_power):
    _alpha_deflated = target_alpha * 49 / 50
    _tri = bd.calculate_triangular_boundaries(
        n_analyses=num_analyses, alpha=_alpha_deflated, delta=1.0, n_patients=20
    )
    tri_params = boundaries_to_reverse(_tri[0], _tri[1])
    c0         = tri_params[0]

    _tri_upper, _tri_lower = reverse_to_boundaries(tri_params, num_analyses)
    _n = find_n_integer(_tri_upper, _tri_lower, target_power)
    _max_ess = ss.max_ess(
        n_analyses=num_analyses, upper_bounds=_tri_upper, lower_bounds=_tri_lower,
        n_patients=_n, null_hypothesis=0, variance=3.0
    )
    tri_obj = float(_max_ess / mu)
    print(f"Deflated triangular objective: {tri_obj:.4f}")
    print(f"Triangular params (5-dim): {np.round(tri_params, 4)}")
    return c0, tri_obj, tri_params


# ============================================================
# Search space
# ============================================================

@app.cell
def _(Box, c0, np, tri_params):
    _hw_c     = 0.3
    _hw_delta = 0.3
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
    search_space = Box(lower=_lower, upper=_upper)
    print(f"Search space lower: {np.round(search_space.lower.numpy(), 3)}")
    print(f"Search space upper: {np.round(search_space.upper.numpy(), 3)}")
    return (search_space,)


# ============================================================
# Initialisation
# ============================================================

@app.cell
def _(bd, boundaries_to_reverse, c0, np, num_analyses, objective, search_space, target_alpha, tri_params):
    rng_init = np.random.default_rng(seed=42)
    _alpha_deflated = target_alpha * 49 / 50

    _poc = bd.calculate_pocock_boundaries(n_analyses=num_analyses, alpha=_alpha_deflated, n_patients=20)
    poc_params = boundaries_to_reverse(_poc[0], _poc[1])
    _obf = bd.calculate_of_boundaries(n_analyses=num_analyses, alpha=_alpha_deflated, n_patients=20)
    obf_params = boundaries_to_reverse(_obf[0], _obf[1])

    init_x, init_y = [], []
    for _ref in [tri_params]:   # local: perturb around triangular only
        for _ in range(15):
            _scale = np.clip(np.abs(_ref) * 0.10, 0.01, 0.3)
            _p     = _ref + rng_init.normal(scale=_scale, size=len(_ref))
            _p[1:] = np.clip(_p[1:], 0.0, 4.0)
            _x, _y, _ = objective(_p, num_analyses)
            init_x.append(_x); init_y.append(_y)

    _lb = search_space.lower.numpy()
    _ub = search_space.upper.numpy()
    for _p in rng_init.uniform(_lb, _ub, size=(15, len(_lb))):
        _x, _y, _ = objective(_p, num_analyses)
        init_x.append(_x); init_y.append(_y)

    print(f"Total initial points: {len(init_x)}")
    return init_x, init_y


@app.cell
def _(init_x, init_y, np, trieste):
    design_matrix = np.concatenate(init_x)
    output_vals   = np.concatenate(init_y)
    initial_data  = trieste.data.Dataset(
        query_points=design_matrix, observations=output_vals
    )
    print(f"Initial dataset: {design_matrix.shape[0]} points")
    return design_matrix, initial_data, output_vals


# ============================================================
# GP model
# ============================================================

@app.cell
def _(GaussianProcessRegression, design_matrix, gpflow, output_vals):
    _kernel = gpflow.kernels.Matern52(lengthscales=[1.0] * design_matrix.shape[1])
    _gpr    = gpflow.models.GPR(
        data=(design_matrix, output_vals),
        kernel=_kernel,
        likelihood=gpflow.likelihoods.Gaussian()
    )
    bayes_opt_model = GaussianProcessRegression(_gpr)
    return (bayes_opt_model,)


@app.cell
def _(bayes_opt_model, initial_data, search_space, trieste):
    ask_tell = trieste.ask_tell_optimization.AskTellOptimizer(
        search_space     = search_space,
        datasets         = initial_data,
        models           = bayes_opt_model,
        acquisition_rule = trieste.acquisition.rule.EfficientGlobalOptimization(
            optimizer=trieste.acquisition.optimizer.generate_continuous_optimizer(
                num_optimization_runs=500
            )
        )
    )
    return (ask_tell,)


# ============================================================
# Visualisation helper
# ============================================================

@app.cell
def _(np, plt, gridspec, search_space, tf, tri_params):
    # parameter names for axis labels
    PARAM_NAMES = [r"$c$", r"$\Delta u_3$", r"$\Delta l_3$", r"$\Delta u_2$", r"$\Delta l_2$"]

    # pairs of parameters to plot (index into the 5-dim vector)
    PLOT_PAIRS = [
        (0, 2),   # c vs delta_l3  — most influential pair
        (0, 3),   # c vs delta_u2
        (3, 4),   # delta_u2 vs delta_l2 — first-stage increments
    ]

    GRID_SIZE  = 40   # resolution of the GP posterior grid

    def make_checkpoint_plot(ask_tell, iteration, all_params, all_objs, all_feas):
        """
        For each parameter pair, plot GP posterior mean and std on a 2D grid,
        with all evaluated points overlaid. Other parameters fixed at current best.
        """
        # current best feasible point
        _feas_mask = np.array(all_feas)
        if _feas_mask.any():
            _feas_objs = np.array(all_objs)
            _feas_objs[~_feas_mask] = np.inf
            _best_params = all_params[np.argmin(_feas_objs)]
        else:
            _best_params = tri_params.copy()

        _lb = search_space.lower.numpy()
        _ub = search_space.upper.numpy()

        n_pairs = len(PLOT_PAIRS)
        fig = plt.figure(figsize=(6 * n_pairs, 8))
        fig.suptitle(f"GP Surrogate — Iteration {iteration}\n"
                     f"(other params fixed at current best: {np.round(_best_params, 3)})",
                     fontsize=11)
        gs = gridspec.GridSpec(2, n_pairs, figure=fig, hspace=0.4, wspace=0.3)

        for col, (i, j) in enumerate(PLOT_PAIRS):
            # build grid
            _xi = np.linspace(_lb[i], _ub[i], GRID_SIZE)
            _xj = np.linspace(_lb[j], _ub[j], GRID_SIZE)
            _Xi, _Xj = np.meshgrid(_xi, _xj)

            # construct full parameter matrix — fix non-plotted dims at best_params
            _grid_pts = np.tile(_best_params, (GRID_SIZE * GRID_SIZE, 1))
            _grid_pts[:, i] = _Xi.ravel()
            _grid_pts[:, j] = _Xj.ravel()
            _grid_tf = tf.cast(_grid_pts, tf.float64)

            # GP posterior
            _model = ask_tell.to_result().try_get_final_model()
            _mean, _var = _model.predict(_grid_tf)
            _mean_np = _mean.numpy() if hasattr(_mean, 'numpy') else np.array(_mean)
            _var_np  = _var.numpy()  if hasattr(_var,  'numpy') else np.array(_var)
            _mean = _mean_np.reshape(GRID_SIZE, GRID_SIZE)
            _std  = np.sqrt(_var_np).reshape(GRID_SIZE, GRID_SIZE)

            # posterior mean
            ax_mean = fig.add_subplot(gs[0, col])
            _cm = ax_mean.contourf(_Xi, _Xj, _mean, levels=20, cmap="viridis")
            plt.colorbar(_cm, ax=ax_mean, label="Predicted mean")
            ax_mean.scatter(
                all_params[all_feas, i], all_params[all_feas, j],
                c=np.array(all_objs)[all_feas], cmap="viridis",
                edgecolors="white", linewidths=0.5, s=30, label="Feasible"
            )
            ax_mean.scatter(
                all_params[~all_feas, i], all_params[~all_feas, j],
                c="red", marker="x", s=20, alpha=0.5, label="Infeasible"
            )
            ax_mean.set_xlabel(PARAM_NAMES[i])
            ax_mean.set_ylabel(PARAM_NAMES[j])
            ax_mean.set_title(f"Posterior mean\n{PARAM_NAMES[i]} vs {PARAM_NAMES[j]}")

            # posterior std
            ax_std = fig.add_subplot(gs[1, col])
            _cs = ax_std.contourf(_Xi, _Xj, _std, levels=20, cmap="plasma")
            plt.colorbar(_cs, ax=ax_std, label="Posterior std")
            ax_std.scatter(
                all_params[:, i], all_params[:, j],
                c="white", edgecolors="grey", s=15, alpha=0.5
            )
            ax_std.set_xlabel(PARAM_NAMES[i])
            ax_std.set_ylabel(PARAM_NAMES[j])
            ax_std.set_title(f"Posterior std (uncertainty)\n{PARAM_NAMES[i]} vs {PARAM_NAMES[j]}")

        _path = f"/tf/paul/plots/checkpoint_{iteration:04d}.png"
        fig.savefig(_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {_path}")

    return PARAM_NAMES, PLOT_PAIRS, make_checkpoint_plot


# ============================================================
# BO loop with checkpoint visualisations
# ============================================================

@app.cell
def _(ask_tell, make_checkpoint_plot, np, num_analyses, objective, trieste):
    num_repeats       = 500
    when_to_print     = 50
    checkpoint_every  = 25    # save GP surface plot every N iterations
    n_feasible_bo     = 0

    # track all evaluated points for overlay
    _dataset       = ask_tell.to_result().try_get_final_dataset()
    _qp            = _dataset.query_points
    _obs           = _dataset.observations
    # handle both TF tensor and numpy array
    _qp_np  = _qp.numpy()  if hasattr(_qp,  'numpy') else np.array(_qp)
    _obs_np = _obs.numpy() if hasattr(_obs, 'numpy') else np.array(_obs)
    _all_params    = list(_qp_np)
    _all_objs_raw  = list(_obs_np.flatten())
    _all_feas      = [True] * len(_all_params)  # initial points assumed feasible for plotting

    # initial checkpoint
    print("Saving initial checkpoint...")
    make_checkpoint_plot(
        ask_tell   = ask_tell,
        iteration  = 0,
        all_params = np.array(_all_params),
        all_objs   = np.array(_all_objs_raw),
        all_feas   = np.array(_all_feas, dtype=bool)
    )

    for _i in range(num_repeats):
        x_new = ask_tell.ask()

        _x, _y, _feas = objective(x_new, num_analyses)

        if _feas:
            n_feasible_bo += 1

        ask_tell.tell(trieste.data.Dataset(
            query_points=_x, observations=_y
        ))

        _all_params.append(_x[0])
        _all_objs_raw.append(float(_y[0, 0]))
        _all_feas.append(_feas)

        # checkpoint plot
        if (_i + 1) % checkpoint_every == 0:
            _arr_params = np.array(_all_params)
            _arr_objs   = np.array(_all_objs_raw)
            _arr_feas   = np.array(_all_feas, dtype=bool)
            _best = float(_arr_objs[_arr_feas].min()) if _arr_feas.any() else float('nan')
            print(f"\nLoop {_i+1}. Feasible: {n_feasible_bo}/{_i+1} "
                  f"({100*n_feasible_bo/(_i+1):.0f}%). Best: {_best:.4f}")
            make_checkpoint_plot(
                ask_tell   = ask_tell,
                iteration  = _i + 1,
                all_params = _arr_params,
                all_objs   = _arr_objs,
                all_feas   = _arr_feas
            )
        elif (_i > when_to_print) and ((_i + 1) % 5 == 0):
            print(".", end="")

    print(f"\nDone. Feasible: {n_feasible_bo}/{num_repeats}")
    return n_feasible_bo, num_repeats


# ============================================================
# Results
# ============================================================

@app.cell
def _(ask_tell, find_n_integer, np, num_analyses, reverse_to_boundaries, target_power):
    final_dataset = ask_tell.to_result().try_get_final_dataset()
    _obs          = final_dataset.observations.numpy().flatten()

    best_idx    = int(np.argmin(_obs))
    best_params = final_dataset.query_points.numpy()[best_idx]
    best_obj    = float(_obs[best_idx])
    best_upper, best_lower = reverse_to_boundaries(best_params, num_analyses)
    best_n      = find_n_integer(best_upper, best_lower, target_power)

    print(f"Best objective: {best_obj:.4f}")
    print(f"n per arm:      {best_n}")
    print(f"Upper:          {np.round(best_upper, 4)}")
    print(f"Lower:          {np.round(best_lower, 4)}")
    return best_lower, best_n, best_obj, best_upper, final_dataset


@app.cell
def _(best_lower, best_n, best_upper, delta1, num_analyses, sigma2, sim, target_alpha, target_power, tri_obj):
    _trial = sim.group_sequential_designs(
        n_analyses=num_analyses, upper_bounds=best_upper, lower_bounds=best_lower,
        n_patients=best_n, null_hypothesis=0, alt_hypothesis=delta1, variance=sigma2
    )
    _feas = _trial[1] <= target_alpha
    print(f"alpha': {_trial[1]:.4f} ({'FEASIBLE' if _feas else 'INFEASIBLE'}), "
          f"power': {_trial[2]:.4f}")
    print(f"Deflated triangular benchmark: {tri_obj:.4f}")
    return


@app.cell
def _(ask_tell, delta1, find_n_integer, np, num_analyses, reverse_to_boundaries, sigma2, sim, target_alpha, target_power, tri_obj):
    _dataset    = ask_tell.to_result().try_get_final_dataset()
    _all_params = _dataset.query_points.numpy()
    _all_obs    = _dataset.observations.numpy().flatten()

    _feasible_objs   = []
    _infeasible_objs = []

    for _p, _y in zip(_all_params, _all_obs):
        _u, _l = reverse_to_boundaries(_p, num_analyses)
        _n     = find_n_integer(_u, _l, target_power)
        if _n is None:
            _infeasible_objs.append(_y); continue
        _t = sim.group_sequential_designs(
            n_analyses=num_analyses, upper_bounds=_u, lower_bounds=_l,
            n_patients=_n, null_hypothesis=0, alt_hypothesis=delta1, variance=sigma2
        )
        (_feasible_objs if _t[1] <= target_alpha else _infeasible_objs).append(_y)

    n_total   = len(_all_obs)
    best_feas = float(np.min(_feasible_objs)) if _feasible_objs else float('nan')

    print(f"=== Assessment ===")
    print(f"Feasible: {len(_feasible_objs)}/{n_total} ({100*len(_feasible_objs)/n_total:.1f}%)")
    print(f"Best feasible obj:   {best_feas:.4f}  (benchmark: {tri_obj:.4f})")
    print(f"Best infeasible obj: {float(np.min(_infeasible_objs)) if _infeasible_objs else float('nan'):.4f}")
    print(f"\nCheckpoint plots saved to /tf/paul/plots/")
    return


if __name__ == "__main__":
    app.run()
