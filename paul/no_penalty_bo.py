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
    # Delta-Minimax Group Sequential Design — No Penalty BO
    ## Reverse Parameterisation + Integer $n$ + Local Search + No Penalty

    Identical to the simple local BO except the objective is evaluated
    as the raw $\max_\delta E[N \mid \delta] / \mu$ with **no penalty** for
    $\alpha'$ violations. This allows us to understand the true shape of the
    objective surface without penalty distortion.

    Risk: infeasible designs with liberal boundaries may have lower ESS than
    feasible ones, potentially misleading the GPR. The assessment cell at the
    end reports what fraction of the best designs found are actually feasible.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Imports
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Trial Design Settings
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Integer $n$ by Bisection

    Finds smallest integer $n$ per arm per stage achieving target power.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Objective Function

    $$f(\tilde{\theta}) = \frac{\max_\delta E[N \mid \delta]}{\mu}
    + \mu \cdot \left(\frac{\alpha' - \alpha^*}{\alpha^*}\right)^2
    \cdot \mathbf{1}\{\alpha' > \alpha^*\}$$

    $\beta' \leq \beta^*$ is guaranteed by the integer bisection.
    Only $\alpha'$ requires a penalty term.

    If no feasible $n$ exists, returns a large fixed value (2.0) so the
    GPR learns to avoid such boundary combinations.
    """)
    return


@app.cell
def _(
    delta1,
    find_n_integer,
    mu,
    np,
    reverse_to_boundaries,
    sigma2,
    sim,
    ss,
    target_alpha,
    target_power,
):
    def objective(params, K):
        params = np.asarray(params).flatten()
        upper_bounds, lower_bounds = reverse_to_boundaries(params, K)

        # find integer n achieving target power
        n = find_n_integer(upper_bounds, lower_bounds, target_power)

        if n is None:
            # no feasible n — return large fixed value
            return np.array([params]), np.array([[2.0]]), False

        # raw ESS objective — no alpha' penalty
        max_ess = ss.max_ess(
            n_analyses=K, upper_bounds=upper_bounds, lower_bounds=lower_bounds,
            n_patients=n, null_hypothesis=0, variance=sigma2
        )
        y = float(max_ess / mu)

        # check alpha' for feasibility tracking only — does NOT affect objective
        trial = sim.group_sequential_designs(
            n_analyses=K, upper_bounds=upper_bounds, lower_bounds=lower_bounds,
            n_patients=n, null_hypothesis=0, alt_hypothesis=delta1, variance=sigma2
        )
        feasible = bool(trial[1] <= target_alpha)

        return np.array([params]), np.array([[y]]), feasible

    return (objective,)


@app.cell
def _(bd, boundaries_to_reverse, np, num_analyses, objective, target_alpha):
    _alpha_deflated = target_alpha * 49 / 50

    _tri = bd.calculate_triangular_boundaries(
        n_analyses=num_analyses, alpha=_alpha_deflated, delta=1.0, n_patients=20
    )
    tri_params = boundaries_to_reverse(_tri[0], _tri[1])
    c0         = tri_params[0]

    _, _y, _feas = objective(tri_params, num_analyses)
    tri_obj = float(_y[0, 0])
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
    tri_obj_true = float(_y_true[0, 0])
    print(f"Non-deflated triangular objective (integer n): {tri_obj_true:.4f}, feasible: {_feas_true}")
    print(f"  (note: raw ESS/mu regardless of alpha')")
    return (tri_obj_true,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Search Space
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Initialisation

    - Local: 15 perturbations around triangular design only
    - Global: 5 perturbations around each of Pocock, O'Brien-Fleming, triangular
    - Plus up to 15 feasible random points from pool within search space
    """)
    return


@app.cell
def _(
    bd,
    boundaries_to_reverse,
    np,
    num_analyses,
    objective,
    search_space,
    target_alpha,
    tri_params,
    use_local_search,
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
    n_infeasible_init = 0
    for _ref in _refs:
        for _ in range(_n_pert):
            _scale = np.clip(np.abs(_ref) * 0.10, 0.01, 0.3)
            _p     = _ref + rng_init.normal(scale=_scale, size=len(_ref))
            _p[1:] = np.clip(_p[1:], 0.0, 4.0)
            _x, _y, _feas = objective(_p, num_analyses)
            init_x.append(_x); init_y.append(_y)
            if not _feas:
                n_infeasible_init += 1

    print(f"Perturbation points: {len(init_x)} total, "
          f"{len(init_x)-n_infeasible_init} feasible (alpha' ok), "
          f"{n_infeasible_init} infeasible (all included)")

    # random pool — keep all points regardless of feasibility
    n_random = 15
    _lb = search_space.lower.numpy()
    _ub = search_space.upper.numpy()
    n_rand_infeas = 0
    for _p in rng_init.uniform(_lb, _ub, size=(n_random, len(_lb))):
        _x, _y, _feas = objective(_p, num_analyses)
        init_x.append(_x); init_y.append(_y)
        if not _feas:
            n_rand_infeas += 1

    print(f"Total initial points: {len(init_x)}")
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
    print(f"Initial dataset: {design_matrix.shape[0]} points")
    return design_matrix, initial_data, output_vals


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GP Model

    Single GPR with Matern52 ARD kernel trained on all evaluations.
    No VGP, no failure region.
    """)
    return


@app.cell
def _(GaussianProcessRegression, design_matrix, gpflow, output_vals):
    _kernel = gpflow.kernels.Matern52(
        lengthscales=[1.0] * design_matrix.shape[1]
    )
    _gpr = gpflow.models.GPR(
        data      = (design_matrix, output_vals),
        kernel    = _kernel,
        likelihood = gpflow.likelihoods.Gaussian()
    )
    gpflow.utilities.print_summary(_gpr, fmt="notebook")
    bayes_opt_model = GaussianProcessRegression(_gpr)
    return (bayes_opt_model,)


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bayesian Optimisation Loop

    Standard ask-tell loop. Every evaluation feeds the GPR — feasible designs
    contribute clean objective values, infeasible designs contribute penalised
    values that guide the GPR away from the infeasible region.
    """)
    return


@app.cell
def _(ask_tell, np, num_analyses, objective, trieste):
    num_repeats   = 500
    when_to_print = 50
    n_feasible_bo = 0

    for _i in range(num_repeats):
        x_new = ask_tell.ask()

        _x, _y, _feas = objective(x_new, num_analyses)

        if _feas:
            n_feasible_bo += 1

        # always update GPR — no penalty so all values are meaningful ESS/mu
        ask_tell.tell(trieste.data.Dataset(
            query_points = _x,
            observations = _y
        ))

        if (_i + 1) % when_to_print == 0:
            _all_obs = ask_tell.to_result().try_get_final_dataset().observations.numpy().flatten()
            _current_best = float(_all_obs.min())
            print(f"\nLoop {_i+1} completed. "
                  f"Feasible (alpha' ok): {n_feasible_bo}/{_i+1} "
                  f"({100*n_feasible_bo/(_i+1):.0f}%). "
                  f"Best obj: {_current_best:.4f}", end="")
        elif (_i > when_to_print) and ((_i + 1) % 5 == 0):
            print(".", end="")

    print(f"\nDone. Feasible BO proposals: {n_feasible_bo}/{num_repeats}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Random Search Baseline

    Pure random search over the same local search space with the same number
    of evaluations. No GPR, no acquisition function — just uniform random
    sampling. The running minimum of the random search is compared against
    the BO in the plots below to assess whether the BO is adding value
    over brute force.
    """)
    return


@app.cell
def _(np, num_analyses, objective, search_space):
    rng_baseline = np.random.default_rng(seed=123)
    n_baseline   = 530   # same total as BO (30 init + 500 loop)

    _lb = search_space.lower.numpy()
    _ub = search_space.upper.numpy()
    _candidates = rng_baseline.uniform(_lb, _ub, size=(n_baseline, len(_lb)))

    baseline_objs     = []
    baseline_feasible = []

    for _p in _candidates:
        _, _y, _feas = objective(_p, num_analyses)
        baseline_objs.append(float(_y[0, 0]))
        baseline_feasible.append(_feas)

    baseline_objs     = np.array(baseline_objs)
    baseline_feasible = np.array(baseline_feasible)
    baseline_running_min = np.minimum.accumulate(baseline_objs)

    n_feas_baseline = int(baseline_feasible.sum())
    best_feas_baseline = float(baseline_objs[baseline_feasible].min()) if n_feas_baseline > 0 else float('nan')

    print(f"Random search baseline: {n_baseline} evaluations")
    print(f"Feasible: {n_feas_baseline}/{n_baseline} ({100*n_feas_baseline/n_baseline:.1f}%)")
    print(f"Best overall obj:          {baseline_objs.min():.4f}")
    print(f"Best feasible obj:         {best_feas_baseline:.4f}")

    return baseline_feasible, baseline_objs, baseline_running_min, best_feas_baseline


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Results
    """)
    return


@app.cell
def _(
    ask_tell,
    find_n_integer,
    np,
    num_analyses,
    reverse_to_boundaries,
    target_power,
):
    final_dataset = ask_tell.to_result().try_get_final_dataset()
    _obs          = final_dataset.observations.numpy().flatten()

    # best overall point — may be infeasible
    best_idx    = int(np.argmin(_obs))
    best_params = final_dataset.query_points.numpy()[best_idx]
    best_obj    = float(_obs[best_idx])
    best_upper, best_lower = reverse_to_boundaries(best_params, num_analyses)
    best_n      = find_n_integer(best_upper, best_lower, target_power)

    print(f"Best objective value (may be infeasible): {best_obj:.4f}")
    print(f"n per arm per stage:  {best_n}")
    print(f"Upper boundaries:     {np.round(best_upper, 4)}")
    print(f"Lower boundaries:     {np.round(best_lower, 4)}")
    return best_lower, best_n, best_obj, best_upper, final_dataset


@app.cell
def _(
    best_lower,
    best_n,
    best_upper,
    delta1,
    num_analyses,
    sigma2,
    sim,
    target_alpha,
    target_power,
    tri_obj,
    tri_obj_true,
):
    _trial = sim.group_sequential_designs(
        n_analyses=num_analyses, upper_bounds=best_upper, lower_bounds=best_lower,
        n_patients=best_n, null_hypothesis=0, alt_hypothesis=delta1, variance=sigma2
    )
    _feas = _trial[1] <= target_alpha
    print(f"Achieved alpha':  {_trial[1]:.4f}  (target <= {target_alpha}) — {'FEASIBLE' if _feas else 'INFEASIBLE'}")
    print(f"Achieved power':  {_trial[2]:.4f}  (target >= {target_power})")
    print()
    print(f"Deflated triangular benchmark (integer n): {tri_obj:.4f}")
    print(f"Non-deflated triangular (raw ESS, integer n): {tri_obj_true:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Assessment

    Since there is no penalty, the GPR may have been guided towards infeasible
    designs with low ESS. This cell checks the full dataset for how many designs
    are truly feasible and what the best feasible objective is.
    """)
    return


@app.cell
def _(ask_tell, best_feas_baseline, delta1, find_n_integer, np, num_analyses, reverse_to_boundaries, sigma2, sim, target_alpha, target_power, tri_obj):
    _dataset    = ask_tell.to_result().try_get_final_dataset()
    _all_params = _dataset.query_points.numpy()
    _all_obs    = _dataset.observations.numpy().flatten()

    _feasible_objs   = []
    _infeasible_objs = []

    for _p, _y in zip(_all_params, _all_obs):
        _u, _l = reverse_to_boundaries(_p, num_analyses)
        _n     = find_n_integer(_u, _l, target_power)
        if _n is None:
            _infeasible_objs.append(_y)
            continue
        _t = sim.group_sequential_designs(
            n_analyses=num_analyses, upper_bounds=_u, lower_bounds=_l,
            n_patients=_n, null_hypothesis=0, alt_hypothesis=delta1, variance=sigma2
        )
        if _t[1] <= target_alpha:
            _feasible_objs.append(_y)
        else:
            _infeasible_objs.append(_y)

    n_total     = len(_all_obs)
    n_feas      = len(_feasible_objs)
    n_infeas    = len(_infeasible_objs)
    best_feas   = float(np.min(_feasible_objs))   if _feasible_objs   else float('nan')
    best_infeas = float(np.min(_infeasible_objs)) if _infeasible_objs else float('nan')

    print(f"=== Assessment ===")
    print(f"Total designs:              {n_total}")
    print(f"Feasible (alpha' ok):       {n_feas} ({100*n_feas/n_total:.1f}%)")
    print(f"Infeasible:                 {n_infeas} ({100*n_infeas/n_total:.1f}%)")
    print()
    print(f"Best feasible objective:    {best_feas:.4f}  (deflated triangular: {tri_obj:.4f})")
    print(f"Best infeasible objective:  {best_infeas:.4f}")
    print()
    print(f"Random search best feasible: {best_feas_baseline:.4f}")
    print(f"BO improvement over random:  {best_feas_baseline - best_feas:.4f}")
    if not np.isnan(best_feas) and not np.isnan(best_infeas):
        _gap = best_feas - best_infeas
        print(f"\nGap (best feasible - best infeasible): {_gap:.4f}")
        if _gap > 0.01:
            print(">> GPR was misled: optimiser chased infeasible designs with low ESS.")
            print("   A penalty is needed to constrain the search.")
        else:
            print(">> GPR was not significantly misled: penalty may not be necessary.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Objective function history
    """)
    return


@app.cell
def _(baseline_running_min, final_dataset, np, plt, tri_obj, tri_obj_true):
    _obs         = final_dataset.observations.numpy().flatten()
    _running_min = np.minimum.accumulate(_obs)

    _fig, _axes = plt.subplots(nrows=2, figsize=(10, 7))

    # top panel: raw BO history
    _axes[0].plot(_obs, alpha=0.6, color="steelblue", label="BO proposals")
    _axes[0].axhline(y=tri_obj, color="orange", linestyle="--",
                     label=f"Deflated triangular ({tri_obj:.4f})")
    _axes[0].axhline(y=tri_obj_true, color="red", linestyle="--",
                     label=f"Non-deflated triangular ({tri_obj_true:.4f})")
    _axes[0].set_ylim(0, 2.1)
    _axes[0].set_xlabel("Iteration")
    _axes[0].set_ylabel("Objective value")
    _axes[0].set_title("Objective history (no penalty)")
    _axes[0].legend()

    # bottom panel: BO vs random search running minima
    _n = min(len(_running_min), len(baseline_running_min))
    _axes[1].plot(_running_min[:_n], color="steelblue", label="BO running minimum")
    _axes[1].plot(baseline_running_min[:_n], color="grey", linestyle="--",
                  label="Random search running minimum")
    _axes[1].axhline(y=tri_obj, color="orange", linestyle="--",
                     label=f"Deflated triangular ({tri_obj:.4f})")
    _axes[1].axhline(y=tri_obj_true, color="red", linestyle="--",
                     label=f"Non-deflated triangular ({tri_obj_true:.4f})")
    _axes[1].set_xlabel("Iteration")
    _axes[1].set_ylabel("Best objective so far")
    _axes[1].set_title("BO vs Random Search — Running Minimum")
    _axes[1].legend()

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Best design vs triangular benchmark
    """)
    return


@app.cell
def _(best_lower, best_upper, np, plt):
    if best_upper is not None:
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Save Results
    """)
    return


@app.cell
def _(ask_tell, pd):
    _data = ask_tell.to_result().try_get_final_dataset()

    _df = pd.DataFrame(
        data    = _data.query_points.numpy(),
        columns = ["c", "delta_u3", "delta_l3", "delta_u2", "delta_l2"]
    )
    _df["obj_f"]    = _data.observations.numpy()
    _df["feasible"] = (_df["obj_f"] < 2.0).astype(int)

    _df.to_csv("/tf/paul/no_penalty_bo_results.csv", index=False)
    print(f"Saved {_df.shape[0]} rows.")
    print(f"Feasibility rate: {_df['feasible'].mean():.1%}")
    print(_df.describe())
    return


if __name__ == "__main__":
    app.run()
