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
    # Delta-Minimax Group Sequential Design — Simple Local BO
    ## Reverse Parameterisation + Integer $n$ + Local Search + $\alpha'$ Penalty

    A simpler alternative to the constrained BO approach. Key ideas:

    1. **Reverse parameterisation** — structural constraints automatic, 5 parameters
    2. **Integer $n$ by bisection** — $\beta' \leq \beta^*$ guaranteed by construction
    3. **Local search** — tight box around triangular design
    4. **Single GP** trained on all evaluations with a smooth $\alpha'$ penalty
    5. **Standard EI** — no VGP, no failure region machinery

    The penalty gives the GPR gradient information from infeasible designs.
    Since $\beta'$ is handled by bisection, only $\alpha'$ needs penalising —
    a simpler, smoother landscape than the two-constraint penalty in earlier runs.
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
            # no feasible n — return large value
            return np.array([params]), np.array([[2.0]])

        # evaluate alpha' at this n
        trial = sim.group_sequential_designs(
            n_analyses=K, upper_bounds=upper_bounds, lower_bounds=lower_bounds,
            n_patients=n, null_hypothesis=0, alt_hypothesis=delta1, variance=sigma2
        )
        alpha_prime = trial[1]

        # max ESS
        max_ess = ss.max_ess(
            n_analyses=K, upper_bounds=upper_bounds, lower_bounds=lower_bounds,
            n_patients=n, null_hypothesis=0, variance=sigma2
        )

        # objective: ESS term + alpha' penalty only
        ess_term = max_ess / mu

        if alpha_prime > target_alpha:
            penalty = mu * ((alpha_prime - target_alpha) / target_alpha) ** 2
        else:
            penalty = 0.0

        y = float(np.clip(ess_term + penalty, 0.0, 2.0))

        return np.array([params]), np.array([[y]])

    return (objective,)


@app.cell
def _(bd, boundaries_to_reverse, np, num_analyses, objective, target_alpha):
    _alpha_deflated = target_alpha * 49 / 50

    _tri = bd.calculate_triangular_boundaries(
        n_analyses=num_analyses, alpha=_alpha_deflated, delta=1.0, n_patients=20
    )
    tri_params = boundaries_to_reverse(_tri[0], _tri[1])
    c0         = tri_params[0]

    _, _y = objective(tri_params, num_analyses)
    tri_obj = float(_y[0, 0])
    print(f"Triangular benchmark objective: {tri_obj:.4f}")
    print(f"Triangular params (5-dim): {np.round(tri_params, 4)}")
    print(f"Meeting point c0 = {c0:.4f}")
    return c0, tri_obj, tri_params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Warm Start

    Optionally provide the best design from a previous run as the centre of the
    local search box. Set `warm_start_params` to the 5-dim reverse parameterisation
    vector of the best previous design, or leave as `None` to use the triangular design.

    Previous best: `[1.7964, 0.1643, 0.4340, 0.0, 0.9132]` (obj = 0.2784)
    """)
    return


@app.cell
def _(np, tri_params):
    # Set to None to use triangular design, or paste best params from previous run
    # warm_start_params = np.array([1.7964, 0.1643, 0.4340, 0.0, 0.9132])
    warm_start_params = None

    centre_params = warm_start_params if warm_start_params is not None else tri_params
    _label = "warm start" if warm_start_params is not None else "triangular design"
    print(f"Search box centred on: {_label}")
    print(f"Centre params: {np.round(centre_params, 4)}")
    return centre_params, warm_start_params


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
def _(Box, c0, centre_params, np, use_local_search):
    _hw_c     = 0.3
    _hw_delta = 0.3

    if use_local_search.value:
        _lower = np.array([
            centre_params[0] - _hw_c,
            max(0.0, centre_params[1] - _hw_delta),
            max(0.0, centre_params[2] - _hw_delta),
            max(0.0, centre_params[3] - _hw_delta),
            max(0.0, centre_params[4] - _hw_delta),
        ])
        _upper = np.array([
            centre_params[0] + _hw_c,
            centre_params[1] + _hw_delta,
            centre_params[2] + _hw_delta,
            centre_params[3] + _hw_delta,
            centre_params[4] + _hw_delta,
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
    centre_params,
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

    # local: perturb around centre_params (warm start or triangular)
    # global: perturb around all three reference designs
    _refs   = [centre_params] if use_local_search.value else [poc_params, obf_params, tri_params]
    _n_pert = 15 if use_local_search.value else 5

    init_x, init_y = [], []
    for _ref in _refs:
        for _ in range(_n_pert):
            _scale = np.clip(np.abs(_ref) * 0.10, 0.01, 0.3)
            _p     = _ref + rng_init.normal(scale=_scale, size=len(_ref))
            _p[1:] = np.clip(_p[1:], 0.0, 4.0)
            _x, _y = objective(_p, num_analyses)
            init_x.append(_x); init_y.append(_y)

    n_feasible_init = sum(1 for _y in init_y if _y[0,0] < 2.0)
    print(f"Perturbation points: {len(init_x)} total, {n_feasible_init} with obj < 2.0")

    # random pool — keep all (penalised values inform the GPR)
    n_random = 15
    _lb = search_space.lower.numpy()
    _ub = search_space.upper.numpy()
    for _p in rng_init.uniform(_lb, _ub, size=(n_random, len(_lb))):
        _x, _y = objective(_p, num_analyses)
        init_x.append(_x); init_y.append(_y)

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
def _(ask_tell, num_analyses, objective, trieste):
    num_repeats   = 500
    when_to_print = 50
    n_feasible_bo = 0

    for _i in range(num_repeats):
        x_new = ask_tell.ask()

        _x, _y = objective(x_new, num_analyses)

        _is_feasible = float(_y[0, 0]) < 2.0
        if _is_feasible:
            n_feasible_bo += 1

        ask_tell.tell(trieste.data.Dataset(
            query_points = _x,
            observations = _y
        ))

        if (_i + 1) % when_to_print == 0:
            _all_obs = ask_tell.to_result().try_get_final_dataset().observations.numpy().flatten()
            _feasible_obs = _all_obs[_all_obs < 2.0]
            _current_best = float(_feasible_obs.min()) if len(_feasible_obs) > 0 else float('nan')
            print(f"\nLoop {_i+1} completed. "
                  f"Feasible: {n_feasible_bo}/{_i+1} "
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

    Pure random search over the same local search space with the same total
    number of evaluations (30 initial + 500 BO = 530). Provides a trajectory
    to compare against the BO running minimum — if the BO running minimum
    descends faster or further, it is demonstrating genuine sample efficiency.
    """)
    return


@app.cell
def _(np, num_analyses, objective, search_space):
    rng_baseline  = np.random.default_rng(seed=123)
    n_baseline    = 530   # matches total BO evaluations

    _lb = search_space.lower.numpy()
    _ub = search_space.upper.numpy()

    baseline_objs = []
    for _p in rng_baseline.uniform(_lb, _ub, size=(n_baseline, len(_lb))):
        _, _y = objective(_p, num_analyses)
        _y_val = float(_y[0, 0])
        baseline_objs.append(_y_val)

    baseline_objs        = np.array(baseline_objs)
    baseline_running_min = np.minimum.accumulate(baseline_objs)

    _feasible_mask        = baseline_objs < 2.0
    best_feas_baseline    = float(baseline_objs[_feasible_mask].min()) if _feasible_mask.any() else float('nan')

    print(f"Random search: {n_baseline} evaluations")
    print(f"Feasible:      {_feasible_mask.sum()}/{n_baseline} ({100*_feasible_mask.mean():.1f}%)")
    print(f"Best overall:  {baseline_objs.min():.4f}")
    print(f"Best feasible: {best_feas_baseline:.4f}")
    return baseline_running_min, best_feas_baseline


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

    # best feasible point (obj < 2.0)
    _feasible_mask = _obs < 2.0
    if _feasible_mask.any():
        _feasible_obs = _obs.copy()
        _feasible_obs[~_feasible_mask] = np.inf
        best_idx    = int(np.argmin(_feasible_obs))
        best_params = final_dataset.query_points.numpy()[best_idx]
        best_obj    = float(_obs[best_idx])
        best_upper, best_lower = reverse_to_boundaries(best_params, num_analyses)
        best_n      = find_n_integer(best_upper, best_lower, target_power)

        print(f"Best objective value: {best_obj:.4f}")
        print(f"n per arm per stage:  {best_n}")
        print(f"Upper boundaries:     {np.round(best_upper, 4)}")
        print(f"Lower boundaries:     {np.round(best_lower, 4)}")
    else:
        print("No feasible designs found.")
        best_upper = best_lower = best_n = best_obj = None
    return best_lower, best_n, best_obj, best_upper, final_dataset


@app.cell
def _(
    best_feas_baseline,
    best_lower,
    best_n,
    best_obj,
    best_upper,
    delta1,
    num_analyses,
    sigma2,
    sim,
    target_alpha,
    target_power,
    tri_obj,
):
    if best_upper is not None:
        _trial = sim.group_sequential_designs(
            n_analyses=num_analyses, upper_bounds=best_upper, lower_bounds=best_lower,
            n_patients=best_n, null_hypothesis=0, alt_hypothesis=delta1, variance=sigma2
        )
        print(f"Achieved alpha':  {_trial[1]:.4f}  (target <= {target_alpha})")
        print(f"Achieved power':  {_trial[2]:.4f}  (target >= {target_power})")
        print()
        print(f"Deflated triangular benchmark: {tri_obj:.4f}")
        print(f"Random search best feasible:   {best_feas_baseline:.4f}")
        print(f"BO best feasible:              {best_obj:.4f}")
        print(f"BO improvement over random:    {best_feas_baseline - best_obj:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Objective function history
    """)
    return


@app.cell
def _(
    baseline_running_min,
    best_feas_baseline,
    final_dataset,
    np,
    plt,
    tri_obj,
):
    _obs         = final_dataset.observations.numpy().flatten()
    _feas_obs    = _obs[_obs < 2.0]
    _running_min = np.minimum.accumulate(_obs)

    _fig, _axes = plt.subplots(nrows=2, figsize=(10, 7))

    # top: raw BO history
    _axes[0].plot(_obs, alpha=0.6, color="steelblue", label="BO proposals")
    _axes[0].axhline(y=tri_obj, color="orange", linestyle="--",
                     label=f"Deflated triangular ({tri_obj:.4f})")
    _axes[0].set_ylim(0, 2.1)
    _axes[0].set_xlabel("Iteration")
    _axes[0].set_ylabel("Objective value")
    _axes[0].set_title("Objective history (clipped to y <= 2)")
    _axes[0].legend()

    # bottom: BO vs random search running minima
    _n = min(len(_running_min), len(baseline_running_min))
    _axes[1].plot(_running_min[:_n], color="steelblue", label="BO running minimum")
    _axes[1].plot(baseline_running_min[:_n], color="grey", linestyle="--",
                  label=f"Random search (best feasible: {best_feas_baseline:.4f})")
    _axes[1].axhline(y=tri_obj, color="orange", linestyle="--",
                     label=f"Deflated triangular ({tri_obj:.4f})")
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
def _(
    best_lower,
    best_upper,
    np,
    num_analyses,
    plt,
    reverse_to_boundaries,
    tri_params,
    warm_start_params,
):
    if best_upper is not None:
        _fig, _ax = plt.subplots(figsize=(8, 5))
        _stages = [1, 2, 3]

        # BO best design
        _ax.plot(_stages, best_upper, color="blue", linewidth=2, label="BO best (upper)")
        _ax.plot(_stages, np.concatenate((best_lower[:2], [best_upper[2]])),
                 color="blue", linewidth=2, linestyle="--", label="BO best (lower)")

        # triangular design boundaries (from computed tri_params)
        _tri_upper, _tri_lower = reverse_to_boundaries(tri_params, num_analyses)
        _ax.plot(_stages, _tri_upper, color="red", linewidth=2, label="Triangular (upper)")
        _ax.plot(_stages, np.concatenate((_tri_lower[:2], [_tri_upper[2]])),
                 color="red", linewidth=2, linestyle="--", label="Triangular (lower)")

        # warm start design (only if provided)
        if warm_start_params is not None:
            _ws_upper, _ws_lower = reverse_to_boundaries(warm_start_params, num_analyses)
            _ax.plot(_stages, _ws_upper, color="green", linewidth=2,
                     linestyle=":", label="Warm start (upper)")
            _ax.plot(_stages, np.concatenate((_ws_lower[:2], [_ws_upper[2]])),
                     color="green", linewidth=2, linestyle=(0, (3, 1, 1, 1)),
                     label="Warm start (lower)")

        _ax.set_xlabel("Stage")
        _ax.set_ylabel("Standardised boundary")
        _ax.set_title("Best BO design vs benchmarks")
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

    _df.to_csv("/tf/paul/simple_local_bo_results.csv", index=False)
    print(f"Saved {_df.shape[0]} rows.")
    print(f"Feasibility rate: {_df['feasible'].mean():.1%}")
    print(_df.describe())
    return


if __name__ == "__main__":
    app.run()
