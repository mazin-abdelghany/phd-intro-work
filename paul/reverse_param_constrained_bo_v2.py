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
    # Delta-Minimax Group Sequential Design via Constrained Bayesian Optimisation
    ## Reverse Parameterisation + Integer $n$ by Bisection + Failure Region ($\alpha'$ only)

    This notebook combines three ideas:

    1. **Reverse parameterisation** — structural constraints (monotonicity, meeting
    point) are satisfied automatically by construction. Every proposed design is
    structurally valid.

    2. **Integer $n$ by bisection** — rather than including $n$ as a BO parameter,
    for each proposed boundary set the smallest integer $n$ achieving the required
    power $1 - \beta^*$ is found by bisection. This guarantees $\beta' \leq \beta^*$
    by construction, reducing the BO to 5 parameters and leaving only the $\alpha'$
    constraint to be learned.

    3. **Constrained BO with failure region** — the $\alpha'$ constraint is handled
    by a VGP classifier. The acquisition function is $\text{EI} \times \text{PoV}$,
    where PoV is the probability that $\alpha' \leq \alpha^*$. The GPR is trained
    only on feasible designs (those where both constraints are satisfied).
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
    import tensorflow as tf
    import gpflow
    from gpflow.keras import tf_keras
    import trieste
    from trieste.space import Box
    from trieste.models.gpflow.models import (
        GaussianProcessRegression,
        VariationalGaussianProcess,
    )
    from trieste.models import TrainableProbabilisticModel
    from trieste.models.optimizer import BatchOptimizer
    from trieste.types import Tag
    from trieste.acquisition.rule import EfficientGlobalOptimization
    from trieste.acquisition import (
        SingleModelAcquisitionBuilder,
        ExpectedImprovement,
        Product,
    )

    return (
        BatchOptimizer,
        Box,
        EfficientGlobalOptimization,
        ExpectedImprovement,
        GaussianProcessRegression,
        Product,
        SingleModelAcquisitionBuilder,
        Tag,
        TrainableProbabilisticModel,
        VariationalGaussianProcess,
        gpflow,
        tf,
        tf_keras,
        trieste,
    )


@app.cell
def _():
    from py_group_sequential_designs import feasibility_penalty as fp
    from py_group_sequential_designs import function_to_minimize as fn_min
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

    The BO parameter vector is now 5-dimensional:
    $$\tilde{\theta} = (c,\, \Delta u_K, \Delta l_K, \ldots, \Delta u_2, \Delta l_2)$$

    $n$ is **not** a BO parameter — it is determined by integer bisection for each
    proposed boundary set.
    """)
    return


@app.cell
def _(np):
    def reverse_to_boundaries(params, K):
        """
        Convert 5-dim reverse-param vector (no n) to (upper_bounds, lower_bounds).
        params = [c, delta_u_K, delta_l_K, ..., delta_u_2, delta_l_2]
        """
        params  = np.asarray(params).flatten()
        c       = params[0]
        delta_u = params[1::2][::-1]   # delta_u_2, ..., delta_u_K
        delta_l = params[2::2][::-1]   # delta_l_2, ..., delta_l_K

        upper_bounds = np.array([c + np.sum(delta_u[k:]) for k in range(K)])
        lower_bounds = np.array([c - np.sum(delta_l[k:]) for k in range(K)])
        return upper_bounds, lower_bounds


    def boundaries_to_reverse(upper_bounds, lower_bounds):
        """
        Convert (upper_bounds, lower_bounds) to 5-dim reverse-param vector (no n).
        """
        upper_bounds = np.asarray(upper_bounds)
        lower_bounds = np.asarray(lower_bounds)
        K       = len(upper_bounds)
        c       = upper_bounds[-1]
        delta_u = np.diff(upper_bounds[::-1])
        delta_l = np.diff(lower_bounds)[::-1]
        increments       = np.empty(2 * (K - 1))
        increments[0::2] = delta_u
        increments[1::2] = delta_l
        return np.concatenate([[c], increments])

    return boundaries_to_reverse, reverse_to_boundaries


@app.cell
def _(boundaries_to_reverse, np, reverse_to_boundaries):
    # sanity check
    _upper = np.array([2.12, 1.87, 1.84])
    _lower = np.array([0.0,  1.12, 1.84])
    _params = boundaries_to_reverse(_upper, _lower)
    _u2, _l2 = reverse_to_boundaries(_params, K=3)
    print("Original upper:", _upper)
    print("Recovered upper:", np.round(_u2, 6))
    print("Original lower:", _lower)
    print("Recovered lower:", np.round(_l2, 6))
    print("Params:", np.round(_params, 6))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Integer $n$ by Bisection

    For a given boundary set, find the smallest integer $n$ per arm per stage
    such that power $\geq 1 - \beta^*$. Returns `None` if no $n \leq n_{\max}$
    achieves the required power — the design is then infeasible regardless of $n$.
    """)
    return


@app.cell
def _(delta1, num_analyses, sigma2, sim):
    def find_n_integer(upper_bounds, lower_bounds, target_power,
                       n_min=2, n_max=200):
        """
        Find smallest integer n per arm per stage achieving target power,
        using bisection over integers.

        Returns n (int) if achievable within [n_min, n_max], else None.
        """
        # check n_max is sufficient
        _, _, power_max, _ = sim.group_sequential_designs(
            n_analyses   = num_analyses,
            upper_bounds = upper_bounds,
            lower_bounds = lower_bounds,
            n_patients   = n_max,
            null_hypothesis = 0,
            alt_hypothesis  = delta1,
            variance        = sigma2
        )
        if power_max < target_power:
            return None   # boundaries too poor — no feasible n

        # bisect over integers
        while n_max - n_min > 1:
            n_mid = (n_min + n_max) // 2
            _, _, power_mid, _ = sim.group_sequential_designs(
                n_analyses   = num_analyses,
                upper_bounds = upper_bounds,
                lower_bounds = lower_bounds,
                n_patients   = n_mid,
                null_hypothesis = 0,
                alt_hypothesis  = delta1,
                variance        = sigma2
            )
            if power_mid >= target_power:
                n_max = n_mid
            else:
                n_min = n_mid

        return int(n_max)   # smallest integer n achieving target power

    return (find_n_integer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Objective Function

    For each proposed boundary set $\tilde{\theta} = (c, \Delta u_K, \Delta l_K, \ldots)$:

    1. Convert to $(u_k, l_k)$ via reverse parameterisation
    2. Find integer $n$ by bisection — if none exists, design is infeasible
    3. Compute $\alpha'$ — if $\alpha' > \alpha^*$, design is infeasible
    4. If feasible, compute $\max_\delta E[N \mid \delta] / \mu$ as objective

    Returns $(x, y_{\text{obj}}, \text{feasible})$.
    $\beta' \leq \beta^*$ is **guaranteed by construction** via step 2.
    Only $\alpha'$ needs the VGP.
    """)
    return


@app.cell
def _(delta1, find_n_integer, np, reverse_to_boundaries, sigma2, sim, ss):
    def objective(params, mu, target_power, target_alpha, K):
        params = np.asarray(params).flatten()
        upper_bounds, lower_bounds = reverse_to_boundaries(params, K)

        # step 1: find integer n achieving target power
        n = find_n_integer(
            upper_bounds = upper_bounds,
            lower_bounds = lower_bounds,
            target_power = target_power
        )

        if n is None:
            # no feasible n exists for these boundaries
            return (
                np.array([params]),
                np.array([[np.nan]]),
                np.array([[0.0]])
            )

        # step 2: evaluate alpha' at this n
        trial = sim.group_sequential_designs(
            n_analyses      = K,
            upper_bounds    = upper_bounds,
            lower_bounds    = lower_bounds,
            n_patients      = n,
            null_hypothesis = 0,
            alt_hypothesis  = delta1,
            variance        = sigma2
        )
        alpha_prime = trial[1]

        # step 3: feasibility — only alpha' needs checking
        # (beta' guaranteed by bisection)
        feasible = int(alpha_prime <= target_alpha)

        if not feasible:
            return (
                np.array([params]),
                np.array([[np.nan]]),
                np.array([[0.0]])
            )

        # step 4: compute objective
        max_ess = ss.max_ess(
            n_analyses      = K,
            upper_bounds    = upper_bounds,
            lower_bounds    = lower_bounds,
            n_patients      = n,
            null_hypothesis = 0,
            variance        = sigma2
        )

        y_obj = max_ess / mu   # no penalty needed: both constraints satisfied

        return (
            np.array([params]),
            np.array([[y_obj]]),
            np.array([[1.0]])
        )

    return (objective,)


@app.cell
def _(
    bd,
    boundaries_to_reverse,
    delta1,
    mu,
    np,
    num_analyses,
    objective,
    target_alpha,
    target_power,
):
    _tri = bd.calculate_triangular_boundaries(
        n_analyses=num_analyses, alpha=target_alpha, delta=delta1, n_patients=20
    )
    tri_params = boundaries_to_reverse(_tri[0], _tri[1])
    c0         = tri_params[0]   # first element is c in 5-dim vector

    _, _y, _feas = objective(tri_params, mu, target_power, target_alpha, num_analyses)
    print(f"Triangular benchmark — objective: {_y[0,0]:.4f}, feasible: {bool(_feas[0,0])}")
    print(f"Triangular params (5-dim): {np.round(tri_params, 4)}")
    print(f"Meeting point c0 = {c0:.4f}")
    return c0, tri_params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Search Space

    5-dimensional: $(c, \Delta u_3, \Delta l_3, \Delta u_2, \Delta l_2)$.
    $n$ is no longer a BO parameter.
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
    print(f"\nTriangular params for reference: {np.round(tri_params, 4)}")
    return (search_space,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Initialisation

    - **Perturbations** around Pocock, O'Brien-Fleming, and triangular designs
    - **Random feasible points** from pool sampled within search space

    All points guaranteed to have $\beta' \leq \beta^*$ (via bisection).
    Only $\alpha'$ determines feasibility label.
    """)
    return


@app.cell
def _(
    bd,
    boundaries_to_reverse,
    mu,
    np,
    num_analyses,
    objective,
    search_space,
    target_alpha,
    target_power,
    tri_params,
    use_local_search,
):
    rng_init = np.random.default_rng(seed=42)

    # reference designs in 5-dim reverse parameterisation
    _poc = bd.calculate_pocock_boundaries(
        n_analyses=num_analyses, alpha=target_alpha, n_patients=20
    )
    poc_params = boundaries_to_reverse(_poc[0], _poc[1])

    _obf = bd.calculate_of_boundaries(
        n_analyses=num_analyses, alpha=target_alpha, n_patients=20
    )
    obf_params = boundaries_to_reverse(_obf[0], _obf[1])

    # part 1: perturbations around known designs
    # in local search, only perturb around triangular (assumed near-optimal)
    # in global search, also include Pocock and O'Brien-Fleming for coverage
    _refs_to_perturb = [tri_params] if use_local_search.value else [poc_params, obf_params, tri_params]
    _n_perturb       = 15 if use_local_search.value else 5   # more perturbations in local case

    all_x, all_y_obj, all_feas = [], [], []
    for _ref in _refs_to_perturb:
        for _ in range(_n_perturb):
            _scale = np.clip(np.abs(_ref) * 0.10, 0.01, 0.3)
            _p     = _ref + rng_init.normal(scale=_scale, size=len(_ref))
            _p[1:] = np.clip(_p[1:], 0.0, 4.0)
            _x, _y, _f = objective(_p, mu, target_power, target_alpha, num_analyses)
            all_x.append(_x); all_y_obj.append(_y); all_feas.append(_f)

    n_feas_perturb = sum(int(_f[0,0]) for _f in all_feas)
    print(f"Perturbation points: {len(all_x)} total, {n_feas_perturb} feasible")

    # part 2: random feasible points from pool within search space
    n_target = 15
    n_found  = 0
    n_tried  = 0
    _lb = search_space.lower.numpy()
    _ub = search_space.upper.numpy()
    _candidates = rng_init.uniform(_lb, _ub, size=(1000, len(_lb)))

    for _p in _candidates:
        _x, _y, _f = objective(_p, mu, target_power, target_alpha, num_analyses)
        all_x.append(_x); all_y_obj.append(_y); all_feas.append(_f)
        n_tried += 1
        if int(_f[0, 0]) == 1:
            n_found += 1
        if n_found >= n_target:
            break

    print(f"Random feasible points: {n_found} found from {n_tried} candidates")
    print(f"Total initial points: {len(all_x)}")
    return all_feas, all_x, all_y_obj


@app.cell
def _(all_feas, all_x, all_y_obj, np, trieste):
    _all_x_arr    = np.concatenate(all_x)
    _all_feas_arr = np.concatenate(all_feas)
    _feas_mask    = _all_feas_arr.flatten() == 1

    _obj_x = _all_x_arr[_feas_mask]
    _obj_y = np.concatenate(all_y_obj)[_feas_mask]

    initial_data = {
        "objective": trieste.data.Dataset(
            query_points = _obj_x,
            observations = _obj_y
        ),
        "failure": trieste.data.Dataset(
            query_points = _all_x_arr,
            observations = _all_feas_arr
        )
    }

    print(f"Objective dataset: {_obj_x.shape[0]} feasible points")
    print(f"Failure dataset:   {_all_x_arr.shape[0]} total points")
    print(f"Objective values:  {[round(float(v), 4) for v in _obj_y.flatten()]}")
    return (initial_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Models

    - **GPR** (Matern52, ARD) on feasible designs only — learns objective landscape
    - **VGP** (Squared Exponential, Bernoulli) on all designs — learns $\alpha'$ feasibility
    """)
    return


@app.cell
def _(
    BatchOptimizer,
    GaussianProcessRegression,
    Tag,
    TrainableProbabilisticModel,
    VariationalGaussianProcess,
    gpflow,
    initial_data,
    tf_keras,
):
    _n_dims = initial_data["objective"].query_points.shape[1]

    _gpr_kernel = gpflow.kernels.Matern52(lengthscales=[1.0] * _n_dims)
    _gpr = gpflow.models.GPR(
        data=(
            initial_data["objective"].query_points,
            initial_data["objective"].observations
        ),
        kernel=_gpr_kernel,
        likelihood=gpflow.likelihoods.Gaussian()
    )

    _vgp_kernel = gpflow.kernels.SquaredExponential(lengthscales=[1.0] * _n_dims)
    _vgp = gpflow.models.VGP(
        data=(
            initial_data["failure"].query_points,
            initial_data["failure"].observations
        ),
        kernel=_vgp_kernel,
        likelihood=gpflow.likelihoods.Bernoulli()
    )

    models: dict[Tag, TrainableProbabilisticModel] = {
        "objective": GaussianProcessRegression(_gpr),
        "failure":   VariationalGaussianProcess(
            _vgp,
            optimizer=BatchOptimizer(tf_keras.optimizers.Adam(learning_rate=1e-3)),
            use_natgrads=True
        )
    }
    return (models,)


@app.cell
def _(
    EfficientGlobalOptimization,
    ExpectedImprovement,
    Product,
    SingleModelAcquisitionBuilder,
    tf,
    trieste,
):
    class ProbabilityOfValidity(SingleModelAcquisitionBuilder):
        def prepare_acquisition_function(self, model, dataset=None):
            def acquisition(at):
                mean, _ = model.predict_y(tf.squeeze(at, -2))
                return mean
            return acquisition

    ei     = ExpectedImprovement()
    pov    = ProbabilityOfValidity()
    acq_fn = Product(ei.using("objective"), pov.using("failure"))

    rule = EfficientGlobalOptimization(
        acq_fn,
        optimizer=trieste.acquisition.optimizer.generate_continuous_optimizer(
            num_optimization_runs=500
        )
    )
    return (rule,)


@app.cell
def _(
    initial_data,
    models: "dict[Tag, TrainableProbabilisticModel]",
    rule,
    search_space,
    trieste,
):
    ask_tell = trieste.ask_tell_optimization.AskTellOptimizer(
        search_space     = search_space,
        datasets         = initial_data,
        models           = models,
        acquisition_rule = rule
    )
    return (ask_tell,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bayesian Optimisation Loop

    At each iteration:
    1. **Ask** — propose boundary set $(c, \Delta u_3, \Delta l_3, \Delta u_2, \Delta l_2)$
    2. **Bisect** — find integer $n$ achieving $\beta' \leq \beta^*$
    3. **Check** $\alpha'$ — determines feasibility label
    4. **Tell** — update failure dataset always; objective dataset only if feasible
    """)
    return


@app.cell
def _(
    ask_tell,
    mu,
    np,
    num_analyses,
    objective,
    target_alpha,
    target_power,
    trieste,
):
    num_repeats   = 500
    when_to_print = 50
    n_feasible_bo = 0

    for _i in range(num_repeats):
        x_new = ask_tell.ask()

        _x, _y_obj, _feas = objective(
            params       = x_new,
            mu           = mu,
            target_power = target_power,
            target_alpha = target_alpha,
            K            = num_analyses
        )

        _is_feasible = int(_feas[0, 0]) == 1
        if _is_feasible:
            n_feasible_bo += 1

        new_data = {
            "failure": trieste.data.Dataset(
                query_points = _x,
                observations = _feas
            )
        }

        if _is_feasible:
            new_data["objective"] = trieste.data.Dataset(
                query_points = _x,
                observations = _y_obj
            )
        else:
            new_data["objective"] = trieste.data.Dataset(
                query_points = np.reshape(np.array([]), (0, _x.shape[1])),
                observations = np.reshape(np.array([]), (0, 1))
            )

        ask_tell.tell(new_data=new_data)

        if (_i + 1) % when_to_print == 0:
            print(f"\nLoop {_i+1} completed. "
                  f"Feasible so far: {n_feasible_bo}/{_i+1}", end="")
        elif (_i > when_to_print) and ((_i + 1) % 5 == 0):
            print(".", end="")

    print(f"\nDone. Feasible BO proposals: {n_feasible_bo}/{num_repeats}")
    return


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
    final_obj_dataset = ask_tell.to_result().try_get_final_datasets()["objective"]

    best_idx    = int(np.argmin(final_obj_dataset.observations.numpy()))
    best_params = final_obj_dataset.query_points.numpy()[best_idx]
    best_obj    = float(final_obj_dataset.observations.numpy()[best_idx])

    best_upper, best_lower = reverse_to_boundaries(best_params, num_analyses)
    best_n = find_n_integer(best_upper, best_lower, target_power)

    print(f"Best objective value: {best_obj:.4f}")
    print(f"n per arm per stage:  {best_n}")
    print(f"Upper boundaries:     {np.round(best_upper, 4)}")
    print(f"Lower boundaries:     {np.round(best_lower, 4)}")
    return best_lower, best_n, best_upper, final_obj_dataset


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
):
    _trial = sim.group_sequential_designs(
        n_analyses=num_analyses, upper_bounds=best_upper, lower_bounds=best_lower,
        n_patients=best_n, null_hypothesis=0, alt_hypothesis=delta1, variance=sigma2
    )
    print(f"Achieved alpha':  {_trial[1]:.4f}  (target <= {target_alpha})")
    print(f"Achieved power':  {_trial[2]:.4f}  (target >= {target_power})")
    print()
    print("Triangular benchmark:")
    print(f"  Upper: [2.1196, 1.8735, 1.8356]")
    print(f"  Lower: [0.0000, 1.1241, 1.8356]")
    print(f"  n: 22,  objective: 0.2715")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Objective function history (feasible designs only)
    """)
    return


@app.cell
def _(final_obj_dataset, np, plt):
    _obs         = final_obj_dataset.observations.numpy().flatten()
    _running_min = np.minimum.accumulate(_obs)

    _fig, _axes = plt.subplots(nrows=2, figsize=(10, 7))

    _axes[0].plot(_obs, alpha=0.6, color="steelblue")
    _axes[0].axhline(y=0.271, color="red", linestyle="--",
                     label="Triangular benchmark (0.271)")
    _axes[0].set_xlabel("Feasible iteration")
    _axes[0].set_ylabel("Objective value")
    _axes[0].set_title("Objective history (feasible designs only)")
    _axes[0].legend()

    _axes[1].plot(_running_min, color="steelblue")
    _axes[1].axhline(y=0.271, color="red", linestyle="--",
                     label="Triangular benchmark (0.271)")
    _axes[1].set_xlabel("Feasible iteration")
    _axes[1].set_ylabel("Best objective so far")
    _axes[1].set_title("Running minimum")
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
    _datasets = ask_tell.to_result().try_get_final_datasets()

    _obj = _datasets["objective"]
    _obj_df = pd.DataFrame(
        data    = _obj.query_points.numpy(),
        columns = ["c", "delta_u3", "delta_l3", "delta_u2", "delta_l2"]
    )
    _obj_df["obj_f"] = _obj.observations.numpy()
    _obj_df.to_csv("/tf/paul/constrained_bo_v2_objective.csv", index=False)

    _fail = _datasets["failure"]
    _fail_df = pd.DataFrame(
        data    = _fail.query_points.numpy(),
        columns = ["c", "delta_u3", "delta_l3", "delta_u2", "delta_l2"]
    )
    _fail_df["feasible"] = _fail.observations.numpy().astype(int)
    _fail_df.to_csv("/tf/paul/constrained_bo_v2_failure.csv", index=False)

    print(f"Saved {_obj_df.shape[0]} feasible designs to constrained_bo_v2_objective.csv")
    print(f"Saved {_fail_df.shape[0]} total designs to constrained_bo_v2_failure.csv")
    print(f"Feasibility rate: {_fail_df['feasible'].mean():.1%}")
    return


if __name__ == "__main__":
    app.run()
