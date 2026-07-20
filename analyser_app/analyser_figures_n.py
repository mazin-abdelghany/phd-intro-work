import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    return mlines, np, pd, plt


@app.cell
def _(plt):
    plt.rcParams["figure.dpi"] = 500
    return


@app.cell
def _():
    # group sequential design assessment imports
    from py_group_sequential_designs import generate_boundaries as bd
    from py_group_sequential_designs import simulate as sim
    from py_group_sequential_designs import sample_size as ss
    from py_group_sequential_designs import feasibility_penalty as fp
    from py_group_sequential_designs import function_to_minimize as fn_min

    return bd, fn_min, fp, sim, ss


@app.cell
def _(ss):
    ##############
    # Setup cell #
    ##############

    # design settings
    num_analyses = 3
    target_alpha = 0.05
    target_power = 0.9
    delta0 = 0.
    delta1 = 1.
    sigma2 = 9.

    mu = ss.sample_size_means(
        ratio=1,
        variance=sigma2,
        power=target_power,
        alpha=target_alpha,
        delta=delta1
    )
    return delta0, delta1, mu, num_analyses, sigma2, target_alpha, target_power


@app.cell
def _(fn_min, fp, sim, ss):
    def obj_f(
            mu,
            upper_bounds,
            lower_bounds,
            n_patients,
            n_analyses,
            target_power,
            target_alpha,
            null_hypothesis,
            alternative_hypothesis,
            variance):

        trial_sim = sim.group_sequential_designs(
            n_analyses = n_analyses,
            upper_bounds = upper_bounds,
            lower_bounds = lower_bounds,
            n_patients = n_patients, 
            null_hypothesis = null_hypothesis,
            alt_hypothesis = alternative_hypothesis,
            variance = variance
        )

        alpha_prime = trial_sim[0]
        beta_prime = 1-trial_sim[1]

        max_ess = ss.max_ess(
            n_analyses = n_analyses,
            upper_bounds = upper_bounds,
            lower_bounds = lower_bounds,
            n_patients = n_patients,
            null_hypothesis = null_hypothesis,
            variance = variance
        )

        penalty = fp.smooth_penalty(
            mu = mu,
            power = target_power,
            alpha = target_alpha,
            beta_prime = beta_prime,
            alpha_prime = alpha_prime
        )

        f_val = fn_min.function_to_minimize(max_ess_val = max_ess/mu, penalty = penalty)

        return (
            alpha_prime,
            1-beta_prime,
            max_ess,
            f_val
        )

    return (obj_f,)


@app.cell
def _(
    bd,
    delta0,
    delta1,
    mu,
    num_analyses,
    obj_f,
    sigma2,
    ss,
    target_alpha,
    target_power,
):
    tri = bd.calculate_triangular_boundaries(
        n_analyses = num_analyses,
        alpha = target_alpha
    )

    tri_n_patients = ss.find_sample_size(
        power_target = target_power,
        n_analyses = num_analyses,
        upper_bounds = tri[0],
        lower_bounds = tri[1],
        null_hypothesis = delta0,
        alt_hypothesis = delta1,
        variance = sigma2
    )[0]

    tri_alpha, tri_power, tri_max_ess, tri_obj = obj_f(
        mu = mu,
        upper_bounds = tri[0],
        lower_bounds = tri[1],
        n_analyses = num_analyses,
        n_patients = tri_n_patients,
        target_power = target_power,
        target_alpha = target_alpha,
        null_hypothesis = delta0,
        alternative_hypothesis = delta1,
        variance = sigma2
    )
    return (tri,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Experimental Setup
    """)
    return


@app.cell
def _(mo):
    n_experiments = mo.ui.number(label="No. experiments = ", value=1)
    n_loops = mo.ui.number(label="Number of loops = ", value=1)
    num_methods = mo.ui.number(label="Number of methods/files to compare = ", value=2, start=1)

    mo.vstack([n_experiments, n_loops, num_methods])
    return n_experiments, n_loops, num_methods


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Files & Dynamic Selection
    """)
    return


@app.cell
def _(mo, num_methods):
    labels = mo.ui.array([
        mo.ui.text(placeholder=f"Method_{i+1}", label=f"Method {i+1} Label: ") 
        for i in range(num_methods.value)
    ])

    file_browser = mo.ui.file_browser(
        initial_path = "/workspace/experiments_rand_simann_bo/",
        label = "Select files in the order of the methods."
    )

    mo.vstack([labels, file_browser])
    return file_browser, labels


@app.cell
def _(file_browser, labels, pd):
    datasets = {}
    for _i, label_ui in enumerate(labels.elements):
        label_value = label_ui.value if label_ui.value else f"Method_{_i+1}"
        try:
            filepath = file_browser.path(index=_i)
            if filepath:
                _df = pd.read_csv(filepath)
                datasets[label_value] = _df.iloc[:, 1:]
        except IndexError:
            pass
    return (datasets,)


@app.cell
def _(n_experiments, n_loops, np):
    runs = np.concatenate([np.repeat(f"Run_{i+1}", n_loops.value) for i in range(n_experiments.value)])
    return (runs,)


@app.cell
def _(datasets, runs):
    for label, df in datasets.items():
        if len(df) == len(runs):
            df["runs"] = runs
    return


@app.cell
def _(mo):
    seed_for_runs = mo.ui.number(label="RNG seed: ", value=42)
    return (seed_for_runs,)


@app.cell
def _(mo, seed_for_runs):
    mo.vstack([seed_for_runs])
    return


@app.cell
def _(np, seed_for_runs):
    rng = np.random.default_rng(seed=seed_for_runs.value)
    return (rng,)


@app.cell
def _(n_experiments, rng):
    runs_to_compare = rng.integers(low=1, high=n_experiments.value, size=6)
    return (runs_to_compare,)


@app.cell
def _(mo, runs_to_compare):
    mo.Html(f"We will be summarizing runs: {runs_to_compare}")
    return


@app.cell
def _(runs_to_compare):
    column_runs_compare = ["Run_"+str(run) for run in runs_to_compare]
    return (column_runs_compare,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Distribution of characteristics
    """)
    return


@app.cell
def _(datasets, np, plt):
    plot_cols = ["alpha", "power", "sample_size", "max_ess", "obj_func"]
    num_cols = len(datasets)

    fig, ax = plt.subplots(
        nrows=len(plot_cols), 
        ncols=num_cols, 
        figsize=(4 * num_cols, 11), 
        sharey="row", 
        squeeze=False
    )

    for _i, col in enumerate(plot_cols):
        for j, (_label, data) in enumerate(datasets.items()):
            ax[_i, j].violinplot(
                data[col],
                showextrema=False,
                showmedians=True
            )

            ax[_i, j].text(
                0.5,
                0.7,
                np.round(np.median(data[col]), 3),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax[_i, j].transAxes
            )

    # Set labels dynamic
    for j, _label in enumerate(datasets.keys()):
        ax[0, j].set_title(_label)

    ax[0, 0].set_ylabel("$\\alpha'$")
    ax[1, 0].set_ylabel("$1-\\beta'$")
    ax[2, 0].set_ylabel("Sample size")
    ax[3, 0].set_ylabel("Max ESS")
    ax[4, 0].set_ylabel("Loss")

    for a in ax.flat:
        a.set_xticks([])

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Boundary comparisons
    """)
    return


@app.cell
def _(num_analyses):
    stages = [i+1 for i in range(num_analyses)]
    return (stages,)


@app.cell
def _(num_analyses):
    lower_boundary_value_labels = ["lower" + f"{i+1}" for i in range(num_analyses-1)] + ["upper" + f"{num_analyses}"]
    return (lower_boundary_value_labels,)


@app.cell
def _(num_analyses):
    upper_boundary_value_labels = ["upper" + f"{i+1}" for i in range(num_analyses)]
    return (upper_boundary_value_labels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Best boundary
    """)
    return


@app.cell
def _(lower_boundary_value_labels, np, upper_boundary_value_labels):
    def best_bound_getter(data):
        min_idx = data["obj_func"].idxmin()
        lower = data.loc[min_idx, upper_boundary_value_labels].tolist()
        upper = data.loc[min_idx, lower_boundary_value_labels].tolist()
        obj_f = np.round(data.loc[min_idx, "obj_func"], decimals=4)
        alpha = np.round(data.loc[min_idx, "alpha"], decimals=4)
        power = np.round(data.loc[min_idx, "power"], decimals=4)
        return lower, upper, obj_f, alpha, power

    return (best_bound_getter,)


@app.cell
def _(best_bound_getter, datasets, plt, stages, tri):
    _num_plots = len(datasets)
    _fig, _ax = plt.subplots(1, _num_plots, figsize=(6 * _num_plots, 3.5), sharey=True, squeeze=False)
    _ax = _ax.flatten()

    for _idx, (_label, _data) in enumerate(datasets.items()):
        _upper, _lower, _obj_f, _alpha, _power = best_bound_getter(_data)
        _b = _ax[_idx]

        _b.set_title(_label)
        _b.set(xlabel="Trial stages", xticks=[1, 2, 3])

        _b.plot(stages, tri[0], color="darkorange", label="Tri bound")
        _b.plot(stages, tri[1], color="darkorange")

        _b.plot(stages, _upper, color="purple", label="Best bound")
        _b.plot(stages, _lower, color="purple")

        for _y, _txt in zip(
            [0.96, 0.88, 0.8],
            [f"$\\mathcal{{L}}$ = {_obj_f}", f"$\\alpha$ = {_alpha}", f"$1-\\beta$ = {_power}"]
        ):
            _b.text(0.98, _y, _txt, ha="right", va="top", transform=_b.transAxes)

        _b.legend(loc="lower right")

    _ax[0].set_ylabel("$Z_k$ values")
    _fig.suptitle("Best boundary", y=0.96)
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Best constrained boundary
    """)
    return


@app.cell
def _(lower_boundary_value_labels, np, upper_boundary_value_labels):
    def best_constrained_bound_getter(data):
        constrained_data = data[data["alpha"] <= 0.05]
        if constrained_data.empty:
            constrained_data = data # fallback
        min_idx = constrained_data["obj_func"].idxmin()
        lower = constrained_data.loc[min_idx, upper_boundary_value_labels].tolist()
        upper = constrained_data.loc[min_idx, lower_boundary_value_labels].tolist()
        obj_f = np.round(constrained_data.loc[min_idx, "obj_func"], decimals=4)
        alpha = np.round(constrained_data.loc[min_idx, "alpha"], decimals=4)
        power = np.round(constrained_data.loc[min_idx, "power"], decimals=4)
        return lower, upper, obj_f, alpha, power

    return (best_constrained_bound_getter,)


@app.cell
def _(best_constrained_bound_getter, datasets, plt, stages, tri):
    _num_plots = len(datasets)
    _fig, _ax = plt.subplots(1, _num_plots, figsize=(6 * _num_plots, 3.5), sharey=True, squeeze=False)
    _ax = _ax.flatten()

    for _idx, (_label, _data) in enumerate(datasets.items()):
        _upper, _lower, _obj_f, _alpha, _power = best_constrained_bound_getter(_data)
        _b = _ax[_idx]

        _b.set_title(_label)
        _b.set(xlabel="Trial stages", xticks=[1, 2, 3])

        _b.plot(stages, tri[0], color="darkorange", label="Tri bound")
        _b.plot(stages, tri[1], color="darkorange")

        _b.plot(stages, _upper, color="purple", label="Best bound")
        _b.plot(stages, _lower, color="purple")

        for _y, _txt in zip(
            [0.96, 0.88, 0.8],
            [f"$\\mathcal{{L}}$ = {_obj_f}", f"$\\alpha$ = {_alpha}", f"$1-\\beta$ = {_power}"]
        ):
            _b.text(0.98, _y, _txt, ha="right", va="top", transform=_b.transAxes)

        _b.legend(loc="lower right")

    _ax[0].set_ylabel("$Z_k$ values")
    _fig.suptitle("Best constrained boundary", y=0.96)
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random run assessment
    """)
    return


@app.cell
def _(lower_boundary_value_labels, np, upper_boundary_value_labels):
    def rand_bound_getter(data, run):
        run_data = data[data["runs"] == run]
        if run_data.empty:
            return [0,0,0], [0,0,0], 0, 0, 0
        min_idx = run_data["obj_func"].idxmin()
        lower = run_data.loc[min_idx, upper_boundary_value_labels].tolist()
        upper = run_data.loc[min_idx, lower_boundary_value_labels].tolist()
        obj_f = np.round(run_data.loc[min_idx, "obj_func"], decimals=4)
        alpha = np.round(run_data.loc[min_idx, "alpha"], decimals=4)
        power = np.round(run_data.loc[min_idx, "power"], decimals=4)
        return lower, upper, obj_f, alpha, power

    return (rand_bound_getter,)


@app.cell
def _(column_runs_compare, datasets, mo, plt, rand_bound_getter, stages, tri):
    # Dynamically loops through files and builds a grid of random selected run plots
    if not datasets or len(column_runs_compare) < 2:
        mo.md("")

    _num_plots = len(datasets)
    _fig, _ax = plt.subplots(1, _num_plots, figsize=(6 * _num_plots, 3.5), sharey=True, squeeze=False)
    _ax = _ax.flatten()

    for _idx, (_label, _data) in enumerate(datasets.items()):
        # Assign run dynamically corresponding to the index item position
        run_str = column_runs_compare[_idx % len(column_runs_compare)]
        _upper, _lower, _obj_f, _alpha, _power = rand_bound_getter(_data, run_str)
        _b = _ax[_idx]

        _b.set_title(f"{_label} ({run_str})")
        _b.set(xlabel="Trial stages", xticks=[1, 2, 3])

        _b.plot(stages, tri[0], color="darkorange", label="Tri bound")
        _b.plot(stages, tri[1], color="darkorange")

        _b.plot(stages, _upper, color="purple", label="Best bound")
        _b.plot(stages, _lower, color="purple")

        for y, txt in zip(
            [0.96, 0.88, 0.8],
            [f"$\\mathcal{{L}}$ = {_obj_f}", f"$\\alpha$ = {_alpha}", f"$1-\\beta$ = {_power}"]
        ):
            _b.text(0.98, y, txt, ha="right", va="top", transform=_b.transAxes)

        _b.legend(loc="lower right")

    _ax[0].set_ylabel("$Z_k$ values")
    _fig.suptitle("Best boundary -- Random runs overview", y=0.96)
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Top constrained boundaries
    """)
    return


@app.cell
def _(
    best_constrained_bound_getter,
    column_runs_compare,
    datasets,
    mlines,
    mo,
    np,
    plt,
    runs,
    stages,
    tri,
):
    n_over_5 = 0

    if not datasets or len(column_runs_compare) < 2:
        mo.md("")

    _num_plots = len(datasets)
    _fig, _ax = plt.subplots(1, _num_plots, figsize=(6 * _num_plots, 3.5), sharey=True, squeeze=False)
    _ax = _ax.flatten()

    for _idx, (_label, _data) in enumerate(datasets.items()):
        for _run_idx, _run in enumerate(np.unique(runs)):
            run_data = _data[_data["runs"] == _run]
            _upper, _lower, _obj_f, _alpha, _power = best_constrained_bound_getter(run_data)
        
            # count number of first upper bounds over 5
            if _upper[0] >= 5: n_over_5 +=1
        
            _b = _ax[_idx]

            _b.set_title(_label)
            _b.set(xlabel="Trial stages", xticks=[1, 2, 3])

            _b.plot(stages, _upper, color="purple", alpha = 0.15)
            _b.plot(stages, _lower, color="purple", alpha = 0.15)

        # draw triangular bounds
        _b.plot(stages, tri[0], color="darkorange", zorder = 3)
        _b.plot(stages, tri[1], color="darkorange", zorder = 3)

        # custom legend
        best_bound_label = mlines.Line2D([], [], color='purple', label = "Best bounds")
        tri_bound_label = mlines.Line2D([], [], color ="darkorange", label = "Tri bounds")
        _b.legend(handles = [best_bound_label, tri_bound_label], loc="lower right")

    _ax[0].set_ylabel("$Z_k$ values")
    _fig.suptitle(f"Top {len(np.unique(runs))} constrained boundaries", y=0.96)
    plt.tight_layout()
    plt.gca()
    return (n_over_5,)


@app.cell(hide_code=True)
def _(mo, n_over_5):
    mo.Html(f"There are {n_over_5} first upper bounds that are greater than 5.")
    return


@app.cell
def _(
    datasets,
    lower_boundary_value_labels,
    np,
    num_analyses,
    plt,
    runs,
    upper_boundary_value_labels,
):
    _fig, _ax = plt.subplots(nrows=2, ncols=num_analyses, figsize=(11,4), sharey=True)

    _upper = {key: [] for key in upper_boundary_value_labels}
    _lower = {key: [] for key in lower_boundary_value_labels}

    for _idx, (_label, _data) in enumerate(datasets.items()):
        for _run_idx, _run in enumerate(np.unique(runs)):
            _run_data = _data[_data["runs"] == _run]

            _constrained_data = _run_data[_run_data["alpha"] <= 0.05]
        
            _min_idx = _constrained_data["obj_func"].idxmin()

            # get the upper bound values
            for _key in _upper:
                _upper[_key].append(_constrained_data.loc[_min_idx, _key])

            # get the lower bound values
            for _key in _lower:
                _lower[_key].append(_constrained_data.loc[_min_idx, _key])
        
        for colu, key in enumerate(upper_boundary_value_labels):
            _ax[0, colu].ecdf(_upper[key], label=_label)
            _ax[0, colu].set_title(key)
            _ax[0, colu].legend()
    
        for colu, key in enumerate(lower_boundary_value_labels):
            _ax[1, colu].ecdf(_lower[key], label=_label)
            _ax[1, colu].set_title(key)
            _ax[1, colu].legend()

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
