import marimo

__generated_with = "0.23.14"
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
    import matplotlib as mpl

    return mlines, mpl, np, pd, plt


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Experimental Setup
    """)
    return


@app.cell
def _(mo):
    num_analyses = mo.ui.number(label="Number of analyses = ", value=3, start=1)

    mo.vstack([num_analyses])
    return (num_analyses,)


@app.cell
def _(ss):
    ##############
    # Setup cell #
    ##############

    # design settings
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
    return delta0, delta1, mu, sigma2, target_alpha, target_power


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
        n_analyses = num_analyses.value,
        alpha = target_alpha
    )

    tri_n_patients = ss.find_sample_size(
        power_target = target_power,
        n_analyses = num_analyses.value,
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
        n_analyses = num_analyses.value,
        n_patients = tri_n_patients,
        target_power = target_power,
        target_alpha = target_alpha,
        null_hypothesis = delta0,
        alternative_hypothesis = delta1,
        variance = sigma2
    )
    return (tri,)


@app.cell
def _(mo):
    num_methods = mo.ui.number(label="Number of methods/files to compare = ", value=2, start=1)

    mo.vstack([num_methods])
    return (num_methods,)


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
        initial_path = "/tf/experiments_rand_simann_bo/",
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
def _(mo):
    wason_included = mo.ui.switch(label="Wason data included")
    return (wason_included,)


@app.cell
def _(mo, wason_included):
    mo.vstack([wason_included, mo.md(f"Has value: {wason_included.value}")])
    return


@app.cell
def _(datasets, np, wason_included):
    if wason_included.value:
        n_experiments = 10
        n_loops = 10000

        # generate labels for data frame
        # if we have experiments <1000, then we need 3 spaces 
        # to fill. this is the length of the string and then
        # we raise 10 to this number to obtain the labels
        space_needed_for_label = len(str(n_loops))
        label_range = 10**(space_needed_for_label)

        index_wason = [
            i 
            for start in range(label_range, (n_experiments+1)*label_range, label_range) 
            for i in range(start + 1, start + (n_loops+1))
        ]


        datasets["Wason"]["index"] = np.array(index_wason)
    return


@app.function
def parse_index(index):
    s = str(index)

    # First try a 2-digit experiment (10-99)
    if len(s) >= 5:
        exp = int(s[:2])
        run = int(s[2:])
        if 10 <= exp <= 99 and 100 <= run <= 9999:
            return exp, run

    # Otherwise it must be a 1-digit experiment (1-9)
    exp = int(s[:1])
    run = int(s[1:])
    if 1 <= exp <= 9 and 100 <= run <= 9999:
        return exp, run

    raise ValueError("Invalid index")


@app.cell
def _(datasets, np):
    n_experiments_dict = dict()
    n_loops_dict = dict()

    # the last index contains the total number of experiments and loops
    # e.g., 10500 = 10 experiments, 500 loops
    # !! this would if there were >99 experiments !!
    for _idx, (_label, _data) in enumerate(datasets.items()):
        _experiment, _run = parse_index(  _data.iloc[-1]["index"].astype(np.int_) )
        n_experiments_dict[_label] = _experiment
        n_loops_dict[_label] = _run
    return n_experiments_dict, n_loops_dict


@app.cell
def _(mo, n_experiments_dict):
    mo.Html("<br>".join(
        f"There are {n_experiments_dict[key]} experiments in {key}"
        for key in n_experiments_dict
    ))
    return


@app.cell
def _(n_experiments_dict, n_loops_dict, np):
    runs_dict = dict()

    for _key in n_experiments_dict:
        run_labels = [
            np.repeat(f"Run_{i + 1}", n_loops_dict[_key])
            for i in range(n_experiments_dict[_key])
        ]

        runs_dict[_key] = np.concatenate(run_labels)
    return (runs_dict,)


@app.cell
def _(datasets, runs_dict):
    for label, df in datasets.items():
        df["runs"] = runs_dict[label]
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
def _(n_experiments_dict, rng):
    # compare runs with the high of the rng at the min of number of experiments in methods
    runs_to_compare = rng.integers(low=1, high=min(n_experiments_dict.values()), size=6)
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
    stages = [i+1 for i in range(num_analyses.value)]
    return (stages,)


@app.cell
def _(num_analyses):
    lower_boundary_value_labels = (
        ["lower" + f"{i+1}" for i in range(num_analyses.value-1)] + ["upper" + f"{num_analyses.value}"]
    )
    return (lower_boundary_value_labels,)


@app.cell
def _(num_analyses):
    upper_boundary_value_labels = ["upper" + f"{i+1}" for i in range(num_analyses.value)]
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
        _b.set(xlabel="Trial stages", xticks=stages)

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
        _b.set(xlabel="Trial stages", xticks=stages)

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
        _b.set(xlabel="Trial stages", xticks=stages)

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
def _(datasets, labels):
    # empty dictionaries to fill with sorted constrained data
    _constrained_data = dict()
    sorted_constrained_data = dict()

    for _i, _label_ui in enumerate(labels.elements):
        # get the labels for the sorted constrained data dictionary
        _label_value = _label_ui.value if _label_ui.value else f"Method_{_i+1}"

        # get rows only where alpha <= 0.05
        _constrained_data[_label_value] = datasets[_label_value][datasets[_label_value]["alpha"] <= 0.05]

        # group by the run, get the index of the minimum value for the objective function, then sort
        # reset the index, so indexing can start from 0
        # save this in the dictionary position for the corresponding label
        idx = (
            _constrained_data[_label_value]
            .groupby("runs")["obj_func"]
            .idxmin()
        )

        sorted_constrained_data[_label_value] = (
            _constrained_data[_label_value]
            .loc[idx]
            .sort_values("obj_func")
            .reset_index(drop=True)
        )
    return (sorted_constrained_data,)


@app.cell
def _(
    column_runs_compare,
    datasets,
    lower_boundary_value_labels,
    mlines,
    mo,
    n_experiments_dict,
    plt,
    sorted_constrained_data,
    stages,
    tri,
    upper_boundary_value_labels,
):
    if not datasets or len(column_runs_compare) < 2:
        mo.md("")

    _num_plots = len(datasets)
    _fig, _ax = plt.subplots(1, _num_plots, figsize=(5 * _num_plots, 6), sharey=True, squeeze=False)
    _ax = _ax.flatten()

    for _idx, (_label, _data) in enumerate(datasets.items()):
        for _run_idx in range(n_experiments_dict[_label]):

            _b = _ax[_idx]

            _b.set_title(_label + f", top {n_experiments_dict[_label]} bounds")
            _b.set(xlabel="Trial stages", xticks=stages)

            _b.plot(stages, 
                    sorted_constrained_data[_label].loc[_run_idx, upper_boundary_value_labels], 
                    color="purple", alpha = 0.15)

            _b.plot(stages, 
                    sorted_constrained_data[_label].loc[_run_idx, lower_boundary_value_labels], 
                    color="purple", alpha = 0.15)

        # draw triangular bounds
        _b.plot(stages, tri[0], color="darkorange", zorder = 3)
        _b.plot(stages, tri[1], color="darkorange", zorder = 3)

        # custom legend
        best_bound_label = mlines.Line2D([], [], color='purple', label = "Best bounds")
        tri_bound_label = mlines.Line2D([], [], color ="darkorange", label = "Tri bounds")
        _b.legend(handles = [best_bound_label, tri_bound_label], loc="lower right")

    _ax[0].set_ylabel("$Z_k$ values")
    _fig.suptitle(f"Top $x$ constrained boundaries", y=0.96)
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Top sorted, constrained boundaries with slider
    """)
    return


@app.cell
def _(mo, n_experiments_dict):
    slider = mo.ui.slider(start=0, stop=min(n_experiments_dict.values())-1)
    return (slider,)


@app.cell
def _(mo, slider):
    mo.vstack([slider, mo.md(f"Has value: {slider.value}")])
    return


@app.cell
def _(
    column_runs_compare,
    datasets,
    lower_boundary_value_labels,
    mlines,
    mo,
    n_experiments_dict,
    num_analyses,
    plt,
    slider,
    sorted_constrained_data,
    stages,
    tri,
    upper_boundary_value_labels,
):
    if not datasets or len(column_runs_compare) < 2:
        mo.md("")

    _num_plots = len(datasets)
    _fig, _ax = plt.subplots(1, _num_plots, figsize=(6 * _num_plots, 3.5), sharey=True, squeeze=False)
    _ax = _ax.flatten()

    for _idx, (_label, _data) in enumerate(datasets.items()):

        _b = _ax[_idx]

        if num_analyses.value > 3:
            lower_ylim = -15
            _b.set_ylim(lower_ylim, 15)
        else:
            lower_ylim = -8
            _b.set_ylim(lower_ylim, 8)

        _b.set_title(_label + f", top {n_experiments_dict[_label]} bounds")
        _b.set(xlabel="Trial stages", xticks=stages)

        _b.plot(stages, 
                sorted_constrained_data[_label].loc[slider.value, upper_boundary_value_labels], 
                color="purple")

        _b.plot(stages, 
                sorted_constrained_data[_label].loc[slider.value, lower_boundary_value_labels], 
                color="purple")

        _b.text(stages[1]-1, lower_ylim + 2.3, 
                "$\\alpha$ = " + str(sorted_constrained_data[_label].loc[slider.value, "alpha"].round(4)))

        _b.text(stages[1]-1, lower_ylim + 0.5, 
                "$1-\\beta$ = " + str(sorted_constrained_data[_label].loc[slider.value, "power"].round(4)))

        _b.text(stages[2]-1, lower_ylim + 2.3, 
                "n = " + str(sorted_constrained_data[_label].loc[slider.value, "sample_size"].round(1)))   

        _b.text(stages[2]-1, lower_ylim + 0.5, 
                "Max ESS = " + str(sorted_constrained_data[_label].loc[slider.value, "max_ess"].round(1)))

        _b.text(stages[2]-1, lower_ylim + 4.3, 
                "$\mathcal{L = }$" + str(sorted_constrained_data[_label].loc[slider.value, "obj_func"].round(4)))


        # draw triangular bounds
        _b.plot(stages, tri[0], color="darkorange", zorder = 3)
        _b.plot(stages, tri[1], color="darkorange", zorder = 3)

        # custom legend
        _best_bound_label = mlines.Line2D([], [], color='purple', label = "Best bounds")
        _tri_bound_label = mlines.Line2D([], [], color ="darkorange", label = "Tri bounds")
        _b.legend(handles = [_best_bound_label, _tri_bound_label], loc="lower right")

    _ax[0].set_ylabel("$Z_k$ values")
    _fig.suptitle(f"Top $x$ constrained boundaries, sorted by loss", y=0.96)
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Empirical CDF of top constrained bounds
    """)
    return


@app.cell
def _(mpl):
    # get colors to use
    colors = mpl.color_sequences["tab20"]
    return (colors,)


@app.cell
def _(
    datasets,
    lower_boundary_value_labels,
    np,
    num_analyses,
    plt,
    runs_dict,
    stages,
    upper_boundary_value_labels,
):
    _fig, _ax = plt.subplots(nrows=2, ncols=num_analyses.value, figsize=(11,5), sharey=True)
    _ax[1, stages[-1] - 1].axis("off")

    for _idx, (_label, _data) in enumerate(datasets.items()):

        _upper = {key: [] for key in upper_boundary_value_labels}
        _lower = {key: [] for key in lower_boundary_value_labels}

        for _run_idx, _run in enumerate(np.unique(runs_dict[_label])):
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
            _ax[0, colu].set_title(key[0:5] + " bound " + key[5])
            if colu == stages[-1] - 1:
                _ax[0, colu].legend()

        for colu, key in enumerate(lower_boundary_value_labels):
            if colu == stages[-1] - 1: break
            _ax[1, colu].ecdf(_lower[key], label=_label)
            _ax[1, colu].set_title(key[0:5] + " bound " + key[5])
            #_ax[1, colu].legend()

    _fig.suptitle(f"ECDF of top $x$ constrained boundaries", y=0.96)
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Box plots of bounds
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Distribution of all bounds
    """)
    return


@app.cell
def _(
    best_constrained_bound_getter,
    colors,
    datasets,
    lower_boundary_value_labels,
    n_experiments_dict,
    n_loops_dict,
    np,
    num_analyses,
    plt,
    stages,
    upper_boundary_value_labels,
):
    _fig, _ax = plt.subplots()

    _num_plots = len(datasets)
    _fig, _ax = plt.subplots(1, _num_plots, figsize=(4 * _num_plots, 5), sharey=True, squeeze=False)
    _ax = _ax.flatten()

    color_idx = 0

    for _idx, (_label, _data) in enumerate(datasets.items()):
        _upper, _lower, _obj_f, _alpha, _power = best_constrained_bound_getter(_data)
        for stage in range(num_analyses.value):

            _b = _ax[_idx]

            # title and xlabel
            _b.set_title(_label + f", distribution of all {n_experiments_dict[_label] * n_loops_dict[_label]} bounds")
            _b.set(xlabel="Trial stages", xticks=[1, 2, 3])

            # best constrained boundaries
            _b.scatter(stages, _upper, color="black", zorder=4)
            _b.scatter(stages, _lower, color="black", zorder=4)

            # upper bound plots       
            _b.violinplot(_data[upper_boundary_value_labels[stage]], positions=[stage+1],
                           showmeans=False, 
                           showmedians=False,
                           showextrema=False,
                           facecolor=(colors[color_idx], 0.3))

            # calculate the median and quartiles
            q1_upper, median_upper, q3_upper = np.percentile(_data[upper_boundary_value_labels[stage]], 
                                                             [25, 50, 75])

            _b.vlines(stage+1, q1_upper, q3_upper, color=colors[color_idx], linestyle='-', lw=5)
            _b.hlines(median_upper, stage+1-0.07, stage+1+0.07, color=colors[color_idx], zorder=3)

            # lower bound plots
            _b.violinplot(_data[lower_boundary_value_labels[stage]], positions=[stage+1],
                           showmeans=False, 
                           showmedians=False,
                           showextrema=False,
                           facecolor=(colors[color_idx], 0.3))

            # calculate the median and quartiles
            q1_lower, median_lower, q3_lower = np.percentile(_data[lower_boundary_value_labels[stage]], 
                                                             [25, 50, 75])

            _b.vlines(stage+1, q1_lower, q3_lower, color=colors[color_idx], linestyle='-', lw=5)
            _b.hlines(median_lower, stage+1-0.07, stage+1+0.07, color=colors[color_idx], zorder=3)

            color_idx += 1
        # reset index to 0 so upper and lower plots are paired
        color_idx = 0

    _ax[0].set_ylabel("$Z_k$ values")
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Distribution of best constrained bounds
    """)
    return


@app.cell
def _(
    best_constrained_bound_getter,
    colors,
    datasets,
    lower_boundary_value_labels,
    n_experiments_dict,
    np,
    plt,
    runs_dict,
    stages,
    upper_boundary_value_labels,
):
    _fig, _ax = plt.subplots()

    _num_plots = len(datasets)
    _fig, _ax = plt.subplots(1, _num_plots, figsize=(4 * _num_plots, 5), sharey=True, squeeze=False)
    _ax = _ax.flatten()

    _color_idx = 0

    for _idx, (_label, _data) in enumerate(datasets.items()):

        _upper = {key: [] for key in upper_boundary_value_labels}
        _lower = {key: [] for key in lower_boundary_value_labels}

        _upper1, _lower1, _obj_f, _alpha, _power = best_constrained_bound_getter(_data)

        for _run_idx, _run in enumerate(np.unique(runs_dict[_label])):
            _run_data = _data[_data["runs"] == _run]

            _constrained_data = _run_data[_run_data["alpha"] <= 0.05]
            _min_idx = _constrained_data["obj_func"].idxmin()

            # get the upper bound values
            for _key in _upper:
                _upper[_key].append(_constrained_data.loc[_min_idx, _key])

            # get the lower bound values
            for _key in _lower:
                _lower[_key].append(_constrained_data.loc[_min_idx, _key])

        for _colu, _key in enumerate(upper_boundary_value_labels):

            _b = _ax[_idx]

            # title and labels
            _b.set_title(_label + f", distribution of {n_experiments_dict[_label]} best bounds")
            _b.set(xlabel="Trial stages", xticks=[1, 2, 3])

            # best constrained boundaries
            _b.scatter(stages, _upper1, color="black", zorder=4)
            _b.scatter(stages, _lower1, color="black", zorder=4)

            # upper bound plots       
            _b.violinplot(_upper[_key], positions=[_colu+1],
                           showmeans=False, 
                           showmedians=False,
                           showextrema=False,
                           facecolor=(colors[_color_idx], 0.3))

            # calculate the median and quartiles
            _q1_upper, _median_upper, _q3_upper = np.percentile(_upper[_key], [25, 50, 75])

            _b.vlines(_colu+1, _q1_upper, _q3_upper, color=colors[_color_idx], linestyle='-', lw=5)
            _b.hlines(_median_upper, _colu+1-0.07, _colu+1+0.07, color=colors[_color_idx], zorder=3)

            _color_idx += 1

        # reset index to 0 so upper and lower plots are paired
        _color_idx = 0

        for _colu, _key in enumerate(lower_boundary_value_labels):
            _b = _ax[_idx]

            # lower bound plots
            _b.violinplot(_lower[_key], positions=[_colu+1],
                           showmeans=False, 
                           showmedians=False,
                           showextrema=False,
                           facecolor=(colors[_color_idx], 0.3))

            # calculate the median and quartiles
            _q1_lower, _median_lower, _q3_lower = np.percentile(_lower[_key], [25, 50, 75])

            _b.vlines(_colu+1, _q1_lower, _q3_lower, color=colors[_color_idx], linestyle='-', lw=5)
            _b.hlines(_median_lower, _colu+1-0.07, _colu+1+0.07, color=colors[_color_idx], zorder=3)

            _color_idx += 1
        # reset index to 0 so upper and lower plots are paired
        _color_idx = 0


    _ax[0].set_ylabel("$Z_k$ values")

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Loss improvement over iterations
    """)
    return


@app.cell
def _(mo, n_experiments_dict):
    slider2 = mo.ui.slider(start=0, stop=min(n_experiments_dict.values())-1)
    return (slider2,)


@app.cell
def _(mo, slider2):
    mo.vstack([slider2, mo.md(f"Has value: {slider2.value}")])
    return


@app.cell
def _(datasets, np, plt, runs_dict, slider2):
    _fig, _ax = plt.subplots(figsize=(12,5))

    style = ["-", "--", ":", "-."]

    for _idx, (_label, _data) in enumerate(datasets.items()):

        run_to_assess = np.unique(runs_dict[_label])[slider2.value]
        data_to_assess = _data[_data["runs"] == run_to_assess]

        obj_func_vals = data_to_assess["obj_func"].to_numpy()
        x_plot = np.arange(len(obj_func_vals))

        running_min = np.minimum.accumulate(obj_func_vals)
        _ax.step(x_plot, running_min, label = _label, linestyle=style[_idx])

    _ax.set_title(f"Minimum objective function value over iteration, {run_to_assess}")
    _ax.set_xlabel("Iteration number")
    _ax.set_ylabel("Objective function, $\mathcal{L}$")

    _ax.legend()
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
