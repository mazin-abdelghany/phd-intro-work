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

    return np, pd, plt


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


@app.cell
def _(mo):
    n_experiments = mo.ui.number(label="No. experiments = ")
    return (n_experiments,)


@app.cell
def _(mo):
    n_loops = mo.ui.number(label="Number of loops = ")
    return (n_loops,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Experimental setup
    """)
    return


@app.cell
def _(mo, n_experiments):
    mo.vstack(
        [n_experiments]
    )
    return


@app.cell
def _(mo, n_loops):
    mo.vstack(
        [n_loops]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data files to compare
    """)
    return


@app.cell
def _(mo):
    label_a = mo.ui.text(placeholder="data_label", label="Method 1: ")
    return (label_a,)


@app.cell
def _(mo):
    label_b = mo.ui.text(placeholder="data_label", label="Method 2: ")
    return (label_b,)


@app.cell
def _(label_a, label_b, mo):
    mo.vstack([label_a, label_b])
    return


@app.cell
def _(mo):
    file_browser = mo.ui.file_browser(initial_path = "/tf/experiments_rand_simann_bo/",
                                      label = "Select files in the order of the methods.")
    return (file_browser,)


@app.cell
def _(file_browser, mo):
    mo.vstack([file_browser])
    return


@app.cell
def _(file_browser, pd):
    # import the data
    data_a = pd.read_csv(file_browser.path(index=0))
    data_b = pd.read_csv(file_browser.path(index=1))

    # remove the first column as it is not needed
    data_a = data_a.iloc[:, 1:]
    data_b = data_b.iloc[:, 1:]
    return data_a, data_b


@app.cell
def _(n_experiments, n_loops, np):
    runs = np.concatenate([np.repeat(f"Run_{i+1}", n_loops.value) for i in range(n_experiments.value)])
    return (runs,)


@app.cell
def _(data_a, data_b, runs):
    data_a["runs"] = runs
    data_b["runs"] = runs
    return


@app.cell
def _(mo):
    seed_for_runs = mo.ui.number(label="RNG seed: ")
    return (seed_for_runs,)


@app.cell
def _(mo, seed_for_runs):
    mo.vstack(
        [seed_for_runs]
    )
    return


@app.cell
def _(np, seed_for_runs):
    rng = np.random.default_rng(seed=seed_for_runs.value)
    return (rng,)


@app.cell
def _(rng):
    runs_to_compare = rng.integers(low=1, high=51, size=6)
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
def _(data_a, data_b, label_a, label_b, np, plt):
    plot_cols = ["alpha", "power", "sample_size", "max_ess", "obj_func"]

    fig, ax = plt.subplots(nrows=len(plot_cols), ncols=2, figsize=(8,11), sharey="row")

    for i, col in enumerate(plot_cols):
        for j, data in enumerate([data_a, data_b]):
            ax[i, j].violinplot(
                data[col],
                showextrema=False,
                showmedians=True
            )

            ax[i,j].text(
                0.5,
                0.7,
                np.round(np.median(data[col]), 3),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax[i,j].transAxes
            )

    ax[0,0].set_title(label_a.value)
    ax[0,1].set_title(label_b.value)

    ax[0,0].set_ylabel("$\\alpha'$")
    ax[1,0].set_ylabel("$1-\\beta'$")
    ax[2,0].set_ylabel("Sample size")
    ax[3,0].set_ylabel("Max ESS")
    ax[4,0].set_ylabel("Loss")

    for a in ax.flat:
        a.set_xticks([])

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Boundary comparisons
    """)
    return


@app.cell
def _(np):
    def best_bound_getter(data):
        min_idx = data["obj_func"].idxmin()
        lower = data.loc[min_idx, ["upper1", "upper2","upper3"]].tolist()
        upper = data.loc[min_idx, ["lower1", "lower2","upper3"]].tolist()
        obj_f = np.round(data.loc[min_idx, "obj_func"], decimals=4)
        alpha = np.round(data.loc[min_idx, "alpha"], decimals=4)
        power = np.round(data.loc[min_idx, "power"], decimals=4)
        return lower, upper, obj_f, alpha, power

    return (best_bound_getter,)


@app.cell
def _(num_analyses):
    stages = [i+1 for i in range(num_analyses)]
    return (stages,)


@app.cell
def _(best_bound_getter, data_a, data_b, label_a, label_b, plt, stages, tri):
    _fig, _ax = plt.subplots(1, 2, figsize=(14,3), sharey=True)

    _results = [
        (*best_bound_getter(data_a), _ax[0]),
        (*best_bound_getter(data_b), _ax[1]),
    ]

    for _upper, _lower, _obj_f, _alpha, _power, _b in _results:
        _b.set(xlabel="Trial stages", xticks=[1,2,3])

        _b.plot(stages, tri[0], color="darkorange", label="Tri bound")
        _b.plot(stages, tri[1], color="darkorange")

        _b.plot(stages, _upper, color="purple", label="Best bound")
        _b.plot(stages, _lower, color="purple")

        for _y, _txt in zip(
            [0.96, 0.88, 0.8],
            [f"$\\mathcal{{L}}$ = {_obj_f}",
             f"$\\alpha$ = {_alpha}",
             f"$1-\\beta$ = {_power}"]
        ):
            _b.text(0.98, _y, _txt,
                   ha="right", va="top",
                   transform=_b.transAxes)

    _ax[0].set_title(label_a.value)
    _ax[1].set_title(label_b.value)

    _fig.suptitle("Best boundary", y=1.05)

    _ax[0].set_ylabel("$Z_k$ values")

    for _i in range(_ax.shape[0]):
        _ax[_i].legend(loc="lower right")

    plt.gca()
    return


@app.cell
def _(np):
    def best_constrained_bound_getter(data):
        constrained_data = data[data["alpha"] <= 0.05]
        min_idx = constrained_data["obj_func"].idxmin()
        lower = constrained_data.loc[min_idx, ["upper1", "upper2","upper3"]].tolist()
        upper = constrained_data.loc[min_idx, ["lower1", "lower2","upper3"]].tolist()
        obj_f = np.round(constrained_data.loc[min_idx, "obj_func"], decimals=4)
        alpha = np.round(constrained_data.loc[min_idx, "alpha"], decimals=4)
        power = np.round(constrained_data.loc[min_idx, "power"], decimals=4)
        return lower, upper, obj_f, alpha, power

    return (best_constrained_bound_getter,)


@app.cell
def _(
    best_constrained_bound_getter,
    data_a,
    data_b,
    label_a,
    label_b,
    plt,
    stages,
    tri,
):
    _fig, _ax = plt.subplots(1, 2, figsize=(14,3), sharey=True)

    _results = [
        (*best_constrained_bound_getter(data_a), _ax[0]),
        (*best_constrained_bound_getter(data_b), _ax[1]),
    ]

    for _upper, _lower, _obj_f, _alpha, _power, _b in _results:
        _b.set(xlabel="Trial stages", xticks=[1,2,3])

        _b.plot(stages, tri[0], color="darkorange", label="Tri bound")
        _b.plot(stages, tri[1], color="darkorange")

        _b.plot(stages, _upper, color="purple", label="Best bound")
        _b.plot(stages, _lower, color="purple")

        for _y, _txt in zip(
            [0.96, 0.88, 0.8],
            [f"$\\mathcal{{L}}$ = {_obj_f}",
             f"$\\alpha$ = {_alpha}",
             f"$1-\\beta$ = {_power}"]
        ):
            _b.text(0.98, _y, _txt,
                   ha="right", va="top",
                   transform=_b.transAxes)

    _ax[0].set_title(label_a.value)
    _ax[1].set_title(label_b.value)

    _fig.suptitle("Best constrained boundary", y=1.05)

    _ax[0].set_ylabel("$Z_k$ values")

    for _i in range(_ax.shape[0]):
        _ax[_i].legend(loc="lower right")

    plt.gca()
    return


@app.cell
def _(np):
    def rand_bound_getter(data, run):
        run_data = data[data["runs"] == run]
        min_idx = run_data["obj_func"].idxmin()
        lower = run_data.loc[min_idx, ["upper1", "upper2","upper3"]].tolist()
        upper = run_data.loc[min_idx, ["lower1", "lower2","upper3"]].tolist()
        obj_f = np.round(run_data.loc[min_idx, "obj_func"], decimals=4)
        alpha = np.round(run_data.loc[min_idx, "alpha"], decimals=4)
        power = np.round(run_data.loc[min_idx, "power"], decimals=4)
        return lower, upper, obj_f, alpha, power

    return (rand_bound_getter,)


@app.cell
def _(
    column_runs_compare,
    data_a,
    data_b,
    label_a,
    label_b,
    plt,
    rand_bound_getter,
    stages,
    tri,
):
    _fig, _ax = plt.subplots(1, 2, figsize=(14,3), sharey=True)

    _results = [
        (*rand_bound_getter(data_a, column_runs_compare[0]), _ax[0]),
        (*rand_bound_getter(data_b, column_runs_compare[1]), _ax[1]),
    ]

    for _upper, _lower, _obj_f, _alpha, _power, _b in _results:
        _b.set(xlabel="Trial stages", xticks=[1,2,3])

        _b.plot(stages, tri[0], color="darkorange", label="Tri bound")
        _b.plot(stages, tri[1], color="darkorange")

        _b.plot(stages, _upper, color="purple", label="Best bound")
        _b.plot(stages, _lower, color="purple")

        for _y, _txt in zip(
            [0.96, 0.88, 0.8],
            [f"$\\mathcal{{L}}$ = {_obj_f}",
             f"$\\alpha$ = {_alpha}",
             f"$1-\\beta$ = {_power}"]
        ):
            _b.text(0.98, _y, _txt,
                   ha="right", va="top",
                   transform=_b.transAxes)

    _ax[0].set_title(label_a.value + " " + column_runs_compare[0])
    _ax[1].set_title(label_b.value + " " + column_runs_compare[1])

    _fig.suptitle("Best boundary -- Random run 1", y=1.05)

    _ax[0].set_ylabel("$Z_k$ values")

    for _i in range(_ax.shape[0]):
        _ax[_i].legend(loc="lower right")

    plt.gca()
    return


@app.cell
def _(
    column_runs_compare,
    data_a,
    data_b,
    label_a,
    label_b,
    plt,
    rand_bound_getter,
    stages,
    tri,
):
    _fig, _ax = plt.subplots(1, 2, figsize=(14,3), sharey=True)

    _results = [
        (*rand_bound_getter(data_a, column_runs_compare[2]), _ax[0]),
        (*rand_bound_getter(data_b, column_runs_compare[3]), _ax[1]),
    ]

    for _upper, _lower, _obj_f, _alpha, _power, _b in _results:
        _b.set(xlabel="Trial stages", xticks=[1,2,3])

        _b.plot(stages, tri[0], color="darkorange", label="Tri bound")
        _b.plot(stages, tri[1], color="darkorange")

        _b.plot(stages, _upper, color="purple", label="Best bound")
        _b.plot(stages, _lower, color="purple")

        for _y, _txt in zip(
            [0.96, 0.88, 0.8],
            [f"$\\mathcal{{L}}$ = {_obj_f}",
             f"$\\alpha$ = {_alpha}",
             f"$1-\\beta$ = {_power}"]
        ):
            _b.text(0.98, _y, _txt,
                   ha="right", va="top",
                   transform=_b.transAxes)

    _ax[0].set_title(label_a.value + " " + column_runs_compare[2])
    _ax[1].set_title(label_b.value + " " + column_runs_compare[3])

    _fig.suptitle("Best boundary -- Random run 2", y=1.05)

    _ax[0].set_ylabel("$Z_k$ values")

    for _i in range(_ax.shape[0]):
        _ax[_i].legend(loc="lower right")

    plt.gca()
    return


if __name__ == "__main__":
    app.run()
