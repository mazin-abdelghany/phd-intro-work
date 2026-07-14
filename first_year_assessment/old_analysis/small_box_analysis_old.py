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
    # Setup
    """)
    return


@app.cell
def _(ss):
    # design settings
    num_analyses = 3
    target_alpha = 0.05
    target_power = 0.9
    delta0 = 0.
    delta1 = 1.
    sigma2 = 3.

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
    # this function contains a penalty for non-monotonicity
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
    np,
    num_analyses,
    obj_f,
    sigma2,
    ss,
    target_alpha,
    target_power,
):
    tri = bd.calculate_triangular_boundaries(
        n_analyses = num_analyses,
        alpha = target_alpha,
        delta = delta1
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


    print(f"Original trriangular params: {np.round(np.concatenate((tri[0], tri[1])), 4)}")
    print(f"Triangular benchmark objective: {tri_obj:.4f}")
    print(f"Triangular alpha: {tri_alpha:.4f}")
    print(f"Triangular delta alpha: {abs(0.05-tri_alpha):.4f}")
    print(f"Triangular power: {tri_power:.4f}")
    print(f"Triangular delta power: {abs(0.9-tri_power):.4f}")
    print(f"Triangular sample size: {tri_n_patients:.1f}")
    print(f"Triangular max ESS: {tri_max_ess:.1f}")
    return tri, tri_alpha, tri_max_ess, tri_obj


@app.cell
def _():
    n_loops = 500
    return (n_loops,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Import the data
    """)
    return


@app.cell
def _(pd):
    # import the run with sample size from uniform distribution
    small_box_rand = pd.read_csv(
        filepath_or_buffer="/tf/experiments_rand_simann_bo/random_search_experiments/old_data/small_box_ss_change.csv"
    )

    small_box_bo = pd.read_csv(
        filepath_or_buffer="/tf/experiments_rand_simann_bo/bayes_opt_experiments/old_data/small_box_bo.csv"
    )

    # importing the corrected data
    small_box_sim_ann = pd.read_csv(
        filepath_or_buffer=
        "/tf/experiments_rand_simann_bo/simulated_annealing_experiments/old_data/small_box_t100_neigh2_corrected.csv"
    )

    # remove the first column as it is not needed
    small_box_rand = small_box_rand.iloc[:, 1:]
    small_box_bo = small_box_bo.iloc[:, 1:]
    small_box_sim_ann = small_box_sim_ann.iloc[:, 1:]
    return small_box_bo, small_box_rand, small_box_sim_ann


@app.cell
def _(small_box_sim_ann):
    # reassign the objective function column to the correct values
    small_box_sim_ann["obj_func"] = small_box_sim_ann["obj_func_corrected"]
    return


@app.cell
def _(np):
    runs = np.concatenate([np.repeat(f"Run_{i+1}", 500) for i in range(50)])
    return (runs,)


@app.cell
def _(
    delta0,
    delta1,
    mu,
    num_analyses,
    obj_f,
    sigma2,
    target_alpha,
    target_power,
):
    # correct the objective function calculation in sim anneal
    def correct_obj_func_error(data, index):
        upper_bounds = data.iloc[index][["upper1", "upper2", "upper3"]].to_list()
        lower_bounds = data.iloc[index][["lower1", "lower2", "upper3"]].to_list()
        sample_size = data.iloc[index]["sample_size"]

        _, _, _, obj = obj_f(
            mu = mu,
            upper_bounds = upper_bounds,
            lower_bounds = lower_bounds,
            n_analyses = num_analyses,
            n_patients = sample_size,
            target_power = target_power,
            target_alpha = target_alpha,
            null_hypothesis = delta0,
            alternative_hypothesis = delta1,
            variance = sigma2
        )

        return obj

    return


@app.cell
def _():
    # used to recalculate the objective function that was saved wrong
    # small_box_sim_ann["obj_func_corrected"] = [
    #     correct_obj_func_error(small_box_sim_ann, idx)
    #     for idx in small_box_sim_ann.index
    # ]
    return


@app.cell
def _():
    # used to save the data for the objective function that was saved wrong
    # small_box_sim_ann.to_csv("/tf/2026-04-t21/simulated_annealing_experiments/small_box_t100_neigh2_corrected.csv")
    return


@app.cell
def _(runs, small_box_bo, small_box_rand, small_box_sim_ann):
    small_box_rand["runs"] = runs
    small_box_bo["runs"] = runs
    small_box_sim_ann["runs"] = runs
    return


@app.cell
def _(np):
    rng = np.random.default_rng(seed=19804368890745)
    return (rng,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Random run summary
    """)
    return


@app.cell
def _(rng):
    rng.integers(low=1, high=51, size=9)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random search
    """)
    return


@app.cell
def _(np, small_box_rand):
    np.round(
        small_box_rand[small_box_rand["runs"] == "Run_4"][["alpha","power","sample_size","max_ess","obj_func"]].mean(),
        decimals = 2
    )
    return


@app.cell
def _(np, small_box_rand):
    np.round(
        small_box_rand[small_box_rand["runs"] == "Run_5"][["alpha","power","sample_size","max_ess","obj_func"]].mean(),
        decimals = 2
    )
    return


@app.cell
def _(np, small_box_rand):
    np.round(
        small_box_rand[small_box_rand["runs"] == "Run_48"][["alpha","power","sample_size","max_ess","obj_func"]].mean(),
        decimals = 2
    )
    return


@app.cell
def _(np, small_box_rand):
    np.round(
        small_box_rand[["alpha","power","sample_size","max_ess","obj_func"]].describe(),
        decimals = 2
    )
    return


@app.cell
def _(np, small_box_rand):
    np.round(
        small_box_rand[["alpha","power","sample_size","max_ess","obj_func"]].iloc[np.argmin(small_box_rand["obj_func"])],
        decimals = 2
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulated annealing
    """)
    return


@app.cell
def _(np, small_box_sim_ann):
    np.round(
        small_box_sim_ann[small_box_sim_ann["runs"] == "Run_9"][["alpha","power","sample_size","max_ess","obj_func"]].mean(),
        decimals = 2
    )
    return


@app.cell
def _(np, small_box_sim_ann):
    np.round(
        small_box_sim_ann[small_box_sim_ann["runs"] == "Run_18"][["alpha","power","sample_size","max_ess","obj_func"]].mean(),
        decimals = 2
    )
    return


@app.cell
def _(np, small_box_sim_ann):
    np.round(
        small_box_sim_ann[small_box_sim_ann["runs"] == "Run_21"][["alpha","power","sample_size","max_ess","obj_func"]].mean(),
        decimals = 2
    )
    return


@app.cell
def _(np, small_box_sim_ann):
    np.round(
        small_box_sim_ann[["alpha","power","sample_size","max_ess","obj_func"]].describe(),
        decimals = 2
    )
    return


@app.cell
def _(np, small_box_sim_ann):
    np.round(
        small_box_sim_ann[["alpha","power","sample_size","max_ess","obj_func"]].iloc[np.argmin(small_box_sim_ann["obj_func"])],
        decimals = 2
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayesian optimisation
    """)
    return


@app.cell
def _(np, small_box_bo):
    np.round(
        small_box_bo[small_box_bo["runs"] == "Run_2"][["alpha","power","sample_size","max_ess","obj_func"]].mean(),
        decimals = 2
    )
    return


@app.cell
def _(np, small_box_bo):
    np.round(
        small_box_bo[small_box_bo["runs"] == "Run_9"][["alpha","power","sample_size","max_ess","obj_func"]].mean(),
        decimals = 2
    )
    return


@app.cell
def _(np, small_box_bo):
    np.round(
        small_box_bo[small_box_bo["runs"] == "Run_44"][["alpha","power","sample_size","max_ess","obj_func"]].mean(),
        decimals = 2
    )
    return


@app.cell
def _(np, small_box_bo):
    np.round(
        small_box_bo[["alpha","power","sample_size","max_ess","obj_func"]].describe(),
        decimals = 2
    )
    return


@app.cell
def _(np, small_box_bo):
    np.argmin(small_box_bo["obj_func"])
    return


@app.cell
def _(np, small_box_bo):
    np.round(
        small_box_bo[["alpha","power","sample_size","max_ess","obj_func"]].iloc[np.argmin(small_box_bo["obj_func"])],
        decimals = 2
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plots
    """)
    return


@app.cell
def _(np, plt, small_box_bo, small_box_rand, small_box_sim_ann):
    fig, ax = plt.subplots(nrows=5, ncols=3, figsize=(8,11), sharey="row")

    ax[0,0].violinplot(small_box_rand["alpha"], showextrema=False, showmedians=True)
    ax[0,0].text(0.95, 0.05, np.round(np.median(small_box_rand["alpha"]), decimals=2))

    ax[0,1].violinplot(small_box_sim_ann["alpha"], showextrema=False, showmedians=True)
    ax[0,1].text(0.95, 0.05, np.round(np.median(small_box_sim_ann["alpha"]), decimals=2))

    ax[0,2].violinplot(small_box_bo["alpha"], showextrema=False, showmedians=True)
    ax[0,2].text(0.95, 0.05, np.round(np.median(small_box_bo["alpha"]), decimals=2))

    ax[1,0].violinplot(small_box_rand["power"], showextrema=False, showmedians=True)
    ax[1,0].text(0.965, 0.89, np.round(np.median(small_box_rand["power"]), decimals=2))

    ax[1,1].violinplot(small_box_sim_ann["power"], showextrema=False, showmedians=True)
    ax[1,1].text(0.965, 0.89, np.round(np.median(small_box_sim_ann["power"]), decimals=2))

    ax[1,2].violinplot(small_box_bo["power"], showextrema=False, showmedians=True)
    ax[1,2].text(0.965, 0.89, np.round(np.median(small_box_bo["power"]), decimals=2))

    ax[2,0].violinplot(small_box_rand["sample_size"], showextrema=False, showmedians=True)
    ax[2,0].text(0.97, 54, np.round(np.median(small_box_rand["sample_size"]).astype(int), decimals=0))

    ax[2,1].violinplot(small_box_sim_ann["sample_size"], showextrema=False, showmedians=True)
    ax[2,1].text(0.97, 56, np.round(np.median(small_box_sim_ann["sample_size"]).astype(int), decimals=0))

    ax[2,2].violinplot(small_box_bo["sample_size"], showextrema=False, showmedians=True)
    ax[2,2].text(0.97, 55, np.round(np.median(small_box_bo["sample_size"]).astype(int), decimals=0))

    ax[3,0].violinplot(small_box_rand["max_ess"], showextrema=False, showmedians=True)
    ax[3,0].text(0.97, 132, np.round(np.median(small_box_rand["max_ess"]).astype(int), decimals=0))

    ax[3,1].violinplot(small_box_sim_ann["max_ess"], showextrema=False, showmedians=True)
    ax[3,1].text(0.96, 135, np.round(np.median(small_box_sim_ann["max_ess"])).astype(int))

    ax[3,2].violinplot(small_box_bo["max_ess"], showextrema=False, showmedians=True)
    ax[3,2].text(0.96, 133, np.round(np.median(small_box_bo["max_ess"])).astype(int))

    ax[4,0].violinplot(small_box_rand["obj_func"], showextrema=False, showmedians=True)
    ax[4,0].text(0.985, 7, np.round(np.median(small_box_rand["obj_func"])).astype(int))

    ax[4,1].violinplot(small_box_sim_ann["obj_func"], showextrema=False, showmedians=True)
    ax[4,1].text(0.985, 7, np.round(np.median(small_box_sim_ann["obj_func"])).astype(int))

    ax[4,2].violinplot(small_box_bo["obj_func"], showextrema=False, showmedians=True)
    ax[4,2].text(0.985, 7, np.round(np.median(small_box_bo["obj_func"])).astype(int))

    ax[0,0].set_title("Random")
    ax[0,1].set_title("Sim anneal")
    ax[0,2].set_title("Bayes opt")

    ax[0,0].set_ylabel("$\\alpha'$")
    ax[1,0].set_ylabel("$1-\\beta'$")
    ax[2,0].set_ylabel("Sample size")
    ax[3,0].set_ylabel("Max ESS")
    ax[4,0].set_ylabel("Loss")

    for a in ax.flat:
        a.set_xticks([])

    plt.savefig("/tf/first_year_assessment/analysis/violin_plots_small_box.png", dpi=300, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Best indices
    """)
    return


@app.cell
def _(n_loops, np, small_box_bo):
    best_indices = []
    for ell in range(50):
        _start_index = ell * n_loops
        _stop_index = _start_index + n_loops

        _analysis_set = small_box_bo.iloc[_start_index:_stop_index, 6:11]
        best_index = np.argmin(_analysis_set['obj_func'])
        best_indices.append(best_index)

    print(f"Minimum best index: {np.min(best_indices)+1}")
    print(f"Maximum best index: {np.max(best_indices)+1}")
    print(f"Average best index: {np.mean(best_indices)+1}")
    print(f"Median best index: {np.median(best_indices)+1}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Feasibility and strict feasibility
    """)
    return


@app.cell
def _(n_loops, np, pd, small_box_bo, target_alpha, target_power):
    feasibility = {
        "run" : [],
        "feasibility" : [],
        "strict_feasibility" : [],
        "total_strict_feasibility" : []
    }

    epsilon1 = 0.01
    epsilon2 = 0.02

    for m in range(50):
        _start_index = m * n_loops
        _stop_index = _start_index + n_loops

        _analysis_set = small_box_bo.iloc[_start_index:_stop_index, 6:11]

        within_e1 = (_analysis_set["alpha"] <= target_alpha + epsilon1) & (_analysis_set["power"] >= target_power - epsilon2)

        within_e2_alpha = (_analysis_set["alpha"]>=target_alpha-epsilon1) & (_analysis_set["alpha"]<=target_alpha+epsilon1)
        within_e2_power = (_analysis_set["power"]>=target_power-epsilon2) & (_analysis_set["power"]<=target_power+epsilon2)

        within_e2 = within_e2_alpha & within_e2_power

        feasibility["run"].append(m+1)
        feasibility["feasibility"].append(np.mean(within_e1))
        feasibility["strict_feasibility"].append(np.mean(within_e2))
        feasibility["total_strict_feasibility"].append(np.sum(within_e2))

    pd.DataFrame(feasibility).describe()*100
    return


@app.cell
def _(n_loops, np, small_box_bo, tri_alpha, tri_max_ess, tri_obj):
    tri_diff = abs(0.05-tri_alpha)

    diffs = {
        "run" : [],
        "alpha_diff" : [],
        "max_ess_diff" : [],
        "obj_func_diff" : []
    }

    for _i in range(50):
        _start_index = _i * n_loops
        _stop_index = _start_index + n_loops

        _analysis_set = small_box_bo.iloc[_start_index:_stop_index, 6:11]

        diffs["run"].append(_i+1)

        diffs["alpha_diff"].append(np.sum(abs(_analysis_set["alpha"]-0.05) < tri_diff))

        diffs["max_ess_diff"].append(np.sum(_analysis_set["max_ess"] < tri_max_ess))

        diffs["obj_func_diff"].append(np.sum(_analysis_set["obj_func"] < tri_obj))
    return (diffs,)


@app.cell
def _(diffs, pd):
    pd.DataFrame(diffs).describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Design boundaries comparison
    """)
    return


@app.cell
def _(np):
    def best_bound_getter(data):
        lower=data.loc[data["obj_func"].idxmin(), ["upper1", "upper2","upper3"]].tolist()
        upper=data.loc[data["obj_func"].idxmin(), ["lower1", "lower2","upper3"]].tolist()
        obj_f= np.round(data.loc[data["obj_func"].idxmin(), "obj_func"], decimals=4)
        return lower, upper, obj_f

    return (best_bound_getter,)


@app.cell
def _(
    best_bound_getter,
    num_analyses,
    plt,
    small_box_bo,
    small_box_rand,
    small_box_sim_ann,
    tri,
):
    _fig, _ax = plt.subplots(nrows=1, ncols=3, figsize=(14,3), sharey=True)

    stages = [i+1 for i in range(num_analyses)]

    lower_rand, upper_rand, obj_f_rand = best_bound_getter(small_box_rand)
    lower_sim_ann, upper_sim_ann, obj_f_sim = best_bound_getter(small_box_sim_ann)
    lower_bo, upper_bo, obj_f_bo = best_bound_getter(small_box_bo)

    for _i in range(_ax.shape[0]):
        _ax[_i].set_xlabel("Trial stages")
        _ax[_i].set_xticks([1,2,3])
        _ax[_i].plot(stages, tri[0], color = "darkorange", label = "Tri bound")
        _ax[_i].plot(stages, tri[1], color = "darkorange")

    _ax[0].plot(stages, upper_rand, color = "purple", label = "Best bound")
    _ax[0].plot(stages, lower_rand, color = "purple")
    _ax[0].text(2.4,0, "$\\mathcal{L}$ = "+str(obj_f_rand))

    _ax[1].plot(stages, upper_sim_ann, color = "purple", label = "Best bound")
    _ax[1].plot(stages, lower_sim_ann, color = "purple")
    _ax[1].text(2.4,0, "$\\mathcal{L}$ = "+str(obj_f_sim))

    _ax[2].plot(stages, upper_bo, color = "purple", label = "Best bound")
    _ax[2].plot(stages, lower_bo, color = "purple")
    _ax[2].text(2.4,0, "$\\mathcal{L}$ = "+str(obj_f_bo))

    _ax[0].set_title("Random")
    _ax[1].set_title("Sim anneal")
    _ax[2].set_title("Bayes opt")

    _fig.suptitle("Small hyperrectangle", y=1.05)

    _ax[0].set_ylabel("$Z_k$ values")

    for _i in range(_ax.shape[0]):
        _ax[_i].legend(loc="upper right")

    plt.savefig("/tf/first_year_assessment/analysis/best_bounds_small_box.png", dpi=300, bbox_inches="tight")
    plt.show()
    return (stages,)


@app.cell
def _(
    best_bound_getter,
    plt,
    small_box_bo,
    small_box_rand,
    small_box_sim_ann,
    stages,
    tri,
):
    _fig, _ax = plt.subplots(nrows=1, ncols=3, figsize=(14,3), sharey=True)

    lower_rand1, upper_rand1, obj_f_rand1 = best_bound_getter(small_box_rand[small_box_rand["runs"]=="Run_24"])
    lower_sim_ann1, upper_sim_ann1, obj_f_sim1 = best_bound_getter(small_box_sim_ann[small_box_sim_ann["runs"]=="Run_9"])
    lower_bo1, upper_bo1, obj_f_bo1 = best_bound_getter(small_box_bo[small_box_bo["runs"]=="Run_6"])

    for _i in range(_ax.shape[0]):
        _ax[_i].set_xlabel("Trial stages")
        _ax[_i].set_xticks([1,2,3])
        _ax[_i].plot(stages, tri[0], color = "darkorange", label = "Tri bound")
        _ax[_i].plot(stages, tri[1], color = "darkorange")

    _ax[0].plot(stages, upper_rand1, color = "purple", label = "Best bound")
    _ax[0].plot(stages, lower_rand1, color = "purple")
    _ax[0].text(2.4,2.7, "$\\mathcal{L}$ = "+str(obj_f_rand1))

    _ax[1].plot(stages, upper_sim_ann1, color = "purple", label = "Best bound")
    _ax[1].plot(stages, lower_sim_ann1, color = "purple")
    _ax[1].text(2.4,2.7, "$\\mathcal{L}$ = "+str(obj_f_sim1))

    _ax[2].plot(stages, upper_bo1, color = "purple", label = "Best bound")
    _ax[2].plot(stages, lower_bo1, color = "purple")
    _ax[2].text(2.4,2.7, "$\\mathcal{L}$ = "+str(obj_f_bo1))

    _ax[0].set_title("Random $-$ Run 24")
    _ax[1].set_title("Sim anneal $-$ Run 9")
    _ax[2].set_title("Bayes opt $-$ Run 6")

    _fig.suptitle("Small hyperrectangle", y=1.05)

    _ax[0].set_ylabel("$Z_k$ values")

    for _i in range(_ax.shape[0]):
        _ax[_i].legend(loc="lower right")

    plt.savefig("/tf/first_year_assessment/analysis/rand_runs1_small_box.png", dpi=300, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(
    best_bound_getter,
    plt,
    small_box_bo,
    small_box_rand,
    small_box_sim_ann,
    stages,
    tri,
):
    _fig, _ax = plt.subplots(nrows=1, ncols=3, figsize=(14,3), sharey=True)

    lower_rand11, upper_rand11, obj_f_rand11 = best_bound_getter(small_box_rand[small_box_rand["runs"]=="Run_44"])
    lower_sim_ann11, upper_sim_ann11, obj_f_sim11 = best_bound_getter(small_box_sim_ann[small_box_sim_ann["runs"]=="Run_10"])
    lower_bo11, upper_bo11, obj_f_bo11 = best_bound_getter(small_box_bo[small_box_bo["runs"]=="Run_14"])

    for _i in range(_ax.shape[0]):
        _ax[_i].set_xlabel("Trial stages")
        _ax[_i].set_xticks([1,2,3])
        _ax[_i].plot(stages, tri[0], color = "darkorange", label = "Tri bound")
        _ax[_i].plot(stages, tri[1], color = "darkorange")

    _ax[0].plot(stages, upper_rand11, color = "purple", label = "Best bound")
    _ax[0].plot(stages, lower_rand11, color = "purple")
    _ax[0].text(2.4,0.5, "$\\mathcal{L}$ = "+str(obj_f_rand11))

    _ax[1].plot(stages, upper_sim_ann11, color = "purple", label = "Best bound")
    _ax[1].plot(stages, lower_sim_ann11, color = "purple")
    _ax[1].text(2.4,0.5, "$\\mathcal{L}$ = "+str(obj_f_sim11))

    _ax[2].plot(stages, upper_bo11, color = "purple", label = "Best bound")
    _ax[2].plot(stages, lower_bo11, color = "purple")
    _ax[2].text(2.4,0.5, "$\\mathcal{L}$ = "+str(obj_f_bo11))

    _ax[0].set_title("Random $-$ Run 44")
    _ax[1].set_title("Sim anneal $-$ Run 10")
    _ax[2].set_title("Bayes opt $-$ Run 14")

    _fig.suptitle("Small hyperrectangle", y=1.05)

    _ax[0].set_ylabel("$Z_k$ values")

    for _i in range(_ax.shape[0]):
        _ax[_i].legend(loc="upper right")

    plt.savefig("/tf/first_year_assessment/analysis/rand_runs2_small_box.png", dpi=300, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
