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
    from py_group_sequential_designs import feasibility_penalty as fp
    from py_group_sequential_designs import boundary_manipulations as fmt_bd
    from py_group_sequential_designs import function_to_minimize as fn_min
    from py_group_sequential_designs import generate_gpr_input as gen_input
    from py_group_sequential_designs import simulate as sim
    from py_group_sequential_designs import sample_size as ss

    return bd, fn_min, fp, sim, ss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Setup
    """)
    return


@app.cell
def _(ss):
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

        alpha_prime = trial_sim[1]
        beta_prime = 1-trial_sim[2]

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
def _(fn_min, fp, target_alpha, target_power):
    def our_obj_func(
        mu,
        power,
        alpha,
        row
    ):

        penalty = fp.smooth_penalty(
            mu = mu,
            power = target_power,
            alpha = target_alpha,
            beta_prime = 1-row["power"],
            alpha_prime = row["alpha"]
        )

        f_val = fn_min.function_to_minimize(
            max_ess_val = row["max_ess"]/mu,
            penalty = penalty
        )

        return f_val

    return (our_obj_func,)


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
    n_experiments = 10
    n_loops = 10000
    return n_experiments, n_loops


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Import the data
    """)
    return


@app.cell
def _(pd):
    wason_cpp = pd.read_csv("/tf/experiments_rand_simann_bo/wason_cpp_experiments/wason_sim_ann_cpp.csv")
    random_search_huge = pd.read_csv("/tf/experiments_rand_simann_bo/random_search_experiments/large_box_10by10k.csv")
    return random_search_huge, wason_cpp


@app.cell
def _(mu, our_obj_func, target_alpha, target_power, wason_cpp):
    # add a column of our objective function for comparison
    wason_cpp["our_obj_func"] = wason_cpp.apply(
        lambda row: our_obj_func(mu = mu, power = target_power, alpha = target_alpha, row = row),
        axis = 1
    )
    return


@app.cell
def _(n_experiments, n_loops, np):
    runs = np.concatenate([np.repeat(f"Run_{i+1}", n_loops) for i in range(n_experiments)])
    return (runs,)


@app.cell
def _(random_search_huge, runs, wason_cpp):
    wason_cpp["runs"] = runs
    random_search_huge["runs"] = runs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Random run summary
    """)
    return


@app.cell
def _(np):
    rng = np.random.default_rng(seed=2391807985)
    return (rng,)


@app.cell
def _(rng):
    rng.integers(low=1, high=11, size=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random search
    """)
    return


@app.cell
def _(np, random_search_huge):
    np.round(
        random_search_huge[random_search_huge["runs"] == "Run_7"][["alpha","power","sample_size","max_ess","obj_func"]].mean(),
        decimals = 2
    )
    return


@app.cell
def _(np, random_search_huge):
    np.round(
        random_search_huge[random_search_huge["runs"] == "Run_2"][["alpha","power","sample_size","max_ess","obj_func"]].mean(),
        decimals = 2
    )
    return


@app.cell
def _(np, random_search_huge):
    np.round(
        random_search_huge[random_search_huge["runs"] == "Run_5"][["alpha","power","sample_size","max_ess","obj_func"]].mean(),
        decimals = 2
    )
    return


@app.cell
def _(np, random_search_huge):
    np.round(
        random_search_huge[["alpha","power","sample_size","max_ess","obj_func"]].describe(),
        decimals = 2
    )
    return


@app.cell
def _(np, random_search_huge):
    np.round(
        random_search_huge[["alpha","power","sample_size","max_ess","obj_func"]].iloc[np.argmin(random_search_huge["obj_func"])],
        decimals = 2
    )
    return


@app.cell
def _(random_search_huge):
    random_search_huge.loc[10953]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wason data
    """)
    return


@app.cell
def _(np, wason_cpp):
    np.round(
        wason_cpp[wason_cpp["runs"] == "Run_2"][["alpha","power","sample_size","max_ess","obj_func"]].mean(),
        decimals = 2
    )
    return


@app.cell
def _(np, wason_cpp):
    np.round(
        wason_cpp[wason_cpp["runs"] == "Run_10"][["alpha","power","sample_size","max_ess","obj_func"]].mean(),
        decimals = 2
    )
    return


@app.cell
def _(np, wason_cpp):
    np.round(
        wason_cpp[wason_cpp["runs"] == "Run_8"][["alpha","power","sample_size","max_ess","obj_func"]].mean(),
        decimals = 2
    )
    return


@app.cell
def _(np, wason_cpp):
    np.round(
        wason_cpp[["alpha","power","sample_size","max_ess","obj_func"]].describe(),
        decimals = 2
    )
    return


@app.cell
def _(np, wason_cpp):
    np.round(
        wason_cpp[["alpha","power","sample_size","max_ess","obj_func"]].describe(),
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
def _(np, plt, random_search_huge, wason_cpp):
    fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(8,11), sharey="row")

    ax[0,0].violinplot(random_search_huge["alpha"], showextrema=False, showmedians=True)
    ax[0,0].text(0.95, 0.06, np.round(np.median(random_search_huge["alpha"]), decimals=2))

    ax[0,1].violinplot(wason_cpp["alpha"], showextrema=False, showmedians=True)
    ax[0,1].text(0.95, 0.06, np.round(np.median(wason_cpp["alpha"]), decimals=2))

    ax[1,0].violinplot(random_search_huge["power"], showextrema=False, showmedians=True)
    ax[1,0].text(0.965, 0.89, np.round(np.median(random_search_huge["power"]), decimals=2))

    ax[1,1].violinplot(wason_cpp["power"], showextrema=False, showmedians=True)
    ax[1,1].text(0.965, 0.89, np.round(np.median(wason_cpp["power"]), decimals=2))

    ax[2,0].violinplot(random_search_huge["sample_size"], showextrema=False, showmedians=True)
    ax[2,0].text(0.97, 56, np.round(np.median(random_search_huge["sample_size"]).astype(int), decimals=0))

    ax[2,1].violinplot(wason_cpp["sample_size"], showextrema=False, showmedians=True)
    ax[2,1].text(0.97, 61, np.round(np.median(wason_cpp["sample_size"]).astype(int), decimals=0))

    ax[3,0].violinplot(random_search_huge["max_ess"], showextrema=False, showmedians=True)
    ax[3,0].text(0.97, 153, np.round(np.median(random_search_huge["max_ess"]).astype(int), decimals=0))

    ax[3,1].violinplot(wason_cpp["max_ess"], showextrema=False, showmedians=True)
    ax[3,1].text(0.96, 165, np.round(np.median(wason_cpp["max_ess"])).astype(int))

    ax[4,1].violinplot(wason_cpp["obj_func"], showextrema=False, showmedians=True)
    ax[4,1].text(0.985, 9, np.round(np.median(wason_cpp["obj_func"])).astype(int))

    ax[5,0].violinplot(random_search_huge["obj_func"], showextrema=False, showmedians=True)
    ax[5,0].text(0.985, 11, np.round(np.median(random_search_huge["obj_func"])).astype(int))

    ax[5,1].violinplot(wason_cpp["our_obj_func"], showextrema=False, showmedians=True)
    ax[5,1].text(0.985, 9, np.round(np.median(wason_cpp["our_obj_func"])).astype(int))

    ax[0,0].set_title("Random")
    ax[0,1].set_title("Wason")

    ax[0,0].set_ylabel("$\\alpha'$")
    ax[1,0].set_ylabel("$1-\\beta'$")
    ax[2,0].set_ylabel("Sample size")
    ax[3,0].set_ylabel("Max ESS")
    ax[4,0].set_ylabel("Wason loss")
    ax[5,0].set_ylabel("Our loss")

    for a in ax.flat:
        a.set_xticks([])

    # plt.savefig("/tf/first_year_assessment/analysis/violin_plots_large_box.png", dpi=300, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Best indices
    """)
    return


@app.cell
def _(n_experiments, n_loops, np, random_search_huge):
    best_indices = []
    for ell in range(n_experiments):
        _start_index = ell * n_loops
        _stop_index = _start_index + n_loops

        _analysis_set = random_search_huge.iloc[_start_index:_stop_index, 7:12]
        best_index = np.argmin(_analysis_set['obj_func'])
        best_indices.append(best_index)

    print(f"Minimum best index: {np.min(best_indices)+1}")
    print(f"Maximum best index: {np.max(best_indices)+1}")
    print(f"Average best index: {np.mean(best_indices)+1}")
    print(f"Median best index: {np.median(best_indices)+1}")
    return


@app.cell
def _(n_experiments, n_loops, np, wason_cpp):
    _best_indices = []
    for _ell in range(n_experiments):
        _start_index = _ell * n_loops
        _stop_index = _start_index + n_loops

        _analysis_set = wason_cpp.iloc[_start_index:_stop_index, 8:13]
        _best_index = np.argmin(_analysis_set['obj_func'])
        _best_indices.append(_best_index)

    print(f"Minimum best index: {np.min(_best_indices)+1}")
    print(f"Maximum best index: {np.max(_best_indices)+1}")
    print(f"Average best index: {np.mean(_best_indices)+1}")
    print(f"Median best index: {np.median(_best_indices)+1}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Feasibility and strict feasibility
    """)
    return


@app.cell
def _(
    n_experiments,
    n_loops,
    np,
    pd,
    random_search_huge,
    target_alpha,
    target_power,
):
    feasibility = {
        "run" : [],
        "feasibility" : [],
        "strict_feasibility" : [],
        "total_strict_feasibility" : []
    }

    epsilon1 = 0.01
    epsilon2 = 0.02

    for m in range(n_experiments):
        _start_index = m * n_loops
        _stop_index = _start_index + n_loops

        _analysis_set = random_search_huge.iloc[_start_index:_stop_index, 6:11]

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
def _(
    n_experiments,
    n_loops,
    np,
    random_search_huge,
    tri_alpha,
    tri_max_ess,
    tri_obj,
):
    tri_diff = abs(0.05-tri_alpha)

    diffs = {
        "run" : [],
        "alpha_diff" : [],
        "max_ess_diff" : [],
        "obj_func_diff" : []
    }

    for _i in range(n_experiments):
        _start_index = _i * n_loops
        _stop_index = _start_index + n_loops

        _analysis_set = random_search_huge.iloc[_start_index:_stop_index, 7:12]

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
def _(random_search_huge):
    random_search_huge.loc[random_search_huge["obj_func"].idxmin(), ["alpha", "power"]]
    return


@app.cell
def _(wason_cpp):
    wason_cpp.loc[wason_cpp["obj_func"].idxmin(), ["alpha", "power"]]
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
def _(np, random_search_huge):
    np.min(random_search_huge["obj_func"])
    return


@app.cell
def _(
    best_bound_getter,
    np,
    num_analyses,
    plt,
    random_search_huge,
    tri,
    wason_cpp,
):
    _fig, _ax = plt.subplots(nrows=1, ncols=2, figsize=(14,3), sharey=True)

    stages = [i+1 for i in range(num_analyses)]

    lower_rand, upper_rand, obj_f_rand = best_bound_getter(random_search_huge)
    lower_sim_ann, upper_sim_ann, obj_f_sim = best_bound_getter(wason_cpp)

    for _i in range(_ax.shape[0]):
        _ax[_i].set_xlabel("Trial stages")
        _ax[_i].set_xticks([1,2,3])
        _ax[_i].plot(stages, tri[0], color = "darkorange", label = "Tri bound")
        _ax[_i].plot(stages, tri[1], color = "darkorange")

    _ax[0].plot(stages, upper_rand, color = "purple", label = "Best bound")
    _ax[0].plot(stages, lower_rand, color = "purple")
    _ax[0].text(2.4,-0.5, "$\\mathcal{L}$ = "+str(obj_f_rand))

    _ax[1].plot(stages, upper_sim_ann, color = "purple", label = "Best bound")
    _ax[1].plot(stages, lower_sim_ann, color = "purple")
    _ax[1].text(2.4,-0.5, "$\\mathcal{L}$ = "+str(obj_f_sim))
    _ax[1].text(2.4,-1, "our loss = "+str(np.round(wason_cpp.loc[wason_cpp["obj_func"].idxmin(), "our_obj_func"], decimals=4)))

    _ax[0].set_title("Random")
    _ax[1].set_title("Sim anneal")

    _fig.suptitle("Large hyperrectangle", y=1.05)

    _ax[0].set_ylabel("$Z_k$ values")

    for _i in range(_ax.shape[0]):
        _ax[_i].legend(loc="upper right")

    # plt.savefig("/tf/first_year_assessment/analysis/best_bounds_large_box.png", dpi=300, bbox_inches="tight")
    plt.show()
    return (stages,)


@app.cell
def _(best_bound_getter, np, plt, random_search_huge, stages, tri, wason_cpp):
    _fig, _ax = plt.subplots(nrows=1, ncols=2, figsize=(14,3), sharey=True)

    lower_rand1, upper_rand1, obj_f_rand1 = best_bound_getter(random_search_huge[random_search_huge["runs"]=="Run_3"])
    lower_sim_ann1, upper_sim_ann1, obj_f_sim1 = best_bound_getter(wason_cpp[wason_cpp["runs"]=="Run_9"])

    for _i in range(_ax.shape[0]):
        _ax[_i].set_xlabel("Trial stages")
        _ax[_i].set_xticks([1,2,3])
        _ax[_i].plot(stages, tri[0], color = "darkorange", label = "Tri bound")
        _ax[_i].plot(stages, tri[1], color = "darkorange")

    _ax[0].plot(stages, upper_rand1, color = "purple", label = "Best bound")
    _ax[0].plot(stages, lower_rand1, color = "purple")
    _ax[0].text(2.4,-0.5, "$\\mathcal{L}$ = "+str(obj_f_rand1))

    _ax[1].plot(stages, upper_sim_ann1, color = "purple", label = "Best bound")
    _ax[1].plot(stages, lower_sim_ann1, color = "purple")
    _ax[1].text(2.4,-0.5, "$\\mathcal{L}$ = "+str(obj_f_sim1))
    _ax[1].text(2.4,-1, "our loss = "+str(np.round(wason_cpp[wason_cpp["runs"]=="Run_9"].loc[wason_cpp[wason_cpp["runs"]=="Run_9"]["obj_func"].idxmin(), "our_obj_func"], decimals=4)))

    _ax[0].set_title("Random $-$ Run 2")
    _ax[1].set_title("Sim anneal $-$ Run 9")

    _fig.suptitle("Large hyperrectangle", y=1.05)

    _ax[0].set_ylabel("$Z_k$ values")

    for _i in range(_ax.shape[0]):
        _ax[_i].legend(loc="upper right")

    # plt.savefig("/tf/first_year_assessment/analysis/rand_runs1_large_box.png", dpi=300, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
