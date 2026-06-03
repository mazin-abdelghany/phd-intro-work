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

    return np, pd


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
    return tri_alpha, tri_max_ess, tri_obj


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
    runs_to_compare = rng.integers(low=1, high=11, size=9)
    return (runs_to_compare,)


@app.cell
def _(mo, runs_to_compare):
    mo.Html(f"We will be summarizing runs: {runs_to_compare}")
    return


@app.cell
def _(runs_to_compare):
    column_runs_compare = ["Run_"+str(run) for run in runs_to_compare]
    return (column_runs_compare,)


@app.cell
def _(column_runs_compare, data_a):
    table_a = {
        column_runs_compare[0] : [],
        column_runs_compare[1] : [],
        column_runs_compare[2] : [],
        "grand_mean" : [],
        "grand_std" : [],
        "best_design" : [],
        "best_constrained_design" : []
    }

    table_a[column_runs_compare[0]] = data_a[
        data_a["runs"]==column_runs_compare[0]
        ][["alpha","power","sample_size","max_ess","obj_func"]].mean().to_numpy()

    table_a[column_runs_compare[1]] = data_a[
        data_a["runs"]==column_runs_compare[1]
        ][["alpha","power","sample_size","max_ess","obj_func"]].mean().to_numpy()

    table_a[column_runs_compare[2]] = data_a[
        data_a["runs"]==column_runs_compare[2]
        ][["alpha","power","sample_size","max_ess","obj_func"]].mean().to_numpy()

    table_a["grand_mean"] = data_a[["alpha","power","sample_size","max_ess","obj_func"]].mean().to_numpy()

    table_a["grand_std"] = data_a[["alpha","power","sample_size","max_ess","obj_func"]].std().to_numpy()

    best_bounds_a = data_a[data_a["obj_func"]==data_a["obj_func"].min()]

    table_a["best_design"] = best_bounds_a[["alpha","power","sample_size","max_ess","obj_func"]].to_numpy().flatten()

    constrained_a = data_a[data_a["alpha"] <= 0.05]

    best_contrained_bounds_a = constrained_a[constrained_a["obj_func"]==constrained_a["obj_func"].min()]

    table_a["best_constrained_design"] = best_contrained_bounds_a[["alpha","power","sample_size","max_ess","obj_func"]].to_numpy().flatten()
    return (table_a,)


@app.cell
def _(pd, table_a):
    summary_a = pd.DataFrame(table_a, index=["alpha_prime","beta_prime","sample_size","max_ess","loss"])
    return (summary_a,)


@app.cell
def _(mo):
    round_to = mo.ui.number(label = "Round digits", start=1)
    mo.vstack(
        [round_to]
    )
    return (round_to,)


@app.cell(hide_code=True)
def _(label_a, mo):
    mo.md(rf"""
    # Summary of {label_a.value}
    """)
    return


@app.cell
def _(round_to, summary_a):
    summary_a.transpose().round(round_to.value)
    return


@app.cell
def _(column_runs_compare, data_b):
    table_b = {
        column_runs_compare[3] : [],
        column_runs_compare[4] : [],
        column_runs_compare[5] : [],
        "grand_mean" : [],
        "grand_std" : [],
        "best_design" : [],
        "best_constrained_design" : []
    }

    table_b[column_runs_compare[3]] = data_b[
        data_b["runs"]==column_runs_compare[3]
        ][["alpha","power","sample_size","max_ess","obj_func"]].mean().to_numpy()

    table_b[column_runs_compare[4]] = data_b[
        data_b["runs"]==column_runs_compare[4]
        ][["alpha","power","sample_size","max_ess","obj_func"]].mean().to_numpy()

    table_b[column_runs_compare[5]] = data_b[
        data_b["runs"]==column_runs_compare[5]
        ][["alpha","power","sample_size","max_ess","obj_func"]].mean().to_numpy()

    table_b["grand_mean"] = data_b[["alpha","power","sample_size","max_ess","obj_func"]].mean().to_numpy()

    table_b["grand_std"] = data_b[["alpha","power","sample_size","max_ess","obj_func"]].std().to_numpy()

    best_bounds_b = data_b[data_b["obj_func"]==data_b["obj_func"].min()]

    table_b["best_design"] = best_bounds_b[["alpha","power","sample_size","max_ess","obj_func"]].to_numpy().flatten()

    constrained_b = data_b[data_b["alpha"] <= 0.05]

    best_contrained_bounds_b = constrained_b[constrained_b["obj_func"]==constrained_b["obj_func"].min()]

    table_b["best_constrained_design"] = best_contrained_bounds_b[["alpha","power","sample_size","max_ess","obj_func"]].to_numpy().flatten()
    return (table_b,)


@app.cell
def _(pd, table_b):
    summary_b = pd.DataFrame(table_b, index=["alpha_prime","beta_prime","sample_size","max_ess","loss"])
    return (summary_b,)


@app.cell(hide_code=True)
def _(label_b, mo):
    mo.md(rf"""
    # Summary of {label_b.value}
    """)
    return


@app.cell
def _(round_to, summary_b):
    summary_b.transpose().round(round_to.value)
    return


@app.cell
def _(data_a, data_b, n_experiments, n_loops, np):
    best_index_a = []
    best_index_b = []

    for i in range(n_experiments.value):
        start_index = i * n_loops.value
        stop_index = start_index + n_loops.value

        analysis_set_a = data_a.loc[start_index:stop_index]
        best_index_a.append(np.argmin(analysis_set_a["obj_func"]))

        analysis_set_b = data_b.loc[start_index:stop_index]
        best_index_b.append(np.argmin(analysis_set_b["obj_func"]))
    return best_index_a, best_index_b


@app.cell
def _(best_index_a, best_index_b, np):
    best_indices = {
        "min_best_idx" : [np.min(best_index_a)+1, np.min(best_index_b)+1],
        "max_best_idx" : [np.max(best_index_a)+1, np.max(best_index_b)+1],
        "mean_best_idx": [np.mean(best_index_a)+1,np.mean(best_index_b)+1],
        "median_best_idx":[np.median(best_index_a)+1,np.median(best_index_b)+1]
    }
    return (best_indices,)


@app.cell
def _(best_indices, label_a, label_b, pd):
    best_indices_df = pd.DataFrame(best_indices, index=[label_a.value, label_b.value])
    return (best_indices_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Best indices
    """)
    return


@app.cell
def _(best_indices_df):
    best_indices_df.transpose().round()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Feasibility
    """)
    return


@app.cell
def _(mo):
    epsilon1 = mo.ui.number(start=0, stop=1, label="Epsilon 1", step=0.005)
    return (epsilon1,)


@app.cell
def _(mo):
    epsilon2 = mo.ui.number(start=0, stop=1, label="Epsilon 2", step=0.005)
    return (epsilon2,)


@app.cell
def _(epsilon1, epsilon2, mo):
    mo.vstack([epsilon1, epsilon2])
    return


@app.cell(hide_code=True)
def _(epsilon1, epsilon2, mo):
    mo.md(rf"""
    Feasibility is defined as: 
    - $\alpha'\le$ ($\alpha$ + {epsilon1.value}) **AND**
    - $1-\beta'\ge$  ($\beta$ - {epsilon2.value})

    Strict feasibility is defined as: 
    - $\alpha$ - {epsilon1.value} $\le\alpha'\le$ $\alpha$ + {epsilon1.value}
    - $1-\beta$ - {epsilon2.value} $\le 1-\beta' \le$ $1-\beta$ + {epsilon2.value}
    """)
    return


@app.cell
def _(
    data_a,
    data_b,
    epsilon1,
    epsilon2,
    n_experiments,
    n_loops,
    np,
    target_alpha,
    target_power,
):
    feasibility_a = []
    feasibility_b = []

    strict_feas_a = []
    strict_feas_b = []

    total_strict_feas_a = []
    total_strict_feas_b = []

    for j in range(n_experiments.value):
        start_index1 = j * n_loops.value
        stop_index1 = start_index1 + n_loops.value

        analysis_set_a1 = data_a.loc[start_index1:stop_index1]
        analysis_set_b1 = data_b.loc[start_index1:stop_index1]

        # feasibility
        alpha_feas_a = analysis_set_a1["alpha"] <= target_alpha + epsilon1.value
        power_feas_a = analysis_set_a1["power"] >= target_power - epsilon2.value

        feasibility_a.append(
            np.mean(alpha_feas_a & power_feas_a)
        )

        alpha_feas_b = analysis_set_b1["alpha"] <= target_alpha + epsilon1.value
        power_feas_b = analysis_set_b1["power"] >= target_power - epsilon2.value

        feasibility_b.append(
            np.mean(alpha_feas_b & power_feas_b)
        )

        # strict feasibility
        within_e1_alpha_lower = target_alpha - epsilon1.value <= analysis_set_a1["alpha"]
        within_e1_alpha_upper = analysis_set_a1["alpha"] <= target_alpha+epsilon1.value

        strict_feas_alpha_a = within_e1_alpha_lower & within_e1_alpha_upper

        within_e2_power_lower = target_power - epsilon2.value <= analysis_set_a1["power"]
        within_e2_power_upper = analysis_set_a1["power"] <= target_power+epsilon2.value

        strict_feas_power_a = within_e2_power_lower & within_e2_power_upper

        strict_feas_a.append(
            np.mean(strict_feas_alpha_a & strict_feas_power_a)
        )

        within_e1_alpha_lower_b = target_alpha-epsilon1.value <= analysis_set_b1["alpha"]
        within_e1_alpha_upper_b = analysis_set_b1["alpha"] <= target_alpha+epsilon1.value

        strict_feas_alpha_b = within_e1_alpha_lower_b & within_e1_alpha_upper_b

        within_e2_power_lower_b = target_power-epsilon2.value <= analysis_set_b1["power"]
        within_e2_power_upper_b = analysis_set_b1["power"] <= target_power+epsilon2.value

        strict_feas_power_b = within_e2_power_lower_b & within_e2_power_upper_b

        strict_feas_b.append(
            np.mean(strict_feas_alpha_b & strict_feas_power_b)
        )

        # total strict feasibility
        total_strict_feas_a.append(
            np.sum(strict_feas_alpha_a & strict_feas_power_a)
        )

        total_strict_feas_b.append(
            np.sum(strict_feas_alpha_b & strict_feas_power_b)
        )
    return feasibility_a, feasibility_b, strict_feas_a, strict_feas_b


@app.cell
def _(feasibility_a, feasibility_b, np, strict_feas_a, strict_feas_b):
    feasibility = {
        "mean_feasibility_%" : [np.mean(feasibility_a)*100, np.mean(feasibility_b)*100],
        "mean_strict_feasibility_%" : [np.mean(strict_feas_a)*100, np.mean(strict_feas_b)*100],
    }
    return (feasibility,)


@app.cell
def _(feasibility, label_a, label_b, pd, round_to):
    pd.DataFrame(feasibility, index=[label_a.value, label_b.value]).transpose().round(round_to.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Difference between triangular
    """)
    return


@app.cell
def _(
    data_a,
    data_b,
    n_experiments,
    n_loops,
    np,
    tri_alpha,
    tri_max_ess,
    tri_obj,
):
    tri_diff = abs(0.05 - tri_alpha)

    n_diff_less_than_tri_a = []
    n_ess_less_than_tri_a = []
    n_loss_less_than_tri_a = []

    n_diff_less_than_tri_b = []
    n_ess_less_than_tri_b = []
    n_loss_less_than_tri_b = []

    for k in range(n_experiments.value):
        start_index2 = k * n_loops.value
        stop_index2 = start_index2 + n_loops.value

        analysis_set_a2 = data_a.loc[start_index2:stop_index2]
        analysis_set_b2 = data_b.loc[start_index2:stop_index2]

        # alpha < tri
        n_diff_less_than_tri_a.append(
            np.sum(abs(analysis_set_a2["alpha"] - 0.05) < tri_diff)
        )

        n_diff_less_than_tri_b.append(
            np.sum(abs(analysis_set_b2["alpha"] - 0.05) < tri_diff)
        )

        # max_ess < tri
        n_ess_less_than_tri_a.append(
            np.sum(analysis_set_a2["max_ess"] < tri_max_ess)
        )

        n_ess_less_than_tri_b.append(
            np.sum(analysis_set_b2["max_ess"] < tri_max_ess)
        )

        # obj func < tri
        n_loss_less_than_tri_a.append(
            np.sum(analysis_set_a2["obj_func"] < tri_obj)
        )

        n_loss_less_than_tri_b.append(
            np.sum(analysis_set_b2["obj_func"] < tri_obj)
        )
    return (
        n_diff_less_than_tri_a,
        n_diff_less_than_tri_b,
        n_ess_less_than_tri_a,
        n_ess_less_than_tri_b,
        n_loss_less_than_tri_a,
        n_loss_less_than_tri_b,
    )


@app.cell
def _(
    n_diff_less_than_tri_a,
    n_diff_less_than_tri_b,
    n_ess_less_than_tri_a,
    n_ess_less_than_tri_b,
    n_loss_less_than_tri_a,
    n_loss_less_than_tri_b,
    np,
):
    tri_comp = {
        "median_n_alpha_closer" : [np.median(n_diff_less_than_tri_a), np.median(n_diff_less_than_tri_b)],
        "median_n_ess_closer" : [np.median(n_ess_less_than_tri_a), np.median(n_ess_less_than_tri_b)],
        "max_n_loss_closer" : [np.max(n_loss_less_than_tri_a), np.max(n_loss_less_than_tri_b)]
    }
    return (tri_comp,)


@app.cell
def _(label_a, label_b, pd, tri_comp):
    pd.DataFrame(tri_comp, index=[label_a.value, label_b.value]).transpose()
    return


if __name__ == "__main__":
    app.run()
