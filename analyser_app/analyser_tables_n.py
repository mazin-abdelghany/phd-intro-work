import marimo

__generated_with = "0.23.16"
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Experimental Setup
    """)
    return


@app.cell
def _(mo):
    num_methods = mo.ui.number(label="Number of methods/files to compare = ", value=2, start=1)

    mo.vstack([num_methods])
    return (num_methods,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dynamic Data Files & Labels
    """)
    return


@app.cell
def _(mo, num_methods):
    # Generates dynamic text inputs based on the number of selected methods
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
    # Dynamic file importer processing arbitrary dataframes
    datasets = {}
    for i, label_ui in enumerate(labels.elements):
        label_value = label_ui.value if label_ui.value else f"Method_{i+1}"
        try:
            # Safely fetch the file from index if path exists
            filepath = file_browser.path(index=i)
            if filepath:
                _df = pd.read_csv(filepath)
                datasets[label_value] = _df.iloc[:, 1:] # drop first column
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
        n_loops = 11000

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
        if 10 <= exp <= 50:
            return exp, run

    # Otherwise it must be a 1-digit experiment (1-9)
    exp = int(s[:1])
    run = int(s[1:])
    if 1 <= exp <= 9:
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
def _(mo, n_loops_dict):
    mo.Html("<br>".join(
        f"There are {n_loops_dict[key]} experiments in {key}"
        for key in n_loops_dict
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
    round_to = mo.ui.number(label = "Round digits", start=1, value=3)
    mo.vstack([seed_for_runs, round_to])
    return round_to, seed_for_runs


@app.cell
def _(np, seed_for_runs):
    rng = np.random.default_rng(seed=seed_for_runs.value)
    return (rng,)


@app.cell
def _(n_experiments_dict, rng):
    # compare runs with the high of the rng at the min of number of experiments in methods
    runs_to_compare = rng.integers(low=1, high=min(n_experiments_dict.values()), size=9)
    return (runs_to_compare,)


@app.cell
def _(runs_to_compare):
    column_runs_compare = ["Run_"+str(run) for run in runs_to_compare]
    return (column_runs_compare,)


@app.cell
def _(mo, runs_to_compare):
    mo.Html(f"We will be summarizing runs: {runs_to_compare}")
    return


@app.cell
def _(column_runs_compare, datasets, pd, wason_included):
    if wason_included.value:
        summary_columns = ["alpha","power","sample_size","max_ess","obj_func"]
        index_for_table = ["alpha_prime","beta_prime","sample_size","max_ess","loss"]
    else:
        summary_columns = ["alpha","power","sample_size","max_ess","obj_func", "execute_time"]
        index_for_table = ["alpha_prime","beta_prime","sample_size","max_ess","loss", "execute_time"]

    # Process an arbitrary dictionary of summaries
    summaries = {}

    for _label, _df in datasets.items():
        table_data = {
            "grand_mean" : _df[summary_columns].mean().to_numpy(),
            "grand_std" : _df[summary_columns].std().to_numpy()
        }

        # Safely extract comparing slice runs dynamically
        for idx, col_run in enumerate(column_runs_compare[:3]): 
            run_data = _df[_df["runs"] == col_run]
            if not run_data.empty:
                table_data[col_run] = run_data[summary_columns].mean().to_numpy()
            else:
                table_data[col_run] = [0.0] * 6

        # Best structures
        best_bounds = _df[_df["obj_func"] == _df["obj_func"].min()]
        if not best_bounds.empty:
            table_data["best_design"] = best_bounds[summary_columns].iloc[0].to_numpy()

        constrained = _df[_df["alpha"] <= 0.05]
        if not constrained.empty:
            best_constrained = constrained[constrained["obj_func"] == constrained["obj_func"].min()]
            table_data["best_constrained_design"] = best_constrained[summary_columns].iloc[0].to_numpy()
        else:
            table_data["best_constrained_design"] = [0.0] * 6

        # Reconstruct structured frame
        summaries[_label] = pd.DataFrame(
            table_data, 
            index=index_for_table
        )
    return index_for_table, summaries, summary_columns


@app.cell
def _(mo, round_to, summaries):
    # Renders an arbitrary list of summary blocks markdown grids dynamically 
    outputs = []
    for _label, summary_df in summaries.items():
        outputs.append(mo.md(f"### Summary Table of: **{_label}**"))
        outputs.append(summary_df.transpose().round(round_to.value))

    mo.vstack(outputs)
    return


@app.cell
def _(datasets, index_for_table, pd, summary_columns):
    # Process an arbitrary dictionary of summaries
    constrained_summaries = {}

    for _label, _df in datasets.items():

        _min_obj_func_idx = _df.groupby("runs")["obj_func"].idxmin()
        _constrained_df = _df.loc[_min_obj_func_idx]

        constr_table_data = {
            "grand_mean" : ( 
                _constrained_df[summary_columns]
                    .mean()
                    .to_numpy()
             ),
            "grand_std" : (
                _constrained_df[summary_columns]
                    .std()
                    .to_numpy()
             )
        }

        constrained_summaries[_label] = pd.DataFrame(
            constr_table_data,
            index=index_for_table
        )
    return (constrained_summaries,)


@app.cell
def _(constrained_summaries, mo, round_to):
    # Renders an arbitrary list of summary blocks markdown grids dynamically 
    _outputs = []
    for _label, _summary_df in constrained_summaries.items():
        _outputs.append(mo.md(f"### Summary Table of Designs with lowest loss: **{_label}**"))
        _outputs.append(_summary_df.transpose().round(round_to.value))

    mo.vstack(_outputs)
    return


@app.cell
def _(datasets, n_experiments_dict, n_loops_dict, np):
    # Dynamic calculations of best indices
    best_indices_metrics = {}

    for _label, _df in datasets.items():
        best_idx_list = []
        for _i in range(n_experiments_dict[_label]):
            _start_idx = _i * n_loops_dict[_label]
            _stop_idx = _start_idx + n_loops_dict[_label]
            _analysis_set = _df.loc[_start_idx:_stop_idx]
            if not _analysis_set.empty:
                best_idx_list.append(np.argmin(_analysis_set["obj_func"]))

        if best_idx_list:
            best_indices_metrics[_label] = {
                "min_best_idx": np.min(best_idx_list) + 1,
                "max_best_idx": np.max(best_idx_list) + 1,
                "mean_best_idx": np.mean(best_idx_list) + 1,
                "median_best_idx": np.median(best_idx_list) + 1
            }
    return (best_indices_metrics,)


@app.cell
def _(best_indices_metrics, mo, pd):
    if best_indices_metrics:
        best_indices_df = pd.DataFrame(best_indices_metrics).transpose()

    mo.vstack([mo.md("## Best Indices Overview"), best_indices_df.round()])
    return


@app.cell
def _(mo):
    epsilon1 = mo.ui.number(start=0, stop=1, label="Epsilon 1", step=0.005, value=0.01)
    epsilon2 = mo.ui.number(start=0, stop=1, label="Epsilon 2", step=0.005, value=0.01)
    mo.vstack([epsilon1, epsilon2])
    return epsilon1, epsilon2


@app.cell
def _(
    datasets,
    epsilon1,
    epsilon2,
    n_experiments_dict,
    n_loops_dict,
    np,
    target_alpha,
    target_power,
):
    feasibility_summary = {}

    for _label, _df in datasets.items():
        feas_list = []
        strict_feas_list = []

        for j in range(n_experiments_dict[_label]):
            _start_idx = j * n_loops_dict[_label]
            _stop_idx = _start_idx + n_loops_dict[_label]
            _analysis_set = _df.loc[_start_idx:_stop_idx]

            if not _analysis_set.empty:
                alpha_feas = _analysis_set["alpha"] <= target_alpha + epsilon1.value
                power_feas = _analysis_set["power"] >= target_power - epsilon2.value
                feas_list.append(np.mean(alpha_feas & power_feas))

                within_e1_lower = target_alpha - epsilon1.value <= _analysis_set["alpha"]
                within_e1_upper = _analysis_set["alpha"] <= target_alpha + epsilon1.value
                strict_alpha = within_e1_lower & within_e1_upper

                within_e2_lower = target_power - epsilon2.value <= _analysis_set["power"]
                within_e2_upper = _analysis_set["power"] <= target_power + epsilon2.value
                strict_power = within_e2_lower & within_e2_upper

                strict_feas_list.append(np.mean(strict_alpha & strict_power))

        if feas_list:
            feasibility_summary[_label] = {
                "mean_feasibility_%": np.mean(feas_list) * 100,
                "mean_strict_feasibility_%": np.mean(strict_feas_list) * 100
            }
    return (feasibility_summary,)


@app.cell
def _(feasibility_summary, mo, pd, round_to):
    if feasibility_summary:
        df_feas = pd.DataFrame(feasibility_summary).round(round_to.value)

    mo.vstack([mo.md("## Feasibility Statistics"), df_feas])
    return


@app.cell
def _(
    datasets,
    n_experiments_dict,
    n_loops_dict,
    np,
    tri_alpha,
    tri_max_ess,
    tri_obj,
):
    tri_diff = abs(0.05 - tri_alpha)
    tri_comparisons = {}

    for _label, _df in datasets.items():
        n_diff_less = []
        n_ess_less = []
        n_loss_less = []

        for k in range(n_experiments_dict[_label]):
            start_idx = k * n_loops_dict[_label]
            stop_idx = start_idx + n_loops_dict[_label]
            analysis_set = _df.loc[start_idx:stop_idx]

            if not analysis_set.empty:
                n_diff_less.append(np.sum(abs(analysis_set["alpha"] - 0.05) < tri_diff))
                n_ess_less.append(np.sum(analysis_set["max_ess"] < tri_max_ess))
                n_loss_less.append(np.sum(analysis_set["obj_func"] < tri_obj))

        if n_diff_less:
            tri_comparisons[_label] = {
                "median_n_alpha_closer": np.median(n_diff_less),
                "median_n_ess_closer": np.median(n_ess_less),
                "max_n_loss_closer": np.max(n_loss_less)
            }
    return (tri_comparisons,)


@app.cell
def _(mo, pd, tri_comparisons):
    if tri_comparisons:
        df_tri = pd.DataFrame(tri_comparisons)

    mo.vstack([mo.md("## Difference Between Triangular Bounds"), df_tri])
    return


if __name__ == "__main__":
    app.run()
