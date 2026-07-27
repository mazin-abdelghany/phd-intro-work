import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


@app.cell
def _():
    import time
    import gc
    import scipy.stats as stats
    import numpy as np
    import pandas as pd

    return gc, np, pd, time


@app.cell
def _():
    from ax.service.ax_client import AxClient, ObjectiveProperties

    return AxClient, ObjectiveProperties


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

    return bd, fmt_bd, fn_min, fp, sim, ss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Trial Design Settings
    """)
    return


@app.cell
def _(mo):
    num_analyses = mo.ui.number(label="Number of analyses = ", value=5, start=1)

    mo.vstack([num_analyses])
    return (num_analyses,)


@app.cell
def _(num_analyses, ss):
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

    print(f"We are running an experiment with a trial design with {num_analyses.value} stages, with:\na target alpha of {target_alpha},\na target power of {target_power},\na null hypothesis of {delta0},\nan alternative hypothesis of {delta1},\nand an assumed variance of {sigma2}\n")
    print(f"Single-stage sample size mu = {mu:.2f}")
    return delta0, delta1, mu, sigma2, target_alpha, target_power


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reverse parameterisation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    BO vector: $\{c, \Delta u_3, \Delta \ell_3, \Delta u_2, \Delta \ell_2\}$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Objective function
    """)
    return


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
    fmt_bd,
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
        n_analyses = num_analyses.value,
        alpha = target_alpha,
        delta = delta1
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

    tri_params = fmt_bd.boundaries_to_reverse(
        upper_bounds = tri[0],
        lower_bounds = tri[1]
    )

    c0 = tri_params[0]

    print(f"Original triangular params: {np.round(np.concatenate((tri[0], tri[1])), 4)}")
    print(f"Reparameterized triangular params: {np.round(tri_params, 4)}")
    print(f"Meeting point c0 = {c0:.4f}\n")
    print(f"Triangular benchmark objective: {tri_obj:.4f}")
    print(f"Triangular alpha: {tri_alpha:.4f}")
    print(f"Triangular delta alpha: {abs(0.05-tri_alpha):.4f}")
    print(f"Triangular power: {tri_power:.4f}")
    print(f"Triangular delta beta: {abs(0.9-tri_power):.4f}")
    print(f"Triangular sample size: {tri_n_patients:.1f}")
    print(f"Triangular max ESS: {tri_max_ess:.1f}")
    return c0, tri_params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Quantities to follow

    Our design goals are $\alpha=$ {target_alpha} and $\beta=$ {1-target_power}. We will assess:

    - The boundary values and corresponding $\alpha'$, $\beta'$, $n$, and maximum expected sample size for the optimal design.
    - Feasibility: proportion of designs where
    \[
        \alpha' \le \alpha + \epsilon_1 \qquad (1-\beta') \ge (1-\beta) - \epsilon_1
    \]
    - Strict feasibility: proportion of designs where
    \[
        \alpha-\epsilon_2 \le \alpha' \le \alpha + \epsilon_2 \qquad (1-\beta)-\epsilon_2 \le (1-\beta') \le (1-\beta) + \epsilon_2
    \]
    - Does the best overall design $D^\star$ have a maximum expected sample size that is smaller than the triangular design?
    - Does the best overall design $D^\star$ have $\alpha'$ that is closer to the target than the triangular design?
    - The best overall $\mathcal{L}(\cdot)$ value obtained and its associated index (i.e., loop number at which the optimum was reached).
    - Is the best overall $\mathcal{L}(\cdot)$ better than the triangular design?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data structure

    For each iteration, the following will be collected:
    - Execution time
    - Index of iteration
    - The bounds (total 5 for $K=3$)
    - $\alpha'$
    - 1-$\beta'$
    - Sample size per stage
    - Maximum expected sample size
    - Objective function value
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Search space
    """)
    return


@app.cell
def _(c0, mo, np, num_analyses, tri_params):
    search_space_boxes = ['large_box', 'large_box_5_stages', 'small_box', 'triang_box']

    space_dropdown = mo.ui.dropdown(
        options=search_space_boxes,
        value="large_box",
        label="Choose search space:"
    )

    lower_spaces = {}
    upper_spaces = {}

    for key in search_space_boxes:
        if key == "triang_box":
            lower_spaces[key] = np.array([max(0, p - 0.4) for p in tri_params] + [20])
            upper_spaces[key] = np.array([p + 0.4 for p in tri_params] + [160])
            continue

        n = num_analyses.value * 2
        lower = np.zeros(n)
        upper = np.ones(n)

        if key == "large_box":
            upper = upper * 4
            lower[0] = c0 - 3.0
            upper[0] = c0 + 3.0
        elif key == "large_box_5_stages":
            upper = upper * 2
            lower[0] = c0 - 2.0
            upper[0] = c0 + 2.0
            upper[2] = 4
        elif key == "small_box":
            lower[0] = c0 - 1.0
            upper[0] = c0 + 1.0
            upper[2] = 4.0

        if key == "large_box":
            lower[-1] = 20
            upper[-1] = 160
        elif key == "large_box_5_stages":
            lower[-1] = 20
            upper[-1] = 60

        lower_spaces[key] = lower
        upper_spaces[key] = upper
    return lower_spaces, space_dropdown, upper_spaces


@app.cell
def _(lower_spaces, mo, np, space_dropdown, upper_spaces):
    current_lower = lower_spaces[space_dropdown.value]
    current_upper = upper_spaces[space_dropdown.value]

    mo.vstack(
        [
            space_dropdown, 
            mo.md(f"**Lower value:** {np.round(current_lower, decimals=3)}"),
            mo.md(f"**Upper value:** {np.round(current_upper, decimals=3)}")
        ]
    )
    return current_lower, current_upper


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data collection setup
    """)
    return


@app.cell
def _():
    n_experiments = 50
    n_loops = 500
    return n_experiments, n_loops


@app.cell
def _(n_loops):
    space_needed_for_label = len(str(n_loops))
    label_range = 10**(space_needed_for_label)
    return (label_range,)


@app.cell
def _(num_analyses):
    upper_labels = [f"upper{i+1}" for i in range(num_analyses.value)]
    lower_labels = [f"lower{i+1}" for i in range(num_analyses.value - 1)]
    labels = upper_labels + lower_labels

    ordered_keys = ["index"] + labels + [
        "alpha", "power", "sample_size", "max_ess", 
        "obj_func", "execute_time", "seed"
    ]

    bayes_opt_results = {key: [] for key in ordered_keys}
    return bayes_opt_results, labels


@app.cell
def _(np):
    rng = np.random.default_rng(seed = 437591)
    return (rng,)


@app.cell
def _(bayes_opt_results, n_experiments, n_loops, np, rng):
    seed_list = []
    short_seed_list = []

    for _ in range(n_experiments):
        seed = int(np.round(rng.uniform(0, 2**32 - 1)))
        if seed in short_seed_list:
            seed = int(np.round(rng.uniform(0, 2**32 - 1)))

        short_seed_list.append(seed)
        seeds = np.repeat(seed, n_loops)
        seed_list += seeds.tolist()

    bayes_opt_results["seed"] = seed_list
    return (short_seed_list,)


@app.cell
def _(bayes_opt_results, label_range, n_experiments, n_loops):
    index_list = [
        i 
        for start in range(label_range, (n_experiments+1)*label_range, label_range)
        for i in range(start + 1, start + (n_loops+1))
    ]

    bayes_opt_results["index"] = index_list
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Experiment initiation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ============
    ## Bayes opt setup
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    *Note: Ax scales input parameters to `[0,1]` internally and standardizes target outputs natively via BoTorch models, eliminating the need for manual min-max and z-scaling wrappers.*
    """)
    return


@app.cell
def _(mo):
    num_initial_trials = mo.ui.number(label="Number of initial Sobol trials = ", value=20, start=5, stop=100, step=5)
    mo.vstack([num_initial_trials])
    return (num_initial_trials,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## End Bayes opt setup
    ## ============
    """)
    return


@app.cell
def _(
    AxClient,
    ObjectiveProperties,
    bayes_opt_results,
    current_lower,
    current_upper,
    delta0,
    delta1,
    fmt_bd,
    gc,
    labels,
    mu,
    n_experiments,
    n_loops,
    np,
    num_analyses,
    obj_f,
    short_seed_list,
    sigma2,
    target_alpha,
    target_power,
    time,
):
    # Ax parameter configuration
    ax_parameters = []
    for dim in range(len(current_lower)):
        ax_parameters.append({
            "name": f"x_{dim}",
            "type": "range",
            "bounds": [float(current_lower[dim]), float(current_upper[dim])],
            "value_type": "float"
        })

    for i in range(n_experiments):
        np.random.seed(short_seed_list[i])

        ###################
        # Bayes opt model #
        ###################
        ax_client = AxClient(
            random_seed=short_seed_list[i],
            enforce_sequential_optimization=False
        )

        ax_client.create_experiment(
            name=f"gsd_optimization_{i}",
            parameters=ax_parameters,
            objectives={"obj_func": ObjectiveProperties(minimize=True)}
        )

        ############################
        # Start the bayes opt loop #
        ############################
        start_time = time.time()

        for j in range(n_loops):
            # Ask for the next set of parameters
            params, trial_index = ax_client.get_next_trial()

            # Retrieve array structure for our function
            x_new_arr = [params[f"x_{k}"] for k in range(len(current_lower))]
            x_new_sample_size = x_new_arr[(num_analyses.value*2)-1]
            x_new_bounds = x_new_arr[:-1]

            bounds = fmt_bd.reverse_to_boundaries(params = x_new_bounds, K = num_analyses.value)
            bounds_list = np.concatenate( (bounds[0], bounds[1][0:num_analyses.value-1]) )

            # Evaluate objective
            alpha, power, max_ess, y_new = obj_f(
                mu = mu,
                upper_bounds = bounds[0],
                lower_bounds = bounds[1],
                n_analyses = num_analyses.value,
                n_patients = x_new_sample_size,
                target_power = target_power,
                target_alpha = target_alpha,
                null_hypothesis = delta0,
                alternative_hypothesis = delta1,
                variance = sigma2
            )

            # Store the data
            for _i in range(len(bounds_list)):
                bayes_opt_results[labels[_i]].append(bounds_list[_i])

            bayes_opt_results["alpha"].append(alpha)
            bayes_opt_results["power"].append(power)
            bayes_opt_results["sample_size"].append(x_new_sample_size)
            bayes_opt_results["max_ess"].append(max_ess)
            bayes_opt_results["obj_func"].append(y_new)

            # Tell Ax the results
            ax_client.complete_trial(trial_index=trial_index, raw_data={"obj_func": y_new})

            if j % 25 == 0:
                print(".", end="")

        stop_time = time.time()
        execute_time = stop_time - start_time
        bayes_opt_results["execute_time"].extend([execute_time] * n_loops)

        if i % 1 == 0:
            print("\n===========================")
            print(f"= Completed experiment {i+1}. =")
            print("===========================")

        del ax_client
        gc.collect()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Saving data
    """)
    return


@app.cell
def _(bayes_opt_results, pd):
    pd.DataFrame(bayes_opt_results)
    return


@app.cell
def _(n_experiments, n_loops, num_initial_trials):
    file_name = "bo_smooth_ax"
    file_name += "_" + str(n_experiments) + "x" + str(n_loops)
    file_name += "_" + str(num_initial_trials.value) + "_init_trials"
    file_name += ".csv"
    return (file_name,)


@app.cell
def _(file_name):
    file_name
    return


@app.cell
def _(file_name):
    path = "/workspace/experiments_rand_simann_bo/bayes_opt_experiments/" + file_name
    return (path,)


@app.cell
def _(bayes_opt_results, path, pd):
    pd.DataFrame(bayes_opt_results).to_csv(path_or_buf=path)
    return


if __name__ == "__main__":
    app.run()
