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
    import tensorflow as tf
    import tensorflow_probability as tfp

    return gc, np, pd, tf, time


@app.cell
def _():
    import gpflow
    import trieste
    from trieste.space import Box
    from trieste.models.gpflow.models import GaussianProcessRegression

    return Box, GaussianProcessRegression, gpflow, trieste


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

    print(f"We are running an experiment with a trial design with {num_analyses} stages, with:\na target alpha of {target_alpha},\na target power of {target_power},\na null hypothesis of {delta0},\nan alternative hypothesis of {delta1},\nand an assumed variance of {sigma2}\n")
    print(f"Single-stage sample size mu = {mu:.2f}")
    return delta0, delta1, mu, num_analyses, sigma2, target_alpha, target_power


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

    tri_params = fmt_bd.boundaries_to_reverse(
        upper_bounds = tri[0],
        lower_bounds = tri[1]
    )

    c0 = tri_params[0]

    print(f"Original trriangular params: {np.round(np.concatenate((tri[0], tri[1])), 4)}")
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

    Each experiment will be run for 500 loops. For the first experiment, the index will start at 1001 and end at 1500. The next experiment index will start at 2001 and end at 2500. Thus, the thousands place will correspond to the expirment number and the hundreds place and lower will correspond to the loop index.

    Below is a table that indicates what the data structure will look like.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | index | upper1 | upper2 | upper3 | lower1 | lower2 | alpha | power | sample_size | max_ess | obj_func | execute_time |
    |-------|--------|--------|--------|--------|--------|-------|-------|-------------|---------|----------|--------------|
    | 1001  |        |        |        |        |        |       |       |             |         |          |              |
    | ...   |        |        |        |        |        |       |       |             |         |          |              |
    | 1500  |        |        |        |        |        |       |       |             |         |          |              |
    | 2001  |        |        |        |        |        |       |       |             |         |          |              |
    | ...   |        |        |        |        |        |       |       |             |         |          |              |
    | 2500  |        |        |        |        |        |       |       |             |         |          |              |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Experimental setup

    Because both $\alpha$ and $\beta$ are affected by the boundary selection and the per group sample size, $n$, the boundaries and the sample size will be included in the random search.

    ## Pseudocode:
    - Create a seed list of 50 unique seeds and save it.
    - For the number of seeds in the list:
        - Set the seed
        - Generate the vector of 500 x 6: [bounds, n]
        - For each item in the vector:
              - Reverse the boundaries
              - Calculate the values of interest (above)
              - Add the values as a row of the data

    Once the experiments are completed, summary statistics of each run and the for all of the runs can be calculated and presented.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Search space
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The sample size search space maximum was selected to be approximate the same size as the single stage sample size $\mu\approx 155$. The triangular design (considered near-optimal) has a staged sample size for $1-\beta=0.9$ of 65. The lower bound was selected as a floor for the number of patients that would not be too low to get near this power.
    """)
    return


@app.cell
def _():
    # commentary on selection above
    lower_sample_size = 20
    upper_sample_size = 160
    return lower_sample_size, upper_sample_size


@app.cell
def _(c0, lower_sample_size, mo, np, tri_params, upper_sample_size):
    # create a single dropdown
    space_dropdown = mo.ui.dropdown(
        options=['large_box', 'small_box', 'triang_box'],
        value="triang_box",
        label="Choose search space:"
    )

    # lookups for lower and upper spaces based on the selected key
    lower_spaces = {
        'large_box' : np.array([c0 - 3.0, 0.0, 0.0, 0.0, 0.0, lower_sample_size]),
        'small_box' : np.array([c0 - 1, 0.0, 0.0, 0.0, 0.0, lower_sample_size]),
        'triang_box' : np.array([max(0, param - 0.4) for param in tri_params] + [lower_sample_size])
    }

    upper_spaces = {
        'large_box' : np.array([c0 + 3.0, 4.0, 4.0, 4.0, 4.0, upper_sample_size]),
        'small_box' : np.array([c0 + 1, 1.0, 4.0, 1.0, 1.0, upper_sample_size]),
        'triang_box' : np.array([param + 0.4 for param in tri_params] + [upper_sample_size])
    }
    return lower_spaces, space_dropdown, upper_spaces


@app.cell
def _(lower_spaces, mo, np, space_dropdown, upper_spaces):
    # get active arrays based on the dropdown current value
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


@app.cell
def _(Box, current_lower, current_upper, np):
    search_space = Box(
        lower = current_lower, 
        upper = current_upper
    )
    print(f"  lower: {np.round(search_space.lower.numpy(), 3)}")
    print(f"  upper: {np.round(search_space.upper.numpy(), 3)}")
    return (search_space,)


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

    # generate labels for data frame
    # if we have experiments <1000, then we need 3 spaces 
    # to fill. this is the length of the string and then
    # we raise 10 to this number to obtain the labels
    space_needed_for_label = len(str(n_loops))
    label_range = 10**(space_needed_for_label)
    return label_range, n_experiments, n_loops


@app.cell
def _(num_analyses):
    # create the data container that will collect all the values of interest
    # we will use an empty dictionary for memory efficiency and the convert

    # dynamic labels for the bounds
    upper_labels = [f"upper{i+1}" for i in range(num_analyses)]
    lower_labels = [f"lower{i+1}" for i in range(num_analyses - 1)]

    # labels will be used again in the experiment loop
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
    # create a list of seeds to use
    seed_list = [] # for filling the dictionary
    short_seed_list = [] # for using in the loop

    for _ in range(n_experiments):
        # get entropy for the random number generator seed
        seed = int(np.round(rng.uniform(0, 2**32 - 1)))

        # check that we are not repeating seeds
        if seed in short_seed_list:
            seed = int(np.round(rng.uniform(0, 2**32 - 1)))

        short_seed_list.append(seed)

        # create a long seed set to collect into the data.frame
        seeds = np.repeat(seed, n_loops)
        seed_list += seeds.tolist()

    bayes_opt_results["seed"] = seed_list
    return (short_seed_list,)


@app.cell
def _(bayes_opt_results, label_range, n_experiments, n_loops):
    # generate the indices using the pattern described above
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
    ## Old initialisation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    poc = bd.calculate_pocock_boundaries(
        n_analyses=num_analyses, alpha=0.05
    )

    poc_n = ss.find_sample_size(
        power_target = target_power,
        n_analyses = num_analyses,
        upper_bounds = poc[0],
        lower_bounds = poc[1],
        null_hypothesis = delta0,
        alt_hypothesis = delta1,
        variance = sigma2
    )[0]

    poc_rev_bounds = fmt_bd.boundaries_to_reverse(poc[0], poc[1])

    poc_params = np.concatenate((poc_rev_bounds, [poc_n]))
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    obf = bd.calculate_of_boundaries(
        n_analyses=num_analyses, alpha=0.05
    )

    obf_n = ss.find_sample_size(
        power_target = target_power,
        n_analyses = num_analyses,
        upper_bounds = obf[0],
        lower_bounds = obf[1],
        null_hypothesis = delta0,
        alt_hypothesis = delta1,
        variance = sigma2
    )[0]

    obf_rev_bounds = fmt_bd.boundaries_to_reverse(obf[0], obf[1])

    obf_params = np.concatenate((obf_rev_bounds, [obf_n]))
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    _,_,_,poc_obj_f = obj_f(
        mu = mu,
        upper_bounds = poc[0],
        lower_bounds = poc[1],
        n_analyses = num_analyses,
        n_patients = poc_n,
        target_power = target_power,
        target_alpha = target_alpha,
        null_hypothesis = delta0,
        alternative_hypothesis = delta1,
        variance = sigma2
    )

    _,_,_,obf_obj_f = obj_f(
        mu = mu,
        upper_bounds = obf[0],
        lower_bounds = obf[1],
        n_analyses = num_analyses,
        n_patients = obf_n,
        target_power = target_power,
        target_alpha = target_alpha,
        null_hypothesis = delta0,
        alternative_hypothesis = delta1,
        variance = sigma2
    )
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    design_matrix = np.concatenate(
        (np.atleast_2d(standardize(poc_params)), np.atleast_2d(standardize(obf_params)))
    )

    output_vals = np.concatenate((np.atleast_2d(poc_obj_f), np.atleast_2d(obf_obj_f)))
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    print(f"Initial dataset:\n{design_matrix}\n")
    print(f"Initial f(x):\n{output_vals/200}")
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayesian optimisation loop
    """)
    return


@app.cell
def _(
    GaussianProcessRegression,
    bayes_opt_results,
    delta0,
    delta1,
    fmt_bd,
    gc,
    gpflow,
    labels,
    mu,
    n_experiments,
    n_loops,
    np,
    num_analyses,
    obj_f,
    search_space,
    short_seed_list,
    sigma2,
    target_alpha,
    target_power,
    tf,
    time,
    trieste,
):
    for i in range(n_experiments):

        np.random.seed(short_seed_list[i])
        tf.random.set_seed(short_seed_list[i])

        ########################
        # Halton initialisation #
        ########################
        initial_x = search_space.sample_halton(500, seed = short_seed_list[i])
        initial_y = []

        for point in initial_x:

            sample_size = point[5,]
            bounds = point[:-1]

            bounds = fmt_bd.reverse_to_boundaries(params = bounds, K = num_analyses)

            _, _, _, initial_y_new = obj_f(
                mu = mu,
                upper_bounds = bounds[0],
                lower_bounds = bounds[1],
                n_analyses = num_analyses,
                n_patients = sample_size.numpy(),
                target_power = target_power,
                target_alpha = target_alpha,
                null_hypothesis = delta0,
                alternative_hypothesis = delta1,
                variance = sigma2
            )

            initial_y.append(initial_y_new)

        initial_y_formatted = np.atleast_2d(initial_y).transpose()

        initial_data = trieste.data.Dataset(
            query_points = initial_x,
            observations = initial_y_formatted
        )

        #######################
        # GP regression model #
        #######################
        kernel = gpflow.kernels.Matern52(
            lengthscales = [1.0] * (num_analyses * 2)
        )

        #kernel.lengthscales.prior = tfp.distributions.LogNormal(
        #    gpflow.utilities.to_default_float(0), gpflow.utilities.to_default_float(3)
        #        )
        #kernel.variance.prior = tfp.distributions.LogNormal(
        #    gpflow.utilities.to_default_float(0), gpflow.utilities.to_default_float(3)
        #)

        likelihood = gpflow.likelihoods.Gaussian(variance = 1e-1)

        gpflow.set_trainable(likelihood, False)

        gpr = gpflow.models.GPR(
            data      = (initial_x, initial_y_formatted),
            kernel    = kernel,
            likelihood = likelihood
        )

        ###################
        # Bayes opt model #
        ###################
        bayes_opt_model = GaussianProcessRegression(gpr)

        ask_tell = trieste.ask_tell_optimization.AskTellOptimizer(
            search_space     = search_space,
            datasets         = initial_data,
            models           = bayes_opt_model,
            acquisition_rule = trieste.acquisition.rule.EfficientGlobalOptimization(
                optimizer = trieste.acquisition.optimizer.generate_continuous_optimizer()
            )
        )

        ############################
        # Start the bayes opt loop #
        ############################
        start_time = time.time()

        # there are n_loops number of reverse_bounds to iterate through
        for j in range(n_loops):
            x_new = ask_tell.ask()

            x_new_sample_size = x_new[0][5,]
            x_new_bounds = x_new[0][:-1]

            bounds = fmt_bd.reverse_to_boundaries(params = x_new_bounds, K = num_analyses)
            bounds_list = np.concatenate( (bounds[0], bounds[1][0:2]) )

            alpha, power, max_ess, y_new = obj_f(
                mu = mu,
                upper_bounds = bounds[0],
                lower_bounds = bounds[1],
                n_analyses = num_analyses,
                n_patients = x_new_sample_size.numpy(),
                target_power = target_power,
                target_alpha = target_alpha,
                null_hypothesis = delta0,
                alternative_hypothesis = delta1,
                variance = sigma2
            )

            # collect the boundaries using the labels
            for _i in range(len(bounds_list)):
                bayes_opt_results[labels[_i]].extend([bounds_list[_i]])

            # collect the rest of the value of interest
            bayes_opt_results["alpha"].extend([alpha])
            bayes_opt_results["power"].extend([power])
            bayes_opt_results["sample_size"].extend([x_new_sample_size.numpy()])
            bayes_opt_results["max_ess"].extend([max_ess])
            bayes_opt_results["obj_func"].extend([y_new])

            ask_tell.tell(trieste.data.Dataset(
                query_points = x_new,
                observations = np.array([[y_new]])
            ))

            if j % 25 == 0:
                print(".", end="")

            if j % 100 == 0:
                print(gpr.kernel.lengthscales.numpy())
                print(gpr.kernel.variance.numpy())
                print(gpr.likelihood.variance.numpy())

        stop_time = time.time()
        execute_time = stop_time - start_time

        time_list = np.repeat(execute_time, n_loops)
        time_list += time_list.tolist()
        bayes_opt_results["execute_time"].extend(time_list)

        if i % 1 == 0:
            print("\n===========================")
            print(f"= Completed experiment {i+1}. =")
            print("===========================")

        del ask_tell
        del bayes_opt_model
        del gpr
        del kernel

        # see https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session
        tf.keras.backend.clear_session()
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
def _(bayes_opt_results, pd):
    pd.DataFrame(bayes_opt_results).to_csv(
        "/workspace/experiments_rand_simann_bo/bayes_opt_experiments/large_box_bo_smooth.csv"
    )
    return


if __name__ == "__main__":
    app.run()
