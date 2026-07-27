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
def _(c0, mo, np, num_analyses, tri_params):
    search_space_boxes = ['large_box', 'large_box_5_stages', 'small_box', 'triang_box']

    # create a single dropdown
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
            # accounts for the last stage before 
            # meeting point needs to be a larger 
            # value to get to $c$ in some cases
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
    # generate labels for data frame
    # if we have experiments <1000, then we need 3 spaces 
    # to fill. this is the length of the string and then
    # we raise 10 to this number to obtain the labels
    space_needed_for_label = len(str(n_loops))
    label_range = 10**(space_needed_for_label)
    return (label_range,)


@app.cell
def _(num_analyses):
    # create the data container that will collect all the values of interest
    # we will use an empty dictionary for memory efficiency and the convert

    # dynamic labels for the bounds
    upper_labels = [f"upper{i+1}" for i in range(num_analyses.value)]
    lower_labels = [f"lower{i+1}" for i in range(num_analyses.value - 1)]

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
        n_analyses=num_analyses.value, alpha=0.05
    )

    poc_n = ss.find_sample_size(
        power_target = target_power,
        n_analyses = num_analyses.value,
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
        n_analyses=num_analyses.value, alpha=0.05
    )

    obf_n = ss.find_sample_size(
        power_target = target_power,
        n_analyses = num_analyses.value,
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
        n_analyses = num_analyses.value,
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
        n_analyses = num_analyses.value,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ============
    ## Bayes opt setup
    """)
    return


@app.cell
def _(mo):
    scale_input = mo.ui.switch(label="Min-max scale inputs")
    return (scale_input,)


@app.cell
def _(mo, scale_input):
    mo.vstack([scale_input, mo.md(f"Has value: {scale_input.value}")])
    return


@app.cell
def _(mo):
    scale_output = mo.ui.switch(label="Z-scale outputs")
    return (scale_output,)


@app.cell
def _(mo, scale_output):
    mo.vstack([scale_output, mo.md(f"Has value: {scale_output.value}")])
    return


@app.cell
def _(mo):
    num_haltons = mo.ui.number(label="Number of Halton points = ", value=500, start=100, stop=500, step=100)

    mo.vstack([num_haltons])
    return (num_haltons,)


@app.cell
def _(mo):
    do_not_train_error = mo.ui.switch(label="Do not train error")
    return (do_not_train_error,)


@app.cell
def _(do_not_train_error, mo):
    mo.vstack([do_not_train_error, mo.md(f"Has value: {do_not_train_error.value}")])
    return


@app.cell
def _(do_not_train_error, mo):
    if do_not_train_error.value:
        radio = mo.ui.radio(
            options={
                "1e-1": 1e-1,
                "1e-2": 1e-2,
                "1e-3": 1e-3,
                "1e-4": 1e-4,
                "1e-5": 1e-5,
            },
            value="1e-3",
            label="Likelihood variance",
        )
        radio
    else:
        radio = mo.ui.radio(
            options={"1": 1},
            label="Initial value for likelihood variance",
        )
    radio
    return (radio,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## End Bayes opt setup
    ## ============
    """)
    return


@app.cell
def _(Box, current_lower, current_upper, scale_input):
    if scale_input.value:
        # ensure all dimensions are [0,1] with min-max scaling
        # defaults to min and max are set as current upper and lower
        def min_max_scale(x, min, max):
            """Transform values to [0, 1]."""
            return (x - min) / (max - min)

        # defaults to min and max are set as current upper and lower
        def min_max_unscale(x_scaled, min, max):
            """Transform normalized [0, 1] values back."""
            return x_scaled * (max - min) + min

        # because we are min-max scaling, search space will be [0,1]^D
        search_space = Box(
            lower = [0.0] * len(current_lower), 
            upper = [1.0] * len(current_upper)
        )

        print(search_space.lower)
        print(search_space.upper)
    else:
        search_space = Box(
            lower = current_lower,
            upper = current_upper
        )

        print(search_space.lower)
        print(search_space.upper)
    return min_max_unscale, search_space


@app.cell
def _(np, scale_output):
    if scale_output.value:
        def z_score_scale(x, mu, sigma, axis = 0):
            """Standardize input array using Z-score scale: (x - mu) / sigma."""
            return (x - mu) / sigma

        def z_score_unscale(x_scaled, mu, sigma, axis = 0):
            """Restore scaled data back to original space: (x_scaled * sigma) + mu."""
            return (x_scaled * np.expand_dims(sigma, axis=axis)) + np.expand_dims(mu, axis=axis)
    return (z_score_scale,)


@app.cell
def _(
    GaussianProcessRegression,
    bayes_opt_results,
    current_lower,
    current_upper,
    delta0,
    delta1,
    do_not_train_error,
    fmt_bd,
    gc,
    gpflow,
    labels,
    min_max_unscale,
    mu,
    n_experiments,
    n_loops,
    np,
    num_analyses,
    num_haltons,
    obj_f,
    radio,
    scale_input,
    scale_output,
    search_space,
    short_seed_list,
    sigma2,
    target_alpha,
    target_power,
    tf,
    time,
    trieste,
    z_score_scale,
):
    for i in range(n_experiments):

        np.random.seed(short_seed_list[i])
        tf.random.set_seed(short_seed_list[i])

        ########################
        # Halton initialisation #
        ########################
        initial_x = search_space.sample_halton(num_haltons.value, seed = short_seed_list[i])

        # search space is [0,1]^D, thus, need to unscale prior to getting y
        # values for the initial GP fit
        if scale_input.value:
            initial_x_unscaled = min_max_unscale(
                x_scaled = initial_x.numpy(),
                min = current_lower,
                max = current_upper
            )

            initial_points = initial_x_unscaled
        else:
            initial_points = initial_x

        initial_y = []

        # for each point
        for point in initial_points:

            sample_size = point[(num_analyses.value*2)-1]
            bounds = point[:-1]

            bounds = fmt_bd.reverse_to_boundaries(params = bounds, K = num_analyses.value)

            _, _, _, initial_y_new = obj_f(
                mu = mu,
                upper_bounds = bounds[0],
                lower_bounds = bounds[1],
                n_analyses = num_analyses.value,
                n_patients = sample_size,
                target_power = target_power,
                target_alpha = target_alpha,
                null_hypothesis = delta0,
                alternative_hypothesis = delta1,
                variance = sigma2
            )

            initial_y.append(initial_y_new)

        # turn y into [N,1] column vector
        initial_y_formatted = np.array(initial_y, dtype=np.float64).reshape(-1, 1)

        # scale y as well
        if scale_output.value:
            y_mu = np.mean(initial_y_formatted)
            y_sigma = np.std(initial_y_formatted)
            initial_y_formatted = z_score_scale(
                initial_y_formatted,
                mu = y_mu,
                sigma = y_sigma
            )

        initial_data = trieste.data.Dataset(
            query_points = initial_x,
            observations = initial_y_formatted
        )

        #######################
        # GP regression model #
        #######################
        kernel = gpflow.kernels.Matern52(
            lengthscales = [1.0] * (num_analyses.value * 2)
        )

        # these priors regularize the lengthscale to have ~99% of its probability
        # distribution between +/- exp(3*scale)
        # it sets the same prior on all lengthscales; if a different prior is 
        # needed for different dimensions, can do:
        # loc = gpflow.utilities.to_default_float([0.0, 1.0, -0.5])
        # scale = gpflow.utilities.to_default_float([1.0, 2.0, 0.5])
        # kernel.lengthscales.prior = tfp.distributions.LogNormal(
        #     loc=loc,
        #     scale=scale,
        # )
        # kernel.lengthscales.prior = tfp.distributions.LogNormal(
        #     gpflow.utilities.to_default_float(0), gpflow.utilities.to_default_float(2)
        # )
        # kernel.variance.prior = tfp.distributions.LogNormal(
        #     gpflow.utilities.to_default_float(0), gpflow.utilities.to_default_float(3)
        # )

        # consider decreasing for K=3?
        likelihood = gpflow.likelihoods.Gaussian(variance = radio.value)

        if do_not_train_error.value:
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

        # from the source code for trieste.acquisition.optimizer:
        # NUM_SAMPLES_MIN = 5000
        # NUM_SAMPLES_DIM = 1000
        # NUM_RUNS_DIM = 10
        # using the Trieste recommendation from their documentation,
        # num_initial_samples = max(NUM_SAMPLES_MIN, NUM_SAMPLES_DIM * D)
        # where D is the dimensions (K=3 -> D=6; K=5 -> D=10)
        # num_optimization_runs = NUM_RUNS_DIM * D (10 * D)
        ask_tell = trieste.ask_tell_optimization.AskTellOptimizer(
            search_space     = search_space,
            datasets         = initial_data,
            models           = bayes_opt_model,
            acquisition_rule = trieste.acquisition.rule.EfficientGlobalOptimization(
                #optimizer = trieste.acquisition.optimizer.generate_continuous_optimizer(
                #    num_initial_samples=10000,  # see above for reason
                #    num_optimization_runs=100,  # see above for reason
                #    num_recovery_runs=10        # left at default
                #)
            )
        )

        ############################
        # Start the bayes opt loop #
        ############################
        start_time = time.time()

        # there are n_loops number of reverse_bounds to iterate through
        for j in range(n_loops):
            x_new = ask_tell.ask()

            # print("new point:", x_new.numpy())
            # print(ask_tell.to_result().try_get_final_dataset().query_points)
            # print("cond#:", np.linalg.cond(kernel(ask_tell.to_result().try_get_final_dataset().query_points)))
            # X = ask_tell.to_result().try_get_final_dataset().query_points.numpy()
            # d = np.linalg.norm(X - x_new.numpy(), axis=1)
            # print("dist:", d.min())

            if scale_input.value:
                x_new_unscaled = min_max_unscale(
                    x_scaled = x_new.numpy(),
                    min = current_lower,
                    max = current_upper
                )

            if scale_input.value:
                x_new_sample_size = x_new_unscaled[0][(num_analyses.value*2)-1]
                x_new_bounds = x_new_unscaled[0][:-1]
            else:
                x_new_sample_size = x_new[0][(num_analyses.value*2)-1]
                x_new_bounds = x_new[0][:-1]

            bounds = fmt_bd.reverse_to_boundaries(params = x_new_bounds, K = num_analyses.value)
            bounds_list = np.concatenate( (bounds[0], bounds[1][0:num_analyses.value-1]) )

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

            # collect the boundaries using the labels
            for _i in range(len(bounds_list)):
                bayes_opt_results[labels[_i]].append(bounds_list[_i])

            # collect the rest of the value of interest
            bayes_opt_results["alpha"].append(alpha)
            bayes_opt_results["power"].append(power)
            bayes_opt_results["sample_size"].append(x_new_sample_size)
            bayes_opt_results["max_ess"].append(max_ess)
            bayes_opt_results["obj_func"].append(y_new)

            if scale_output.value:
                y_new_scaled = z_score_scale(
                    y_new,
                    mu = y_mu,
                    sigma = y_sigma
                )

                ask_tell.tell(trieste.data.Dataset(
                    query_points = x_new,
                    observations = np.array([[y_new_scaled]])
                ))
            else:
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

        bayes_opt_results["execute_time"].extend([execute_time] * n_loops)

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
def _(
    do_not_train_error,
    error,
    n_experiments,
    n_loops,
    num_haltons,
    scale_input,
    scale_output,
):
    file_name = "bo_smooth"

    file_name += "_" + str(n_experiments) + "x" + str(n_loops)

    if scale_input.value:
        file_name += "_x_min_max"
    if scale_output.value:
        file_name += "_y_z_scaled"
    if do_not_train_error.value:
        file_name += "_" + str(error)

    file_name += "_" + str(num_haltons.value) + "haltons"

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
