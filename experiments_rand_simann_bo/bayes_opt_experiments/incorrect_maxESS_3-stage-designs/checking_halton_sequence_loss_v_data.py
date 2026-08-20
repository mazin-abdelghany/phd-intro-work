import marimo

__generated_with = "0.23.16"
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
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    return np, pd, tf


@app.cell
def _():
    from trieste.space import Box

    return (Box,)


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
    # Trial design settings
    """)
    return


@app.cell
def _(mo):
    num_analyses = mo.ui.number(label="Number of analyses = ", value=3, start=1)

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
    # Search space
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
    # Data generation
    """)
    return


@app.cell
def _():
    n_experiments = 10
    return (n_experiments,)


@app.cell
def _(np):
    rng = np.random.default_rng(seed = 437591)
    return (rng,)


@app.cell
def _(n_experiments, np, rng):
    # create a list of seeds to use
    short_seed_list = [] # for using in the loop

    for _ in range(n_experiments):
        # get entropy for the random number generator seed
        seed = int(np.round(rng.uniform(0, 2**32 - 1)))

        # check that we are not repeating seeds
        if seed in short_seed_list:
            seed = int(np.round(rng.uniform(0, 2**32 - 1)))

        short_seed_list.append(seed)
    return (short_seed_list,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Do we center and scale?
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Get the objective function value for the Halton sequence
    """)
    return


@app.cell
def _(mo):
    num_haltons = mo.ui.number(label="Number of Halton points = ", value=500, start=100, stop=500, step=100)

    mo.vstack([num_haltons])
    return (num_haltons,)


@app.cell
def _(n_experiments):
    halton_dict = {k : [] for k in range(n_experiments)}
    return (halton_dict,)


@app.cell
def _(
    current_lower,
    current_upper,
    delta0,
    delta1,
    fmt_bd,
    halton_dict,
    min_max_unscale,
    mu,
    n_experiments,
    np,
    num_analyses,
    num_haltons,
    obj_f,
    scale_input,
    search_space,
    short_seed_list,
    sigma2,
    target_alpha,
    target_power,
    tf,
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
                x_scaled = initial_x,
                min = current_lower,
                max = current_upper
            )

            initial_points = initial_x_unscaled
        else:
            initial_points = np.array(initial_x, dtype=np.float64)

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
        initial_y_formatted = np.array(initial_y).reshape(-1, 1)

        halton_dict[i] =  tf.concat([initial_points, initial_y_formatted], axis=1)
    return


@app.cell
def _(n_experiments):
    halton_dataframes = {k : [] for k in range(n_experiments)}
    return (halton_dataframes,)


@app.cell
def _(halton_dataframes, halton_dict, n_experiments, num_analyses, pd):
    middle_cols = ["delta" + str(i) for i in range((num_analyses.value * 2) - 2)]

    columns = ["c"] + middle_cols + ["n", "loss"]

    for j in range(n_experiments):
        halton_dataframes[j] = pd.DataFrame(data = halton_dict[0], columns=columns)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Any Halton values better than top designs?
    """)
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
def _(pd):
    # read in the data of interest
    bayes_opt_data = pd.read_csv("/tf/experiments_rand_simann_bo/bayes_opt_experiments/3-stage-designs/large_box_bo_smooth_10x3000.csv")
    return (bayes_opt_data,)


@app.cell
def _(bayes_opt_data):
    last_index = bayes_opt_data["index"].iloc[-1]
    return (last_index,)


@app.cell
def _(last_index):
    experiments, runs = parse_index(last_index)
    return experiments, runs


@app.cell
def _(experiments, n_experiments):
    # ensure the number of experiments is equal
    n_experiments_equal = n_experiments == experiments
    return (n_experiments_equal,)


@app.cell
def _(bayes_opt_data, n_experiments, np, short_seed_list):
    # ensure that all of the seeds are equal
    # the sum should be equal to the number of experiments 
    all_equal = np.sum(
        np.equal(np.unique(bayes_opt_data["seed"]), np.sort(short_seed_list))
    ) == n_experiments
    return (all_equal,)


@app.cell
def _(all_equal, n_experiments_equal):
    if not all_equal:
        raise print("Not all seeds are equal! The rest of the analysis would be invalid.")
    if not n_experiments_equal:
        raise print("The replicate number does not match! The rest of the analysis would be invalid.")
    return


@app.cell
def _(bayes_opt_data, halton_dataframes, n_experiments, runs):
    oops = {
        "how_many" : 0,
        "index" : []
    }

    for ell in range(n_experiments):
        start_index = runs * ell
        end_index = runs * (ell+1)

        bayes_slice_loss = bayes_opt_data[start_index:end_index]["obj_func"]

        if halton_dataframes[ell]["loss"].min() < bayes_slice_loss.min():
            oops["how_many"] += 1
            oops["index"].append(ell+1)
    return (oops,)


@app.cell
def _(oops):
    oops["how_many"]
    return


if __name__ == "__main__":
    app.run()
