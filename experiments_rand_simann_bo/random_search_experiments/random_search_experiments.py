import marimo

__generated_with = "0.23.14"
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
    import scipy.stats as stats
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    return np, pd, time


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
    num_analyses = 5
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
    fmt_bd,
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
        mu = 154,
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
    lower_sample_size = 20.
    upper_sample_size = 160.
    return lower_sample_size, upper_sample_size


@app.cell
def _(
    c0,
    lower_sample_size,
    mo,
    np,
    num_analyses,
    tri_params,
    upper_sample_size,
):
    # create a single dropdown
    space_dropdown = mo.ui.dropdown(
        options=['large_box', 'small_box', 'triang_box'],
        value="triang_box",
        label="Choose search space:"
    )

    search_space_boxes = ['large_box', 'small_box', 'triang_box']

    lower_spaces = {}
    upper_spaces = {}

    for key in search_space_boxes:
        if key == "triang_box":
            lower_spaces[key] = np.array([max(0, p - 0.4) for p in tri_params] + [lower_sample_size])
            upper_spaces[key] = np.array([p + 0.4 for p in tri_params] + [upper_sample_size])
            continue

        n = num_analyses * 2
        lower = np.zeros(n)
        upper = np.ones(n)

        if key == "large_box":
            upper = upper * 4
            lower[0] = c0 - 3.0
            upper[0] = c0 + 3.0
        elif key == "small_box":
            lower[0] = c0 - 1.0
            upper[0] = c0 + 1.0
            upper[2] = 4.0

        lower[-1] = lower_sample_size
        upper[-1] = upper_sample_size

        lower_spaces[key] = lower
        upper_spaces[key] = upper

    # lookups for lower and upper spaces based on the selected key
    #lower_spaces = {
    #    'large_box' : np.array([c0 - 3.0, 0.0, 0.0, 0.0, 0.0, lower_sample_size]),
    #    'small_box' : np.array([c0 - 1, 0.0, 0.0, 0.0, 0.0, lower_sample_size]),
    #    'triang_box' : np.array([max(0, param - 0.4) for param in tri_params] + [lower_sample_size])
    #}

    #upper_spaces = {
    #    'large_box' : np.array([c0 + 3.0, 4.0, 4.0, 4.0, 4.0, upper_sample_size]),
    #    'small_box' : np.array([c0 + 1, 1.0, 4.0, 1.0, 1.0, upper_sample_size]),
    #    'triang_box' : np.array([param + 0.4 for param in tri_params] + [upper_sample_size])
    #}
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

    random_search_large_box = {key: [] for key in ordered_keys}
    return labels, random_search_large_box


@app.cell
def _(label_range, n_experiments, n_loops, random_search_large_box):
    # generate the indices using the pattern described above
    index_list = [
        i 
        for start in range(label_range, (n_experiments+1)*label_range, label_range) 
        for i in range(start + 1, start + (n_loops+1))
    ]

    random_search_large_box["index"] = index_list
    return


@app.cell
def _(n_experiments, n_loops, np, random_search_large_box):
    # create a list of seeds to use
    seed_list = [] # for filling the dictionary
    short_seed_list = [] # for using in the loop

    for _ in range(n_experiments):
        # get entropy for the random number generator seed
        seed = np.random.SeedSequence().entropy
        short_seed_list.append(seed)
        seeds = np.repeat(seed, n_loops)
        seed_list += seeds.tolist()

    random_search_large_box["seed"] = seed_list
    return (short_seed_list,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Experiment initiation
    """)
    return


@app.cell
def _(
    current_lower,
    current_upper,
    delta0,
    delta1,
    fmt_bd,
    labels,
    mu,
    n_experiments,
    n_loops,
    np,
    num_analyses,
    obj_f,
    random_search_large_box,
    short_seed_list,
    sigma2,
    target_alpha,
    target_power,
    time,
):
    for i in range(n_experiments):

        # initialize the rng
        rng = np.random.default_rng(seed = short_seed_list[i])

        # generate n_loops # of reverse bounds, array shape is (n_loops x 5)
        parameters = rng.uniform(current_lower, current_upper, size = (n_loops, len(current_lower)))

        reverse_bounds = parameters[:, 0:len(parameters[0])-1]
        sample_size = parameters[:, len(parameters[0])-1]

        start_time = time.time()

        # there are n_loops number of reverse_bounds to iterate through
        for j, boundaries in enumerate(reverse_bounds):

            bounds = fmt_bd.reverse_to_boundaries(params = boundaries, K = num_analyses)
            bounds_list = np.concatenate( (bounds[0], bounds[1][0:num_analyses-1]) )

            alpha, power, max_ess, obj = obj_f(
                mu = mu,
                upper_bounds = bounds[0],
                lower_bounds = bounds[1],
                n_analyses = num_analyses,
                n_patients = sample_size[j],
                target_power = target_power,
                target_alpha = target_alpha,
                null_hypothesis = delta0,
                alternative_hypothesis = delta1,
                variance = sigma2
            )

            # collect the boundaries using the labels
            for _i in range(len(bounds_list)):
                random_search_large_box[labels[_i]].extend([bounds_list[_i]])

            # collect the rest of the value of interest
            random_search_large_box["alpha"].extend([alpha])
            random_search_large_box["power"].extend([power])
            random_search_large_box["sample_size"].extend([sample_size[j]])
            random_search_large_box["max_ess"].extend([max_ess])
            random_search_large_box["obj_func"].extend([obj])

            if j % 25 == 0:
                print(".", end = "")

        stop_time = time.time()
        execute_time = stop_time - start_time

        time_list = np.repeat(execute_time, n_loops)
        time_list += time_list.tolist()
        random_search_large_box["execute_time"].extend(time_list)

        if i % 10 == 0:
            print("\n===========================")
            print(f"= Completed experiment {i+1}. =")
            print("===========================")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Saving data
    """)
    return


@app.cell
def _(pd, random_search_large_box):
    pd.DataFrame(random_search_large_box)
    return


@app.cell
def _(pd, random_search_large_box):
    pd.DataFrame(random_search_large_box).to_csv(
        "/tf/experiments_rand_simann_bo/random_search_experiments/large_box_50x500.csv"
    )
    return


if __name__ == "__main__":
    app.run()
