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
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    return np, pd, plt, time


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
    return c0, tri, tri_obj, tri_params


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

        n = num_analyses.value * 2
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
    # Neighbouring boundaries

    In simulated annealing, an initial state $s=s_0$ is selected, and for a certain number of interations, a random neighbour is chosen $s_{\texttt{new}}$ that is considered the neighbour of $s$, e.g.,
    \[
    s_{\texttt{new}} \leftarrow \texttt{neighbour}(s)
    \]
    In order to run the simulated annealing algorithm, we need to create this `neighbour()` function.

    Inspiration is taken from the classic application of simulated annealing&mdash;the traveling salesperson problem. The search space for $n=20$ cities to visit by the salesperson is $n!\approx 2.4 \text{ quintillion}$ states. Sufficiently near in this application of simulated annealing is the set of permutations produced by swapping any two neighbouring cities, which is only
    \[
    \sum_{i=1}^{n-1} k = \frac{n(n-1)}{2}=190
    \]

    With this in mind, recall that we are in a 6-dimensional space. We need to consider only sufficiently near neighbours. Thus, a neighbouring state will be defined as modifying a single value within this 6-dimensional space. In order to generate this neighbouring state, we will select&mdash;at random&mdash;a point in the 6-d space, modify it, and then calculate its characterestics.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Neighbour functions

    There are two options for the neighbour functions:
    1. We generate the bounds from the box as is (`rand_neighbour()`).
    2. We modify the current bounds with some small perturbation (`norm_neighbour()`).
    """)
    return


@app.cell
def _(np):
    rngt = np.random.default_rng(seed = 5320945)
    return (rngt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random uniform neighbour
    """)
    return


@app.cell
def _(current_lower, current_upper, num_analyses):
    # the neighbour function
    def rand_neighbour(params, K, rng):

        modifying_params = params.copy()

        # generates discrete uniform values from 0 to num_analyses
        idx_to_change = rng.integers(low=0, high=(num_analyses.value*2))

        modifying_params[idx_to_change] = rng.uniform(
            low = current_lower[idx_to_change], 
            high = current_upper[idx_to_change]
        )

        return modifying_params

    return (rand_neighbour,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Normal distribution neighbour
    """)
    return


@app.cell
def _(np):
    # because the perturbations could land the parameters outside of the bounds
    # a function is created to assess if the new bounds are within the boundaries
    def within_search_space(params, lower_search_bounds, upper_search_bounds):

        within_search_space = (params >= lower_search_bounds) & (params <= upper_search_bounds)

        return np.all(within_search_space)

    return (within_search_space,)


@app.cell
def _(num_analyses, within_search_space):
    def norm_neighbour(params, K, rng, sigma_vector, lower_search_bounds, upper_search_bounds):

        modifying_params = params.copy()

        # generates discrete uniform values from 0 to num_analyses
        idx_to_change = rng.integers(low=0, high=(num_analyses.value*2))

        # make a first change
        # generate normal(0, 1) perturbation and multiply by correct sigma
        perturbation = rng.normal() * sigma_vector[idx_to_change]
        modifying_params[idx_to_change] = params[idx_to_change] + perturbation

        while not within_search_space(modifying_params, lower_search_bounds, upper_search_bounds):
            # try another perturbation
            perturbation = rng.normal() * sigma_vector[idx_to_change]
            modifying_params[idx_to_change] = params[idx_to_change] + perturbation

        return modifying_params

    return (norm_neighbour,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Testing the neighbour functions
    """)
    return


@app.cell
def _(np, tri_params):
    parameter_test = np.concatenate((tri_params, [20]))
    return (parameter_test,)


@app.cell
def _(parameter_test):
    parameter_test
    return


@app.cell
def _(num_analyses, parameter_test, rand_neighbour, rngt):
    rand_neighbour(parameter_test, num_analyses.value, rng = rngt)
    return


@app.cell
def _(np, num_analyses):
    # empirically selected sigma values that will decrease over time
    sigma_vector = np.ones(num_analyses.value*2) * 2
    sigma_vector[(num_analyses.value*2)-1] = 25
    return (sigma_vector,)


@app.cell
def _(sigma_vector):
    sigma_vector
    return


@app.cell
def _(
    current_lower,
    current_upper,
    norm_neighbour,
    num_analyses,
    parameter_test,
    rngt,
    sigma_vector,
):
    norm_neighbour(parameter_test, 
                   num_analyses.value, 
                   rng = rngt, 
                   sigma_vector = sigma_vector, 
                   lower_search_bounds = current_lower,
                   upper_search_bounds = current_upper)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Random uniform neighbour
    """)
    return


@app.cell
def _(np, num_analyses, rand_neighbour, rngt, tri_params):
    bounds_collector = []
    sample_size_collector = []
    num_tests = 2000
    tmp = np.concatenate((tri_params, [20]))
    for _i in range(num_tests):
        tmp = rand_neighbour(tmp, num_analyses.value, rng = rngt)
        bound = tmp[0:(num_analyses.value*2)-1]
        sample_size_collector.append(tmp[(num_analyses.value*2)-1])
        bounds_collector.append(bound.tolist())
    return bounds_collector, num_tests


@app.cell
def _(bounds_collector, fmt_bd, num_analyses, num_tests, plt, tri):
    _fig, _ax = plt.subplots()

    analyses = [i+1 for i in range(num_analyses.value)]

    for _i in range(num_tests):
        _bounds = fmt_bd.reverse_to_boundaries(bounds_collector[_i], K=num_analyses.value)
        _ax.plot(analyses, _bounds[0], color = "purple", alpha = 0.1)
        _ax.plot(analyses, _bounds[1], color = "purple", alpha = 0.1)

    _ax.plot(analyses, tri[0], color = "red", lw = 2)
    _ax.plot(analyses, tri[1], color = "red", lw = 2)

    _fig
    return (analyses,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Random normal perturbation
    """)
    return


@app.cell
def _(
    current_lower,
    current_upper,
    norm_neighbour,
    np,
    num_analyses,
    num_tests,
    rngt,
    sigma_vector,
    tri_params,
):
    bounds_collector1 = []
    sample_size_collector1 = []

    tmp1 = np.concatenate((tri_params, [20]))

    for _i in range(num_tests):

        tmp1 = norm_neighbour(
            tmp1,
            num_analyses.value,
            rng = rngt,
            sigma_vector = sigma_vector,
            lower_search_bounds = current_lower,
            upper_search_bounds = current_upper
        )

        bound1 = tmp1[0:(num_analyses.value*2)-1]
        sample_size_collector1.append(tmp1[(num_analyses.value*2)-1])
        bounds_collector1.append(bound1.tolist())
    return (bounds_collector1,)


@app.cell
def _(analyses, bounds_collector1, fmt_bd, num_analyses, num_tests, plt, tri):
    _fig, _ax = plt.subplots()

    for _i in range(num_tests):
        _bounds = fmt_bd.reverse_to_boundaries(bounds_collector1[_i], K=num_analyses.value)
        _ax.plot(analyses, _bounds[0], color = "purple", alpha = 0.1)
        _ax.plot(analyses, _bounds[1], color = "purple", alpha = 0.1)

    _ax.plot(analyses, tri[0], color = "red", lw = 2)
    _ax.plot(analyses, tri[1], color = "red", lw = 2)

    _fig
    return


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
    upper_labels = [f"upper{i+1}" for i in range(num_analyses.value)]
    lower_labels = [f"lower{i+1}" for i in range(num_analyses.value - 1)]

    # labels will be used again in the experiment loop
    labels = upper_labels + lower_labels

    ordered_keys = ["index"] + labels + [
        "alpha", "power", "sample_size", "max_ess", 
        "obj_func", "temperature", "execute_time", "seed"
    ]

    box_values_collection = {key: [] for key in ordered_keys}
    return box_values_collection, labels


@app.cell
def _(box_values_collection, label_range, n_experiments, n_loops):
    # generate the indices using the pattern described above
    index_list = [
        i 
        for start in range(label_range, (n_experiments+1)*label_range, label_range) 
        for i in range(start + 1, start + (n_loops+1))
    ]

    box_values_collection["index"] = index_list
    return


@app.cell
def _(box_values_collection, n_experiments, n_loops, np):
    # create a list of seeds to use
    seed_list = [] # for filling the dictionary
    short_seed_list = [] # for using in the loop

    for _ in range(n_experiments):
        # get entropy for the random number generator seed
        seed = np.random.SeedSequence().entropy
        short_seed_list.append(seed)
        seeds = np.repeat(seed, n_loops)
        seed_list += seeds.tolist()

    box_values_collection["seed"] = seed_list
    return (short_seed_list,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Experiment initiation
    """)
    return


@app.cell
def _(mo):
    norm_neighbour_on = mo.ui.switch(label="norm_neighbour( )")
    return (norm_neighbour_on,)


@app.cell
def _(mo, norm_neighbour_on):
    mo.vstack([norm_neighbour_on, mo.md(f"Has value: {norm_neighbour_on.value}")])
    return


@app.cell
def _(
    box_values_collection,
    current_lower,
    current_upper,
    delta0,
    delta1,
    fmt_bd,
    labels,
    mu,
    n_experiments,
    n_loops,
    norm_neighbour,
    norm_neighbour_on,
    np,
    num_analyses,
    obj_f,
    rand_neighbour,
    short_seed_list,
    sigma2,
    sigma_vector,
    target_alpha,
    target_power,
    time,
    tri_obj,
):
    for i in range(n_experiments):

        # initialize the rng
        rng = np.random.default_rng(seed = short_seed_list[i])

        # initialise the simulated annealing
        # _init_triang is the below line:
        # initial_params = np.concatenate((tri_params, [tri_n_patients]))

        # _init_rand is the below code chunk:
        initial_params = rng.uniform(
            current_lower,
            current_upper,
            size = len(current_lower)
        )

        f_value = tri_obj.copy()

        # initial values for temperature and standard deviations
        temperature_start = 50
        sigma_vector_start = sigma_vector.copy()

        # small sigma test:
        # sigma_vector_start = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 15])

        # start-end test:
        # sigma_vector_end = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 2])

        f_min = f_value.copy()
        best_design = initial_params.copy()
        current_design = best_design.copy()

        start_time = time.time()

        # there are n_loops number of reverse_bounds to iterate through
        for j in range(n_loops):

            # reduce the sigma vector values
            # this is a linear decay
            sigma_vector_use = sigma_vector_start * (1 - (j/n_loops))
            # below is an exponential decay
            # sigma_vector_use = sigma_vector_start * (sigma_vector_end/sigma_vector_start)**(j/n_loops)

            # generate a new design with neighbour
            if norm_neighbour_on.value:
                candidate_design = norm_neighbour(
                    params = current_design,
                    K = num_analyses.value,
                    rng = rng,
                    sigma_vector = sigma_vector_use,
                    lower_search_bounds = current_lower,
                    upper_search_bounds = current_upper
                )
            else:
                candidate_design = rand_neighbour(params = current_design, K = num_analyses.value, rng = rng)

            # get its characteristics and calculate its penalty
            candidate_bounds = fmt_bd.reverse_to_boundaries(
                params = candidate_design[0:(num_analyses.value*2)-1], 
                K = num_analyses.value
            )
            candidate_n = candidate_design[(num_analyses.value*2)-1]

            # calculate its new function value
            alpha, power, max_ess, f_new = obj_f(
                mu = mu,
                upper_bounds = candidate_bounds[0],
                lower_bounds = candidate_bounds[1],
                n_analyses = num_analyses.value,
                n_patients = candidate_n,
                target_power = target_power,
                target_alpha = target_alpha,
                null_hypothesis = delta0,
                alternative_hypothesis = delta1,
                variance = sigma2
            )

            # reduce the temperature
            temperature = temperature_start * (1 - (j/n_loops))

            uniform_selector = rng.uniform(size = 1)

            if np.exp( -1*(f_new - f_value)/temperature ) >= uniform_selector:
                f_value = f_new.copy()
                current_design = candidate_design.copy()

                if f_new < f_min:
                    best_design = current_design.copy()
                    f_min = f_new.copy()

            # save the boundaries for analysis
            bounds_list = np.concatenate( (candidate_bounds[0], candidate_bounds[1][0:num_analyses.value-1]) )

            # collect the boundaries using the labels
            for _i in range(len(bounds_list)):
                box_values_collection[labels[_i]].extend([bounds_list[_i]])

            # collect the rest of the value of interest
            box_values_collection["alpha"].extend([alpha])
            box_values_collection["power"].extend([power])
            box_values_collection["sample_size"].extend([current_design[(num_analyses.value*2)-1]])
            box_values_collection["max_ess"].extend([max_ess])
            box_values_collection["obj_func"].extend([f_new])
            box_values_collection["temperature"].extend([temperature])

            # collect the best design and f_min
            best_bounds = fmt_bd.reverse_to_boundaries(
                params = best_design[0:(num_analyses.value*2)-1], 
                K = num_analyses.value
            )
            best_bounds_list = np.concatenate( (best_bounds[0], best_bounds[1][0:num_analyses.value-1]) )
            best_n = best_design[(num_analyses.value*2)-1]

            if j % 25 == 0:
                print(".", end = "")

        stop_time = time.time()
        execute_time = stop_time - start_time

        time_list = np.repeat(execute_time, n_loops)
        time_list += time_list.tolist()
        box_values_collection["execute_time"].extend(time_list)

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
def _(box_values_collection, pd):
    box_collections_df = pd.DataFrame(box_values_collection)
    return (box_collections_df,)


@app.cell
def _(box_collections_df):
    box_collections_df
    return


@app.cell
def _(box_collections_df):
    box_collections_df.to_csv(
        "/tf/experiments_rand_simann_bo/simulated_annealing_experiments/large_box_t50_rnorm_50x500.csv")
    return


if __name__ == "__main__":
    app.run()
