import marimo

__generated_with = "0.21.1"
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

    return bd, fn_min, fp, sim, ss


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
    sigma2 = 3.

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


@app.cell
def _(np):
    def reverse_to_boundaries(params, K):
        params = np.asarray(params).flatten()
        c = params[0]

        delta_u = params[1::2][::-1]
        delta_l = params[2::2][::-1]

        upper_bounds = np.array([c + np.sum(delta_u[k:]) for k in range(K)])
        lower_bounds = np.array([c - np.sum(delta_l[k:]) for k in range(K)])

        return upper_bounds, lower_bounds

    def boundaries_to_reverse(upper_bounds, lower_bounds):
        upper_bounds = np.asarray(upper_bounds)
        lower_bounds = np.asarray(lower_bounds)

        K = len(upper_bounds)
        c = upper_bounds[-1]

        delta_u = np.diff(upper_bounds[::-1])
        delta_l = np.diff(lower_bounds)[::-1]

        increments = np.empty(2 * (K - 1))
        increments[0::2] = delta_u
        increments[1::2] = delta_l

        return np.concatenate([[c], increments])

    return boundaries_to_reverse, reverse_to_boundaries


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
def _(
    bd,
    boundaries_to_reverse,
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
        delta = delta1,
        n_patients = 20
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

    tri_params = boundaries_to_reverse(
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


@app.cell
def _(c0, mo, np, tri_params):
    lower_dropdown = mo.ui.dropdown(
        options={
            'large_box' : np.array([c0 - 3.0, 0.0, 0.0, 0.0, 0.0, 2]),
            'small_box' : np.array([c0 - 1, 0.0, 0.0, 0.0, 0.0, 2]),
            'triang_box' : [max(0, param - 0.4) for param in tri_params] + [2]
        },
        value="triang_box",
        label="Choose lower search space:"
    )

    upper_dropdown = mo.ui.dropdown(
        options={
            'large_box' : np.array([c0 + 3.0, 4.0, 4.0, 4.0, 4.0, 100]),
            'small_box' : np.array([c0 + 1, 1.0, 4.0, 1.0, 1.0, 100]),
            'triang_box' : [param + 0.4 for param in tri_params] + [100]
        }, 
        value="triang_box",
        label="Choose upper search space:"
    )
    return lower_dropdown, upper_dropdown


@app.cell
def _(lower_dropdown, mo, np):
    mo.vstack(
        [
            lower_dropdown, 
            mo.md(f"Has value: {np.round(lower_dropdown.value, decimals = 3)}")
        ]
    )
    return


@app.cell
def _(mo, np, upper_dropdown):
    mo.vstack(
        [
            upper_dropdown, 
            mo.md(f"Has value: {np.round(upper_dropdown.value, decimals = 3)}")
        ]
    )
    return


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
    1. We generate the bounds from the box as is.
    2. We modify the current bounds with some small perturbation.
    """)
    return


@app.cell
def _(np):
    rngt = np.random.default_rng(seed = 5320945)
    return (rngt,)


@app.cell
def _(lower_dropdown, np, upper_dropdown):
    # the neighbour function
    def neighbour1(params, K, rng):

        modifying_params = params.copy()

        # generates discrete uniform values from 0 to 5
        idx_to_change = rng.integers(low=0, high=6)

        # depending on the index, the magnitude of change varies
        # that is, much be different for bounds vs. sample size
        num_of_params = K*2
        bound_selected = idx_to_change < num_of_params-1

        if bound_selected:
            new_param = modifying_params[idx_to_change] + rng.uniform(low = -0.5, high = 0.5)
            new_param = np.clip(new_param, lower_dropdown.value[idx_to_change], upper_dropdown.value[idx_to_change])
            modifying_params[idx_to_change] = new_param
        else:
            new_param = modifying_params[idx_to_change] + rng.uniform(low = -2.5, high = 2.5)
            new_param = np.clip(new_param, lower_dropdown.value[idx_to_change], upper_dropdown.value[idx_to_change])
            modifying_params[idx_to_change] = new_param

        #print(params)
        return modifying_params

    return (neighbour1,)


@app.cell
def _(lower_dropdown, upper_dropdown):
    # the neighbour function
    def neighbour2(params, K, rng):

        modifying_params = params.copy()

        # generates discrete uniform values from 0 to 5
        idx_to_change = rng.integers(low=0, high=6)

        modifying_params[idx_to_change] = rng.uniform(
            low = lower_dropdown.value[idx_to_change], 
            high = upper_dropdown.value[idx_to_change]
        )

        return modifying_params

    return (neighbour2,)


@app.cell
def _(np, tri_params):
    np.concatenate((tri_params, [20]))
    return


@app.cell
def _(neighbour1, np, rngt, tri_params):
    neighbour1(np.concatenate((tri_params, [20])), 3, rng = rngt)
    return


@app.cell
def _(neighbour1, np, rngt, tri_params):
    collector_b1 = []
    collector_n1 = []
    num_tests = 2000
    tmp = np.concatenate((tri_params, [20]))
    for _i in range(num_tests):
        tmp = neighbour1(tmp, 3, rng = rngt)
        bound = tmp[0:5]
        collector_n1.append(tmp[5])
        collector_b1.append(bound.tolist())
    return collector_b1, collector_n1, num_tests


@app.cell
def _(collector_b1, num_analyses, num_tests, plt, reverse_to_boundaries, tri):
    _fig, _ax = plt.subplots()

    analyses = [i+1 for i in range(num_analyses)]

    for _i in range(num_tests):
        _bounds = reverse_to_boundaries(collector_b1[_i], K=3)
        _ax.plot(analyses, _bounds[0], color = "purple", alpha = 0.1)
        _ax.plot(analyses, _bounds[1], color = "purple", alpha = 0.1)

    _ax.plot(analyses, tri[0], color = "red", lw = 2)
    _ax.plot(analyses, tri[1], color = "red", lw = 2)

    _fig
    return


@app.cell
def _(np, tri_params):
    np.concatenate((tri_params, [20]))
    return


@app.cell
def _(collector_b1, collector_n1, slider1):
    print(collector_b1[slider1.value], collector_n1[slider1.value])
    return


@app.cell
def _(collector_b1, collector_n1, slider1):
    print(collector_b1[slider1.value+1], collector_n1[slider1.value+1])
    return


@app.cell
def _(mo, num_tests):
    # what is a neighbour?
    slider1 = mo.ui.slider(start=0, stop=num_tests)
    slider1
    return (slider1,)


@app.cell
def _(collector_b1, num_analyses, plt, reverse_to_boundaries, slider1, tri):
    _fig, _ax = plt.subplots()

    _analyses = [i+1 for i in range(num_analyses)]


    _bounds = reverse_to_boundaries(collector_b1[slider1.value], K=3)
    _ax.plot(_analyses, _bounds[0], color = "purple")
    _ax.plot(_analyses, _bounds[1], color = "purple")

    _ax.plot(_analyses, tri[0], color = "red", lw = 2)
    _ax.plot(_analyses, tri[1], color = "red", lw = 2)

    _fig
    return


@app.cell
def _(collector_n1, plt):
    plt.hist(collector_n1, bins = 100)
    return


@app.cell
def _(neighbour2, np, rngt, tri_params):
    collector_b2 = []
    collector_n2 = []
    num_tests2 = 2000
    tmp1 = np.concatenate((tri_params, [20]))
    for _i in range(num_tests2):
        tmp1 = neighbour2(tmp1, 3, rng = rngt)
        bound1 = tmp1[0:5]
        collector_n2.append(tmp1[5])
        collector_b2.append(bound1.tolist())
    return collector_b2, collector_n2, num_tests2


@app.cell
def _(collector_b2, num_analyses, num_tests, plt, reverse_to_boundaries, tri):
    _fig, _ax = plt.subplots()

    _analyses = [i+1 for i in range(num_analyses)]

    for _i in range(num_tests):
        _bounds = reverse_to_boundaries(collector_b2[_i], K=3)
        _ax.plot(_analyses, _bounds[0], color = "purple", alpha = 0.1)
        _ax.plot(_analyses, _bounds[1], color = "purple", alpha = 0.1)

    _ax.plot(_analyses, tri[0], color = "red", lw = 2)
    _ax.plot(_analyses, tri[1], color = "red", lw = 2)

    _fig
    return


@app.cell
def _(collector_b2, collector_n2, slider2):
    print(collector_b2[slider2.value], collector_n2[slider2.value])
    return


@app.cell
def _(collector_b2, collector_n2, slider2):
    print(collector_b2[slider2.value+1], collector_n2[slider2.value+1])
    return


@app.cell
def _(mo, num_tests2):
    # what is a neighbour?
    slider2 = mo.ui.slider(start=0, stop=num_tests2)
    slider2
    return (slider2,)


@app.cell
def _(collector_b2, num_analyses, plt, reverse_to_boundaries, slider2, tri):
    _fig, _ax = plt.subplots()

    _analyses = [i+1 for i in range(num_analyses)]


    _bounds = reverse_to_boundaries(collector_b2[slider2.value], K=3)
    _ax.plot(_analyses, _bounds[0], color = "purple")
    _ax.plot(_analyses, _bounds[1], color = "purple")

    _ax.plot(_analyses, tri[0], color = "red", lw = 2)
    _ax.plot(_analyses, tri[1], color = "red", lw = 2)

    _fig
    return


@app.cell
def _(collector_n2, plt):
    plt.hist(collector_n2, bins = 100)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data collection setup
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Experiment initiation
    """)
    return


@app.cell
def _():
    n_experiments = 50
    n_loops = 500
    return n_experiments, n_loops


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
        "obj_func", "temperature", "execute_time", "seed"
    ]

    box_values_collection = {key: [] for key in ordered_keys}
    return box_values_collection, labels


@app.cell
def _(labels):
    # to save the best design and f_min values
    keys = ["index"] + labels + ["best_n", "f_min"]

    best_values = {key: [] for key in keys}
    return (best_values,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stop!
    If more than 1000 loops are to be run, the index labels in the following code block must be corrected.
    """)
    return


@app.cell
def _(best_values, box_values_collection, n_experiments, n_loops):
    # generate the indices using the pattern described above
    index_list = [
        i 
        for start in range(1000, (n_experiments+1)*1000, 1000) 
        for i in range(start + 1, start + (n_loops+1))
    ]

    box_values_collection["index"] = index_list
    best_values["index"] = index_list
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


@app.cell
def _(
    best_values,
    box_values_collection,
    delta0,
    delta1,
    labels,
    lower_dropdown,
    mu,
    n_experiments,
    n_loops,
    neighbour1,
    np,
    num_analyses,
    obj_f,
    reverse_to_boundaries,
    short_seed_list,
    sigma2,
    target_alpha,
    target_power,
    time,
    tri_obj,
    upper_dropdown,
):
    for i in range(n_experiments):

        # initialize the rng
        rng = np.random.default_rng(seed = short_seed_list[i])

        # initialise the simulated annealing
        # _init_triang is the below line:
        # initial_params = np.concatenate((tri_params, [tri_n_patients]))

        # _init_rand is the below code chunk:
        initial_params = rng.uniform(
            lower_dropdown.value,
            upper_dropdown.value,
            size = len(lower_dropdown.value)
        )

        f_value = tri_obj.copy()

        temperature_start = 100

        f_min = f_value.copy()
        best_design = initial_params.copy()
        current_design = best_design.copy()

        start_time = time.time()

        # there are n_loops number of reverse_bounds to iterate through
        for j in range(n_loops):

            # generate a new design with neighbour
            candidate_design = neighbour1(params = current_design, K = num_analyses, rng = rng)

            # get its characteristics and calculate its penalty
            candidate_bounds = reverse_to_boundaries(params = candidate_design[0:5], K = num_analyses)
            candidate_n = candidate_design[5]

            # calculate its new function value
            alpha, power, max_ess, f_new = obj_f(
                mu = mu,
                upper_bounds = candidate_bounds[0],
                lower_bounds = candidate_bounds[1],
                n_analyses = num_analyses,
                n_patients = candidate_n,
                target_power = target_power,
                target_alpha = target_alpha,
                null_hypothesis = delta0,
                alternative_hypothesis = delta1,
                variance = sigma2
            )

            # change the temperature
            temperature = temperature_start * (1 - (j/n_loops))

            uniform_selector = rng.uniform(size = 1)

            if np.exp( -1*(f_new - f_value)/temperature ) >= uniform_selector:
                f_value = f_new.copy()
                current_design = candidate_design.copy()

                if f_new < f_min:
                    best_design = current_design.copy()
                    f_min = f_new.copy()

            # save the boundaries for analysis
            bounds_list = np.concatenate( (candidate_bounds[0], candidate_bounds[1][0:2]) )

            # collect the boundaries using the labels
            for _i in range(len(bounds_list)):
                box_values_collection[labels[_i]].extend([bounds_list[_i]])

            # collect the rest of the value of interest
            box_values_collection["alpha"].extend([alpha])
            box_values_collection["power"].extend([power])
            box_values_collection["sample_size"].extend([current_design[5]])
            box_values_collection["max_ess"].extend([max_ess])
            box_values_collection["obj_func"].extend([f_value])
            box_values_collection["temperature"].extend([temperature])

            # collect the best design and f_min
            best_bounds = reverse_to_boundaries(params = best_design[0:5], K = num_analyses)
            best_bounds_list = np.concatenate( (best_bounds[0], best_bounds[1][0:2]) )
            best_n = best_design[5]

            for _i in range(len(best_bounds_list)):
                best_values[labels[_i]].extend([best_bounds_list[_i]])

            best_values["best_n"].extend([best_n])
            best_values["f_min"].extend([f_min])

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


@app.cell
def _(best_values, pd):
    best_values_df = pd.DataFrame(best_values)
    return (best_values_df,)


@app.cell
def _(best_values_df, np):
    np.unique(best_values_df["f_min"])[0]
    return


@app.cell
def _(best_values_df, np):
    best_values_df[best_values_df["f_min"] == np.unique(best_values_df["f_min"])[0]].iloc[0, :]
    return


@app.cell
def _(box_values_collection, pd):
    box_collections_df = pd.DataFrame(box_values_collection)
    return (box_collections_df,)


@app.cell
def _(best_values_df):
    best_values_df.to_csv("/tf/2026-04-t21/simulated_annealing_experiments/large_box_t100_neigh1_init_rand_best_vals.csv")
    return


@app.cell
def _(box_collections_df):
    box_collections_df.to_csv("/tf/2026-04-t21/simulated_annealing_experiments/large_box_t100_neigh1_init_rand_results.csv")
    return


if __name__ == "__main__":
    app.run()
