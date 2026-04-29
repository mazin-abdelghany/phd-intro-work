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

    return np, pd, stats, time


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
    return (tri_params,)


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
def _():
    #############
    # large box #
    #############
    # lower = np.array([c0 - 3.0, 0.0, 0.0, 0.0, 0.0])
    # upper = np.array([c0 + 3.0, 4.0, 4.0, 4.0, 4.0])

    #############
    # small box #
    #############
    # lower = np.array([c0 - 1, 0.0, 0.0, 0.0, 0.0])
    # upper = np.array([c0 + 1, 1.0, 4.0, 1.0, 1.0])
    return


@app.cell
def _(np, tri_params):
    ###################
    # near triangular #
    ###################
    lower = []
    upper = []

    for param in tri_params:
        lower.append(max(0, param - 0.4))
        upper.append(param + 0.4)

    print(f"Lower: {np.round(lower, 3)}")
    print(f"Upper: {np.round(upper, 3)}")
    return lower, upper


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
        "obj_func", "execute_time", "seed"
    ]

    random_search_large_box = {key: [] for key in ordered_keys}
    return labels, random_search_large_box


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stop!
    If more than 1000 loops are to be run, the index labels in the following code block must be corrected.
    """)
    return


@app.cell
def _(n_experiments, n_loops, random_search_large_box):
    # generate the indices using the pattern described above
    index_list = [
        i 
        for start in range(1000, (n_experiments+1)*1000, 1000) 
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


@app.cell
def _():
    # generate a distribution from which to sample sample sizes
    loc = 20
    scale = 10

    # calculate the bonuds of the truncated normal
    a = (9 - loc) / scale
    b = (50 - loc) / scale
    return a, b, loc, scale


@app.cell
def _(
    a,
    b,
    delta0,
    delta1,
    labels,
    loc,
    lower,
    mu,
    n_experiments,
    n_loops,
    np,
    num_analyses,
    obj_f,
    random_search_large_box,
    reverse_to_boundaries,
    scale,
    short_seed_list,
    sigma2,
    stats,
    target_alpha,
    target_power,
    time,
    upper,
):
    for i in range(n_experiments):
    
        # initialize the rng
        rng = np.random.default_rng(seed = short_seed_list[i])

        # generate n_loops # of reverse bounds, array shape is (n_loops x 5)
        reverse_bounds = rng.uniform(lower, upper, size = (n_loops, len(lower)))

        # generate n_loops sample sizes from the truncated normal
        sample_size = stats.truncnorm.rvs(size = n_loops, 
                                          a = a, 
                                          b = b, 
                                          loc = loc, 
                                          scale = scale,
                                          random_state = rng)

        start_time = time.time()

        # there are n_loops number of reverse_bounds to iterate through
        for j, boundaries in enumerate(reverse_bounds):

            bounds = reverse_to_boundaries(params = boundaries, K = num_analyses)
            bounds_list = np.concatenate( (bounds[0], bounds[1][0:2]) )

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


@app.cell
def _(pd, random_search_large_box):
    pd.DataFrame(random_search_large_box).to_csv("/tf/2026-04-t20/random_search_experiments/triagular_box.csv")
    return


if __name__ == "__main__":
    app.run()
