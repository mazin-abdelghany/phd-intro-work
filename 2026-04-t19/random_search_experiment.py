import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


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
    delta0 = 0
    delta1 = 1.0
    sigma2 = 3.0

    mu = ss.sample_size_means(
        ratio=1,
        variance=sigma2,
        power=target_power,
        alpha=target_alpha,
        delta=delta1
    )
    print(f"Single-stage sample size mu = {mu:.2f}")
    return delta0, delta1, mu, num_analyses, sigma2, target_alpha, target_power


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reverse parameterization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    BO vector: $(c, \Delta u_3, \Delta l_3, \Delta u_2, \Delta l_2)$
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
    # Test the reverse functions
    """)
    return


@app.cell
def _(bd):
    po_bounds = bd.calculate_pocock_boundaries()
    return (po_bounds,)


@app.cell
def _(po_bounds):
    po_bounds[0]
    return


@app.cell
def _(boundaries_to_reverse, po_bounds):
    boundaries_to_reverse(po_bounds[0], po_bounds[1])
    return


@app.cell
def _(np, po_bounds):
    np.diff(po_bounds[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Objective function
    """)
    return


@app.cell
def _(delta0, delta1, fn_min, fp, num_analyses, sigma2, sim, ss):
    def obj_f(
            mu,
            upper_bounds,
            lower_bounds,
            n_analyses,
            target_power,
            target_alpha):

        n_power09, calc_power = ss.find_sample_size(
            power_target = target_power,
            n_analyses = n_analyses,
            upper_bounds = upper_bounds,
            lower_bounds = lower_bounds,
            null_hypothesis = delta0,
            alt_hypothesis = delta1,
            variance = sigma2
        )

        beta_prime = 1-calc_power

        alpha_prime = sim.group_sequential_designs(
            n_analyses = num_analyses,
            upper_bounds = upper_bounds,
            lower_bounds = lower_bounds,
            n_patients = n_power09, 
            null_hypothesis = delta0,
            alt_hypothesis = delta1,
            variance = sigma2
        )[1]

        max_ess = ss.max_ess(
            n_analyses = n_analyses,
            upper_bounds = upper_bounds,
            lower_bounds = lower_bounds,
            n_patients = n_power09
        )

        penalty = fp.smooth_penalty(
            mu = mu,
            power = target_power,
            alpha = target_alpha,
            beta_prime = beta_prime,
            alpha_prime = alpha_prime
        )

        f_val = fn_min.function_to_minimize(max_ess_val=max_ess/mu, penalty=penalty)

        return (
            alpha_prime,
            calc_power,
            n_power09,
            f_val
        )

    return (obj_f,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Objective function value - triangular
    """)
    return


@app.cell
def _(
    bd,
    boundaries_to_reverse,
    delta1,
    mu,
    np,
    num_analyses,
    obj_f,
    target_alpha,
    target_power,
):
    tri = bd.calculate_triangular_boundaries(
        n_analyses=num_analyses,
        alpha=target_alpha,
        delta=delta1,
        n_patients=20
    )

    _,_,_,tri_obj = obj_f(
        mu = mu,
        upper_bounds = tri[0],
        lower_bounds = tri[1],
        n_analyses = num_analyses,
        target_power = target_power,
        target_alpha = target_alpha
    )

    tri_params = boundaries_to_reverse(tri[0], tri[1])
    c0 = tri_params[0]

    print(f"Triangular benchmark objective: {tri_obj:.4f}")
    print(f"Original trriangular params: {np.round(np.concatenate((tri[0], tri[1])), 4)}")
    print(f"Reparameterized triangular params: {np.round(tri_params, 4)}")
    print(f"Meeting point c0 = {c0:.4f}")
    return c0, tri, tri_obj


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Search space
    """)
    return


@app.cell
def _(c0, np):
    lower = np.array([c0 - 3.0, 0.0, 0.0, 0.0, 0.0])
    upper = np.array([c0 + 3.0, 4.0, 4.0, 4.0, 4.0])
    return lower, upper


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Random search experiment - 1 loop example
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Values of interest
    - Minimum objective function value
    - Index at which minimum objective function value is found
    - Is minimum objective function value lower than triagular design
    - Proportion of designs meeting target $\alpha$ and target $1-\beta$
    - Proportion of designs within margin of error $\epsilon$ of target $\alpha$ and target $1-\beta$
    """)
    return


@app.cell
def _(np):
    # set the seed 
    rng_baseline = np.random.default_rng(seed=123)
    return (rng_baseline,)


@app.cell
def _():
    # ensure that this matches the total # of Bayes opt evals
    n_baseline = 1000
    return (n_baseline,)


@app.cell
def _(lower, n_baseline, rng_baseline, upper):
    bounds_to_test = rng_baseline.uniform(lower, upper, size=(n_baseline, len(lower)))
    return (bounds_to_test,)


@app.cell
def _(
    bounds_to_test,
    mu,
    num_analyses,
    obj_f,
    reverse_to_boundaries,
    target_alpha,
    target_power,
    time,
):
    alphas = []
    powers = []
    sample_sizes = []
    baseline_objs = []

    # start a timer
    start_time = time.time()

    i = 1
    for bounds in bounds_to_test:

        actual_bounds = reverse_to_boundaries(bounds, K = num_analyses)

        alpha, power, sample_size, y_val = obj_f(
            mu = mu,
            upper_bounds = actual_bounds[0],
            lower_bounds = actual_bounds[1],
            n_analyses = num_analyses,
            target_power = target_power,
            target_alpha = target_alpha
        )

        alphas.append(alpha)
        powers.append(power)
        sample_sizes.append(sample_size)
        baseline_objs.append(y_val)

        if i % 100 == 0:
            print(f"Completed loop {i}.")

        i = i + 1

    # end the timer
    end_time = time.time()
    execution_time = end_time - start_time
    return alphas, baseline_objs, execution_time, powers, sample_sizes


@app.cell
def _(
    alphas,
    baseline_objs,
    execution_time,
    n_baseline,
    np,
    powers,
    sample_sizes,
    target_alpha,
    target_power,
    tri_obj,
):
    # turn the lists into numpy arrays
    alphas_np = np.array(alphas)
    powers_np = np.array(powers)
    sample_sizes_np = np.array(sample_sizes)
    baseline_objs_np = np.array(baseline_objs)

    # get values of interest noted above
    obj_min = np.min(baseline_objs_np)
    obj_min_index = np.argmin(baseline_objs_np)
    obj_min_less_tri = obj_min < tri_obj

    # how many designs meet alpha 0.05 and power 0.9
    design_goal_met = (alphas_np <= target_alpha) & (powers_np >= target_power - 0.05)

    # how many designs are within epsilon of alpha and power
    epsilon = 0.01
    within_epsilon = ( (alphas_np <= (target_alpha + epsilon)) & (alphas_np >= (target_alpha - epsilon)) )

    print(f"Random search:    {n_baseline} evaluations")
    print(f"Loop took:        {execution_time/60:.1f} min")
    print(f"Best overall f:   {obj_min:.4f}")
    print(f"Best index:       {obj_min_index}")
    print(f"Feasible:         {np.sum(design_goal_met)}/{n_baseline} ({100*np.mean(design_goal_met):.1f}%)")
    print(f"Feasible epsil:   {np.sum(within_epsilon)}/{n_baseline} ({100*np.mean(within_epsilon):.1f}%)")
    print(f"Better than tri?  {obj_min_less_tri}")
    return alphas_np, obj_min_index, powers_np, sample_sizes_np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The best boundary characteristics
    """)
    return


@app.cell
def _(
    alphas_np,
    bounds_to_test,
    num_analyses,
    obj_min_index,
    powers_np,
    reverse_to_boundaries,
    sample_sizes_np,
):
    print(f"The best boundary: {reverse_to_boundaries(bounds_to_test[obj_min_index], K = num_analyses)}")
    print(f"Its alpha:         {alphas_np[obj_min_index]}")
    print(f"Its power:         {powers_np[obj_min_index]}")
    print(f"Its sample size:   {sample_sizes_np[obj_min_index]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plot the best boundary compared to triangular
    """)
    return


@app.cell
def _(
    bounds_to_test,
    num_analyses,
    obj_min_index,
    plt,
    reverse_to_boundaries,
    tri,
):
    _fig, _ax = plt.subplots()

    _ax.plot([1,2,3], tri[0], color = "red", lw = 2)
    _ax.plot([1,2,3], tri[1], color = "red", lw = 2)

    _ax.plot([1,2,3], reverse_to_boundaries(bounds_to_test[obj_min_index], K = num_analyses)[0],
             color = "blue")
    _ax.plot([1,2,3], reverse_to_boundaries(bounds_to_test[obj_min_index], K = num_analyses)[1],
             color = "blue")

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Run random search 100 times
    """)
    return


@app.cell
def _(
    lower,
    mu,
    np,
    num_analyses,
    obj_f,
    reverse_to_boundaries,
    rng_baseline,
    target_alpha,
    target_power,
    time,
    upper,
):
    # how many loops of n_baseline to run
    n_experiments = 50

    # ensure that this matches the total # of Bayes opt evals
    n_loops = 1000

    # values that we are collecting
    seeds = []
    alphas_list = []
    powers_list = []
    sample_sizes_list = []
    baseline_objs_list = []
    execution_times = []
    all_bounds = []

    j = 1
    for _i in range(n_experiments):
        # set the seed and save it
        seed = int(rng_baseline.uniform(low = 10, high = 100000))

        # make sure that we are not reusing seeds
        while seed in seeds: 
            print(f"Seed {seed} has already been used.")
            seed = int(rng_baseline.uniform(low = 100, high = 10000))
            print(f"New seed {seed} set!")

        # collect the seeeds used
        seeds.append(seed)

        # initialize the rng with this new seed
        rng_in_loop = np.random.default_rng(seed = seed)

        bounds_to_test_loop = rng_in_loop.uniform(lower, upper, size = (n_loops, len(lower)))
        all_bounds.append(bounds_to_test_loop)

        ex_alphas = []
        ex_powers = []
        ex_sample_sizes = []
        ex_baseline_objs = []

        # start a timer
        _start_time = time.time()

        _i = 1
        for _bounds in bounds_to_test_loop:

            _actual_bounds = reverse_to_boundaries(_bounds, K = num_analyses)

            _alpha, _power, _sample_size, _y_val = obj_f(
                mu = mu,
                upper_bounds = _actual_bounds[0],
                lower_bounds = _actual_bounds[1],
                n_analyses = num_analyses,
                target_power = target_power,
                target_alpha = target_alpha
            )

            ex_alphas.append(_alpha)
            ex_powers.append(_power)
            ex_sample_sizes.append(_sample_size)
            ex_baseline_objs.append(_y_val)

            if _i % 200 == 0:
                print(f"Completed inside loop {_i}.")
            _i = _i + 1

        # end the timer
        _end_time = time.time()

        # fill the value collectors
        execution_times.append(_end_time - _start_time)
        alphas_list.append(ex_alphas)
        powers_list.append(ex_powers)
        sample_sizes_list.append(ex_sample_sizes)
        baseline_objs_list.append(ex_baseline_objs)

        if j % 10 == 0:
            print("=============================")
            print(f"= Completed experiment {j}. =")
            print("=============================")
        j = j + 1
    return (
        all_bounds,
        alphas_list,
        baseline_objs_list,
        execution_times,
        n_experiments,
        n_loops,
        powers_list,
        sample_sizes_list,
        seeds,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Summarize the data
    """)
    return


@app.cell
def _(
    alphas_list,
    execution_times,
    np,
    pd,
    powers_list,
    sample_sizes_list,
    seeds,
):
    random_search_large_box = pd.DataFrame(
        {
            'alpha_mean' : np.mean(alphas_list, axis=1), 
            'power_mean' : np.mean(powers_list, axis=1),
            'sample_size_mean' : np.mean(sample_sizes_list, axis=1),
            'execution_times_min' : np.array(execution_times)/60,
            'seed' : seeds
        }
    )
    return (random_search_large_box,)


@app.cell
def _(alphas_list_np, alphas_np, target_alpha):
    # how many designs are within epsilon of alpha and power
    epsilon = 0.01
    within_epsilon = ( (alphas_np <= (target_alpha + epsilon)) & (alphas_np >= (target_alpha - epsilon)) )

    _epsilon = 0.01
    _within_epsilon = ( (alphas_list_np[_i] <= (target_alpha + epsilon)) & (alphas_list_np[_i] >= (target_alpha - epsilon)) )
    return


@app.cell
def _(
    alphas_list,
    baseline_objs_list,
    execution_times,
    n_loops,
    np,
    powers_list,
    seeds,
    target_alpha,
    target_power,
    tri_obj,
):
    alphas_list_np = np.array(alphas_list)
    powers_list_np = np.array(powers_list)

    best_obj = []
    best_index = []
    better_than_tri = []
    design_met = []
    within_epsi = []

    for _i in range(len(seeds)):
        # get values of interest noted above
        _obj_min = np.min(baseline_objs_list[_i])
        _obj_min_index = np.argmin(baseline_objs_list[_i])
        _obj_min_less_tri = _obj_min < tri_obj

        # how many designs meet alpha 0.05 and power 0.9
        _design_goal_met = (alphas_list_np[_i] <= target_alpha) & (powers_list_np[_i] >= (target_power - 0.02))

        # how many designs are within epsilon of alpha and power
        _epsilon = 0.01
        _within_epsilon = ( (alphas_list_np[_i] <= (target_alpha + _epsilon)) & (alphas_list_np[_i] >= (target_alpha - _epsilon)) )

        print(f"Run {_i+1}:")
        print(f"Random search:    {n_loops} evaluations")
        print(f"Loop took:        {execution_times[_i]/60:.2f} min")
        print(f"Best overall f:   {_obj_min:.4f}")
        print(f"Best index:       {_obj_min_index}")
        print(f"Feasible:         {np.sum(_design_goal_met)}/{n_loops} ({100*np.mean(_design_goal_met):.1f}%)")
        print(f"Feasible epsil:   {np.sum(_within_epsilon)}/{n_loops} ({100*np.mean(_within_epsilon):.1f}%)")
        print(f"Better than tri?  {_obj_min_less_tri}")
        print("\n")

        best_obj.append(_obj_min)
        best_index.append(_obj_min_index)
        better_than_tri.append(_obj_min_less_tri)
        design_met.append(_design_goal_met[_i])
        within_epsi.append(_within_epsilon[_i])
    return (
        alphas_list_np,
        best_index,
        best_obj,
        better_than_tri,
        powers_list_np,
    )


@app.cell
def _(best_index, best_obj, better_than_tri, random_search_large_box):
    random_search_large_box['best_obj_val'] = best_obj
    random_search_large_box['best_obj_index'] = best_index
    random_search_large_box['better_than_tri'] = better_than_tri
    return


@app.cell
def _(random_search_large_box):
    random_search_large_box
    return


@app.cell
def _(best_obj, np):
    np.mean(best_obj)
    return


@app.cell
def _(best_index, np):
    np.median(best_index)
    return


@app.cell
def _(better_than_tri, np):
    np.mean(better_than_tri)
    return


@app.cell
def _(
    all_bounds,
    alphas_np,
    best_obj,
    num_analyses,
    powers_np,
    random_search_large_box,
    reverse_to_boundaries,
    sample_sizes_np,
    seeds,
    sim,
):
    for _i in range(len(seeds)):

        best_bounds = reverse_to_boundaries(
            all_bounds[_i][random_search_large_box['best_obj_index'][_i]],
            K = num_analyses
        )

        best_design_properties = sim.group_sequential_designs(
            n_analyses = num_analyses,
            upper_bounds = best_bounds[0],
            lower_bounds = best_bounds[1],
            n_patients = sample_sizes_np[random_search_large_box['best_obj_index'][_i]],
            null_hypothesis=0,
            alt_hypothesis=0.5,
            variance=1
        )

        print(f"Run {_i+1}:")
        print(f"The best boundary: {best_bounds}")
        print(f"Objective val:     {best_obj[_i]}")
        print(f"Alpha:             {alphas_np[random_search_large_box['best_obj_index'][_i]]}")
        print(f"Power:             {powers_np[random_search_large_box['best_obj_index'][_i]]}")
        print(f"Sample size:       {sample_sizes_np[random_search_large_box['best_obj_index'][_i]]}")
        print(f"Expected ss:       {best_design_properties[3]}")
        print("\n")
    return


@app.cell
def _(
    all_bounds,
    num_analyses,
    plt,
    random_search_large_box,
    reverse_to_boundaries,
    seeds,
    tri,
):
    _fig, _ax = plt.subplots()

    _ax.plot([1,2,3], tri[0], color = "red", lw = 2)
    _ax.plot([1,2,3], tri[1], color = "red", lw = 2)

    for _i in range(len(seeds)):
        _bounds = reverse_to_boundaries(
            all_bounds[_i][random_search_large_box['best_obj_index'][_i]],
            K = num_analyses
        )

        _ax.plot([i for i in range(1,num_analyses+1)], _bounds[0], color = "purple", alpha = 0.5)
        _ax.plot([i for i in range(1,num_analyses+1)], _bounds[1], color = "purple", alpha = 0.5)

    _fig
    return


@app.cell
def _(mo, n_experiments):
    slider = mo.ui.slider(start=0, stop=n_experiments-1)
    slider
    return (slider,)


@app.cell
def _(
    all_bounds,
    num_analyses,
    plt,
    random_search_large_box,
    reverse_to_boundaries,
    slider,
    tri,
):
    _fig, _ax = plt.subplots()

    _ax.plot([1,2,3], tri[0], color = "red", lw = 2)
    _ax.plot([1,2,3], tri[1], color = "red", lw = 2)

    _bounds = reverse_to_boundaries(
        all_bounds[slider.value][random_search_large_box['best_obj_index'][slider.value]],
        K = num_analyses
    )

    _ax.plot([i for i in range(1,num_analyses+1)], _bounds[0], color = "purple", alpha = 0.5)
    _ax.plot([i for i in range(1,num_analyses+1)], _bounds[1], color = "purple", alpha = 0.5)

    _ax.set_ylim(-7.5,12.5)

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Save some values
    seeds = []
    alphas_list = []
    powers_list = []
    sample_sizes_list = []
    baseline_objs_list = []
    execution_times = []
    """)
    return


@app.cell
def _(random_search_large_box):
    random_search_large_box.to_csv("random_search_large_box_50.csv")
    return


@app.cell
def _(alphas_list_np, pd):
    random_search_large_box_alphas = pd.DataFrame(alphas_list_np)
    random_search_large_box_alphas.index = [f'row_{i}' for i in range(1, 51)]
    random_search_large_box_alphas.to_csv("2026-04-t19/random_search_large_box_alphas_50.csv")
    return


@app.cell
def _(pd, powers_list_np):
    random_search_large_box_powers = pd.DataFrame(powers_list_np)
    random_search_large_box_powers.index = [f'row_{i}' for i in range(1, 51)]
    random_search_large_box_powers.to_csv("2026-04-t19/random_search_large_box_powers_50.csv")
    return


@app.cell
def _(pd, sample_sizes_list):
    random_search_large_box_sample_size = pd.DataFrame(sample_sizes_list)
    random_search_large_box_sample_size.index = [f'row_{i}' for i in range(1, 51)]
    random_search_large_box_sample_size.to_csv("2026-04-t19/random_search_large_box_sample_size_50.csv")
    return


@app.cell
def _(baseline_objs_list, pd):
    random_search_large_box_baseline_obj = pd.DataFrame(baseline_objs_list)
    random_search_large_box_baseline_obj.index = [f'row_{i}' for i in range(1, 51)]
    random_search_large_box_baseline_obj.to_csv("2026-04-t19/random_search_large_box_baseline_obj_50.csv")
    return


if __name__ == "__main__":
    app.run()
