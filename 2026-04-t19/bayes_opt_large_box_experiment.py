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

    return np, time


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

    return bd, fn_min, fp, sim, ss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Trial design settings
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
    # Objective function
    """)
    return


@app.cell
def _(
    delta0,
    delta1,
    fn_min,
    fp,
    np,
    num_analyses,
    reverse_to_boundaries,
    sigma2,
    sim,
    ss,
):
    def obj_f(
            mu,
            params,
            n_analyses,
            target_power,
            target_alpha):

        params = np.array(params)
        upper_bounds, lower_bounds = reverse_to_boundaries(params = params, K = n_analyses)

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
            max_ess,
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
def _(bd, delta1, num_analyses, target_alpha):
    tri = bd.calculate_triangular_boundaries(
        n_analyses=num_analyses,
        alpha=target_alpha,
        delta=delta1,
        n_patients=20
    )

    tri
    return (tri,)


@app.cell
def _(
    boundaries_to_reverse,
    mu,
    np,
    num_analyses,
    obj_f,
    target_alpha,
    target_power,
    tri,
):
    tri_params = boundaries_to_reverse(lower_bounds = tri[1], upper_bounds = tri[0])

    _,_,_,_,tri_obj = obj_f(
        mu = mu,
        params = tri_params,
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
    return c0, tri_obj


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Search space
    """)
    return


@app.cell
def _(Box, c0, np):
    lower = np.array([c0 - 3.0, 0.0, 0.0, 0.0, 0.0])
    upper = np.array([c0 + 3.0, 4.0, 4.0, 4.0, 4.0])

    search_space = Box(lower=lower, upper=upper)
    print(f"  lower: {np.round(search_space.lower.numpy(), 3)}")
    print(f"  upper: {np.round(search_space.upper.numpy(), 3)}")
    return (search_space,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Initialisation
    """)
    return


@app.cell
def _(bd, boundaries_to_reverse, num_analyses):
    _poc = bd.calculate_pocock_boundaries(
        n_analyses=num_analyses, alpha=0.05, n_patients=20
    )
    poc_params = boundaries_to_reverse(_poc[0], _poc[1])

    _obf = bd.calculate_of_boundaries(
        n_analyses=num_analyses, alpha=0.05, n_patients=20
    )
    obf_params = boundaries_to_reverse(_obf[0], _obf[1])
    return obf_params, poc_params


@app.cell
def _(
    mu,
    num_analyses,
    obf_params,
    obj_f,
    poc_params,
    target_alpha,
    target_power,
):
    _,_,_,_,poc_obj_f = obj_f(
        mu = mu, 
        params = poc_params,
        n_analyses = num_analyses,
        target_power = target_power,
        target_alpha = target_alpha
    )

    _,_,_,_,obf_obj_f = obj_f(
        mu = mu, 
        params = obf_params,
        n_analyses = num_analyses,
        target_power = target_power,
        target_alpha = target_alpha
    )
    return obf_obj_f, poc_obj_f


@app.cell
def _(np, obf_obj_f, obf_params, poc_obj_f, poc_params):
    design_matrix = np.concatenate((np.atleast_2d(poc_params), np.atleast_2d(obf_params)))
    output_vals = np.concatenate((np.atleast_2d(poc_obj_f), np.atleast_2d(obf_obj_f)))
    return design_matrix, output_vals


@app.cell
def _(design_matrix, output_vals, trieste):
    initial_data = trieste.data.Dataset(
        query_points = design_matrix,
        observations = output_vals
    )
    return (initial_data,)


@app.cell
def _(design_matrix, output_vals):
    print(f"Initial dataset:\n{design_matrix}\n")
    print(f"Initial f(x):\n{output_vals}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GP model
    """)
    return


@app.cell
def _(GaussianProcessRegression, design_matrix, gpflow, output_vals):
    _kernel = gpflow.kernels.Matern52(
        lengthscales=[1.0] * design_matrix.shape[1]
    )

    _gpr = gpflow.models.GPR(
        data      = (design_matrix, output_vals),
        kernel    = _kernel,
        likelihood = gpflow.likelihoods.Gaussian()
    )

    gpflow.utilities.print_summary(_gpr, fmt="notebook")
    bayes_opt_model = GaussianProcessRegression(_gpr)
    return (bayes_opt_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bayesian optimisation loop
    """)
    return


@app.cell
def _(bayes_opt_model, initial_data, search_space, trieste):
    ask_tell = trieste.ask_tell_optimization.AskTellOptimizer(
        search_space     = search_space,
        datasets         = initial_data,
        models           = bayes_opt_model,
        acquisition_rule = trieste.acquisition.rule.EfficientGlobalOptimization(
            optimizer = trieste.acquisition.optimizer.generate_continuous_optimizer(
                num_optimization_runs = 500
            )
        )
    )
    return (ask_tell,)


@app.cell
def _(
    ask_tell,
    mu,
    np,
    num_analyses,
    obj_f,
    target_alpha,
    target_power,
    time,
    trieste,
):
    num_repeats   = 1000
    when_to_print = 100
    n_design_goal_met = 0
    design_goal_met_list = []

    epsilon = 0.015
    n_within_epsilon = 0
    within_epsilon_list = []

    # start a timer
    start_time = time.time()

    for _i in range(num_repeats):
        x_new = ask_tell.ask()

        alpha_new, power_new, n_power_09_new, max_ess_new, y_new = obj_f(
            mu = mu, 
            params = x_new,
            n_analyses = num_analyses,
            target_power = target_power,
            target_alpha = target_alpha
        )

        # how many designs meet alpha 0.05 and power 0.9
        design_goal_met = (alpha_new <= target_alpha) & (power_new >= target_power - 0.05)
        if design_goal_met:
            n_design_goal_met += 1
        design_goal_met_list.append(design_goal_met)

        # how many designs are within epsilon of alpha and power
        within_epsilon = ( (alpha_new <= (target_alpha + epsilon)) & (alpha_new >= (target_alpha - epsilon)) )
        if within_epsilon:
            n_within_epsilon += 1
        within_epsilon_list.append(within_epsilon)

        ask_tell.tell(trieste.data.Dataset(
            query_points = x_new,
            observations = np.array([[y_new]])
        ))

        if (_i + 1) % when_to_print == 0:
            print(
                f"\nLoop {_i+1} completed. "
                f"Feasible: {n_design_goal_met}/{_i+1} "
                f"({100*n_design_goal_met/(_i+1):.0f}%).",
                end=""
             )
        elif (_i > when_to_print) and ((_i + 1) % 5 == 0):
            print(".", end="")

    # end the timer
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nDone. Feasible BO proposals: {n_design_goal_met}/{num_repeats}")
    return execution_time, n_within_epsilon, num_repeats, within_epsilon_list


@app.cell
def _(
    ask_tell,
    delta0,
    delta1,
    execution_time,
    n_within_epsilon,
    np,
    num_analyses,
    num_repeats,
    reverse_to_boundaries,
    sigma2,
    ss,
    target_power,
    tri_obj,
    within_epsilon_list,
):
    best_obj_f = np.min(ask_tell.to_result().try_get_final_dataset().observations[2:])

    best_idx = np.argmin(ask_tell.to_result().try_get_final_dataset().observations[2:])

    best_bounds = reverse_to_boundaries(
        ask_tell.to_result().try_get_final_dataset().query_points[2:][best_idx], 
        K=num_analyses
    )

    best_n09, _ = ss.find_sample_size(
        power_target = target_power,
        n_analyses = num_analyses,
        upper_bounds = best_bounds[0],
        lower_bounds = best_bounds[1],
        null_hypothesis = delta0,
        alt_hypothesis = delta1,
        variance = sigma2
    )

    print(f"Bayesian optim:       {num_repeats} evaluations")
    print(f"Loop took:            {execution_time/60:.1f} min")
    print(f"Best objective val:   {best_obj_f}")
    print(f"Best objective idx:   {best_idx}")
    print(f"Feasible epsilon:     {n_within_epsilon}/{num_repeats} ({100*np.mean(within_epsilon_list):.1f}%)")
    print(f"Best n w/power 0.9:   {best_n09}")
    print(f"Better than tri?      {best_obj_f < tri_obj}")
    return best_bounds, best_n09


@app.cell
def _(best_bounds):
    best_bounds
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Best trial design properties
    """)
    return


@app.cell
def _(best_bounds, best_n09, delta0, delta1, num_analyses, sigma2, sim):
    sim.group_sequential_designs(
        n_analyses = num_analyses,
        upper_bounds = best_bounds[0],
        lower_bounds = best_bounds[1],
        n_patients = best_n09,
        null_hypothesis = delta0,
        alt_hypothesis = delta1,
        variance = sigma2
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Run Bayes opt 1000 times
    """)
    return


@app.cell
def _(
    bayes_opt_model,
    delta0,
    delta1,
    initial_data,
    mu,
    np,
    num_analyses,
    obj_f,
    reverse_to_boundaries,
    search_space,
    sigma2,
    ss,
    target_alpha,
    target_power,
    time,
    tri_obj,
    trieste,
):
    # how many loops of n_baseline to run
    n_experiments = 50
    n_loops = 1000

    # collect some important values
    loop_observations = []
    loop_query_points = []
    loop_design_goal_met = []
    loop_within_epsilon = []
    loop_best_idx = []
    loop_best_obj_f = []
    loop_best_bounds = []
    loop_execution_time = []
    loop_best_n09 = []

    for _j in range(n_experiments):

        when_to_print_loop = 100
        n_design_goal_met_loop = 0
        design_goal_met_list_loop = []
    
        epsilon_loop = 0.015
        n_within_epsilon_loop = 0
        within_epsilon_list_loop = []
    
        # start a timer
        start_time_loop = time.time()

        # reset the ask_tell interface every experiment
        loop_ask_tell = trieste.ask_tell_optimization.AskTellOptimizer(
            search_space     = search_space,
            datasets         = initial_data,
            models           = bayes_opt_model,
            acquisition_rule = trieste.acquisition.rule.EfficientGlobalOptimization(
                optimizer = trieste.acquisition.optimizer.generate_continuous_optimizer(
                    num_optimization_runs = 500
                )
            )
        )
    
        for _i in range(n_loops):
            x_new_loop = loop_ask_tell.ask()
    
            alpha_new_loop, power_new_loop, n_power_09_new_loop, max_ess_new_loop, y_new_loop = obj_f(
                mu = mu, 
                params = x_new_loop,
                n_analyses = num_analyses,
                target_power = target_power,
                target_alpha = target_alpha
            )
    
            # how many designs meet alpha 0.05 and power 0.9
            design_goal_met_loop = (alpha_new_loop <= target_alpha) & (power_new_loop >= target_power - 0.05)
            if design_goal_met_loop:
                n_design_goal_met_loop += 1
            design_goal_met_list_loop.append(design_goal_met_loop)
    
            # how many designs are within epsilon of alpha and power
            alpha_target_low = alpha_new_loop <= (target_alpha + epsilon_loop)
            alpha_target_high = alpha_new_loop >= (target_alpha - epsilon_loop)
            within_epsilon_loop = ( alpha_target_low & alpha_target_high )
            if within_epsilon_loop:
                n_within_epsilon_loop += 1
            within_epsilon_list_loop.append(within_epsilon_loop)
    
            loop_ask_tell.tell(trieste.data.Dataset(
                query_points = x_new_loop,
                observations = np.array([[y_new_loop]])
            ))
    
            if (_i + 1) % when_to_print_loop == 0:
                print(
                    f"\nLoop {_i+1} completed. "
                    f"Feasible: {n_design_goal_met_loop}/{_i+1} "
                    f"({100*n_design_goal_met_loop/(_i+1):.0f}%).",
                    end=""
                 )
            elif (_i > when_to_print_loop) and ((_i + 1) % 5 == 0):
                print(".", end="")
    
        # end the timer
        end_time_loop = time.time()
        execution_time_loop = end_time_loop - start_time_loop

        # save helpful items from loop
        loop_observations.append(
            np.array(loop_ask_tell.to_result().try_get_final_dataset().observations[2:]).flatten()
        )

        loop_query_points.append(
            np.array(loop_ask_tell.to_result().try_get_final_dataset().query_points[2:])
        )

        loop_design_goal_met.append(design_goal_met_list_loop)
        loop_within_epsilon.append(within_epsilon_list_loop)

        print("=============================")
        print(f"= Completed experiment {_j+1}. =")
        print("=============================")

        best_obj_f_loop = np.min(loop_ask_tell.to_result().try_get_final_dataset().observations[2:])

        best_idx_loop = np.argmin(loop_ask_tell.to_result().try_get_final_dataset().observations[2:])
    
        best_bounds_loop = reverse_to_boundaries(
            loop_ask_tell.to_result().try_get_final_dataset().query_points[2:][best_idx_loop], 
            K=num_analyses
        )
    
        best_n09_loop, _ = ss.find_sample_size(
            power_target = target_power,
            n_analyses = num_analyses,
            upper_bounds = best_bounds_loop[0],
            lower_bounds = best_bounds_loop[1],
            null_hypothesis = delta0,
            alt_hypothesis = delta1,
            variance = sigma2
        )

        loop_best_obj_f.append(best_obj_f_loop)
        loop_best_idx.append(best_idx_loop)
        loop_best_bounds.append(best_bounds_loop)
        loop_execution_time.append(execution_time_loop)
        loop_best_n09.append(best_n09_loop)
    
        print(f"Bayesian optim:       {n_loops} evaluations")
        print(f"Loop took:            {execution_time_loop/60:.1f} min")
        print(f"Feasible BO:          {n_design_goal_met_loop}/{n_loops} ({100*np.mean(design_goal_met_loop):.1f}%)")
        print(f"Feasible epsilon:     {n_within_epsilon_loop}/{n_loops} ({100*np.mean(within_epsilon_list_loop):.1f}%)")
        print(f"Best objective val:   {best_obj_f_loop}")
        print(f"Best objective idx:   {best_idx_loop}")
        print(f"Best n w/power 0.9:   {best_n09_loop}")
        print(f"Better than tri?      {best_obj_f_loop < tri_obj}\n")
    return (loop_best_bounds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Best bounds characteristics
    """)
    return


@app.cell
def _(best_n09, delta0, delta1, loop_best_bounds, num_analyses, sigma2, sim):
    for (_i, _bound) in enumerate(loop_best_bounds):
        _chars = sim.group_sequential_designs(
            n_analyses = num_analyses,
            upper_bounds = _bound[0],
            lower_bounds = _bound[1],
            n_patients = best_n09,
            null_hypothesis = delta0,
            alt_hypothesis = delta1,
            variance = sigma2
        )

        print(f"Run: {_i+1}")
        print(f"Upper: {_bound[0]}")
        print(f"Lower: {_bound[1]}")
        print(f"Alpha: {_chars[1]}")
        print(f"Beta:  {_chars[2]}")
        print(f"ESS:   {_chars[3]}\n")
    return


if __name__ == "__main__":
    app.run()
