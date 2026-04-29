import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import gc
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import matplotlib.pyplot as plt

    return gc, np, tf


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
    return (c0,)


@app.cell
def _(Box, c0, np):
    lower = np.array([c0 - 3.0, 0.0, 0.0, 0.0, 0.0])
    upper = np.array([c0 + 3.0, 4.0, 4.0, 4.0, 4.0])

    search_space = Box(lower=lower, upper=upper)
    print(f"  lower: {np.round(search_space.lower.numpy(), 3)}")
    print(f"  upper: {np.round(search_space.upper.numpy(), 3)}")
    return (search_space,)


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
    # First experiment
    """)
    return


@app.cell
def _(
    GaussianProcessRegression,
    design_matrix,
    gc,
    gpflow,
    initial_data,
    mu,
    np,
    num_analyses,
    obj_f,
    output_vals,
    search_space,
    target_alpha,
    target_power,
    tf,
    trieste,
):
    num_repeats   = 20
    when_to_print = 10
    n_design_goal_met = 0
    design_goal_met_list = []

    epsilon = 0.015
    n_within_epsilon = 0
    within_epsilon_list = []

    # outside for loop is number of experiments
    for _ in range(2):

        np.random.seed(1793)
        tf.random.set_seed(1793)

        kernel = gpflow.kernels.Matern52(
            lengthscales=[1.0] * design_matrix.shape[1]
        )
    
        gpr = gpflow.models.GPR(
            data      = (design_matrix, output_vals),
            kernel    = kernel,
            likelihood = gpflow.likelihoods.Gaussian()
        )
    
        bayes_opt_model = GaussianProcessRegression(gpr)
    
        # reset the ask_tell interface
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

        # inside for loop is number of Bayes opt runs
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
                    f"Feasible: {n_within_epsilon}/{_i+1} "
                    f"({100*n_within_epsilon/(_i+1):.0f}%).",
                    end=""
                 )
            elif (_i > when_to_print) and ((_i + 1) % 5 == 0):
                print(".", end="")

        print(f"\nDone. Feasible BO proposals: {n_within_epsilon}/{num_repeats}")
        print(ask_tell.to_result().try_get_final_dataset().observations)

        del ask_tell
        del bayes_opt_model
        del gpr
        del kernel
        gc.collect()
    return


if __name__ == "__main__":
    app.run()
