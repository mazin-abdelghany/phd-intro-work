# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.0",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To do:
    - Trial penalty function without step
    - Run Bayes opt without normalization
    - Try different kernels (Matern with and without step)
    - Check GPR model if all values are trainable (in full Bayes opt loop)
    - Try GPR model with low likelihood variance (low but not as close to zero as 1e-5)
    - Can consider training only first boundary values and force monotonicity with a functional form (similar to O’Brien-Fleming)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Setup
    """)
    return


@app.cell
def _():
    # imports for study design step (Step 1)
    import numpy as np
    import pandas as pd
    from scipy import stats
    from scipy import optimize

    # imports for GP regression (Step 3)
    import gpflow

    # imports for Bayes opt (Step 4-6)
    import trieste
    from trieste.space import Box
    from trieste.models.gpflow.models import GaussianProcessRegression
    import tensorflow as tf
    from trieste.experimental.plotting import plot_regret

    return Box, GaussianProcessRegression, gpflow, np, tf, trieste


@app.cell
def _():
    from py_group_sequential_designs import boundaries as bd
    from py_group_sequential_designs import feasibility_penalty as fp
    from py_group_sequential_designs import format_boundaries_after_ask as fmt_bd
    from py_group_sequential_designs import function_to_minimize as fn_min
    from py_group_sequential_designs import generate_gpr_input as gen_input
    from py_group_sequential_designs import simulate as sim
    from py_group_sequential_designs import sample_size as ss

    return bd, fmt_bd, fn_min, fp, gen_input, sim, ss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bayes opt default values and setup
    """)
    return


@app.cell
def _(ss):
    # some set defaults
    num_analyses = 3
    target_alpha = 0.05
    target_power = 0.9
    important_diff_delta = 1
    assumed_variance = 3

    # to obtain mu (sample size at one stage)
    mu = ss.sample_size_means(
        ratio=1,
        variance=assumed_variance,
        power=target_power,
        alpha=target_alpha,
        delta=important_diff_delta
    )
    return (
        assumed_variance,
        important_diff_delta,
        mu,
        num_analyses,
        target_alpha,
        target_power,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate X Y helper function
    """)
    return


@app.cell
def _(fn_min, fp, gen_input, np, ss):
    # create a function that generates the points x and y that will be
    # included in the design matrix X and Y
    def generate_x_y(
            mu,
            upper_bounds,
            lower_bounds,
            n_analyses,
            target_power,
            target_alpha,
            alpha_prime,
            beta_prime,
            n_power09):

        # 2. Generate the GPR input values
        # note that the input includes the sample size at power 0.9
        x = gen_input.generate_gpr_input(
            n_analyses = n_analyses,
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
            n_patients=n_power09)

        # 3. Generate maximum expected sample size and feasibility penalty
        max_ess_new = ss.max_ess(
            n_analyses=n_analyses,
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
            n_patients=n_power09)

        penalty = fp.new_penalty(
            mu = mu,
            power=target_power,
            alpha=target_alpha,
            beta_prime=beta_prime,
            alpha_prime=alpha_prime
        )


        # 4. Calculate the function value (GPR output)
        y = fn_min.function_to_minimize(max_ess_val=max_ess_new, penalty=penalty)

        return (np.array([x]), np.array([[y]]))

    return (generate_x_y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First 3 points for Bayes opt
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Point 1
    """)
    return


@app.cell
def _(
    assumed_variance,
    bd,
    generate_x_y,
    important_diff_delta,
    mu,
    num_analyses,
    ss,
    target_alpha,
    target_power,
):
    # simulate the trial design 
    poc_simulation = bd.calculate_pocock_boundaries(
        n_analyses=num_analyses,
        alpha=target_alpha,
        n_patients=20
    )

    # find the number of patients that achieves 90% power (beta 0.1)
    # here we get beta_prime
    poc_n_power09, poc_power = ss.find_sample_size(
        n_analyses = num_analyses,
        upper_bounds = poc_simulation[0],
        lower_bounds = poc_simulation[1],
        alt_hypothesis = important_diff_delta,
        variance = assumed_variance
    )

    x1, y1 = generate_x_y(
        mu = mu,
        upper_bounds = poc_simulation[0],
        lower_bounds = poc_simulation[1],
        n_analyses = num_analyses,
        target_power = target_power,
        target_alpha = target_alpha,
        alpha_prime = poc_simulation[3],
        beta_prime = 1-poc_power,
        n_power09 = poc_n_power09
    )
    return poc_n_power09, poc_power, poc_simulation, x1, y1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Point 2
    """)
    return


@app.cell
def _(
    assumed_variance,
    bd,
    generate_x_y,
    important_diff_delta,
    mu,
    num_analyses,
    ss,
    target_alpha,
    target_power,
):
    # simulate the trial design 
    of_simulation = bd.calculate_of_boundaries(
        n_analyses=num_analyses,
        alpha=target_alpha,
        n_patients=20
    )

    # find the number of patients that achieves 90% power (beta 0.1) and beta_prime
    of_n_power09, of_power = ss.find_sample_size(
        n_analyses = num_analyses,
        upper_bounds = of_simulation[0],
        lower_bounds = of_simulation[1],
        alt_hypothesis = important_diff_delta,
        variance = assumed_variance
    )

    x2, y2 = generate_x_y(
        mu = mu,
        upper_bounds = of_simulation[0],
        lower_bounds = of_simulation[1],
        n_analyses = num_analyses,
        target_power = target_power,
        target_alpha = target_alpha,
        alpha_prime = of_simulation[3],
        beta_prime = 1-of_power,
        n_power09 = of_n_power09
    )
    return of_n_power09, of_power, of_simulation, x2, y2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Point 3
    """)
    return


@app.cell
def _(
    assumed_variance,
    bd,
    generate_x_y,
    important_diff_delta,
    mu,
    num_analyses,
    ss,
    target_alpha,
    target_power,
):
    # simulate the trial design 
    tri_simulation = bd.calculate_triangular_boundaries(
        n_analyses=num_analyses,
        alpha=target_alpha,
        delta=important_diff_delta,
        n_patients=20
    )

    # find the number of patients that achieves 90% power (beta 0.1) and beta_prime
    tri_n_power09, tri_power = ss.find_sample_size(
        n_analyses = num_analyses,
        upper_bounds = tri_simulation[0],
        lower_bounds = tri_simulation[1],
        alt_hypothesis = important_diff_delta,
        variance = assumed_variance
    )

    x3, y3 = generate_x_y(
        mu = mu,
        upper_bounds = tri_simulation[0],
        lower_bounds = tri_simulation[1],
        n_analyses = num_analyses,
        target_power = target_power,
        target_alpha = target_alpha,
        alpha_prime = tri_simulation[3],
        beta_prime = 1-tri_power,
        n_power09 = tri_n_power09
    )
    return tri_n_power09, tri_power, tri_simulation, x3, y3


@app.cell
def _(np, x1, x2, x3):
    design_matrix = np.concatenate((x1, x2, x3))
    design_matrix
    return (design_matrix,)


@app.cell
def _(np, y1, y2, y3):
    output_vals = np.concatenate((y1, y2, y3))
    output_vals
    return (output_vals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bayes opt without normalization
    """)
    return


@app.cell
def _(GaussianProcessRegression, gpflow):
    def build_model(X, Y):

        kernel = gpflow.kernels.SquaredExponential()

        likelihood = gpflow.likelihoods.Gaussian()

        gpr = gpflow.models.GPR(
            data = (X, Y),
            kernel = kernel,
            likelihood = likelihood
        )

        gpflow.utilities.print_summary(gpr, fmt="notebook")

        return GaussianProcessRegression(gpr)

    return (build_model,)


@app.cell
def _(build_model, design_matrix, output_vals):
    bayes_opt_model = build_model(
        X = design_matrix, 
        Y = output_vals
    )
    return (bayes_opt_model,)


@app.cell
def _(design_matrix, output_vals, trieste):
    # create a dataset that works well with trieste
    initial_data = trieste.data.Dataset(
        query_points = design_matrix, 
        observations = output_vals
    )
    return (initial_data,)


@app.cell
def _(Box):
    # create the search space using trieste Box function
    search_space = Box(
        lower = [-6, -6, -6, -6, -6, 4], 
        upper = [6, 6, 6, 6, 6, 100]
    )
    return (search_space,)


@app.cell
def _(initial_data):
    initial_data
    return


@app.cell
def _(bayes_opt_model, initial_data, search_space, trieste):
    ask_tell = trieste.ask_tell_optimization.AskTellOptimizer(
        search_space = search_space,
        datasets = initial_data,
        models = bayes_opt_model,
        acquisition_rule = trieste.acquisition.rule.EfficientGlobalOptimization(
            optimizer = 
              trieste.acquisition.optimizer.generate_continuous_optimizer(
                num_optimization_runs = 5000
            )
        )
    )
    return (ask_tell,)


@app.cell
def _(
    ask_tell,
    assumed_variance,
    fmt_bd,
    generate_x_y,
    important_diff_delta,
    mu,
    num_analyses,
    sim,
    target_alpha,
    target_power,
    trieste,
):
    # takes approximately 1 minute to run

    num_repeats = 500

    for _i in range(num_repeats):
        x_results = ask_tell.ask()

        new_inputs = fmt_bd.format_boundaries_after_ask(
            n_analyses = num_analyses,
            result = x_results
        )

        new_sim_trial = sim.group_sequential_designs(
            n_analyses = num_analyses,
            upper_bounds = new_inputs[0],
            lower_bounds = new_inputs[1],
            n_patients = new_inputs[2],
            null_hypothesis = 0,
            alt_hypothesis = important_diff_delta,
            variance = assumed_variance
        )

        new_x, new_y = generate_x_y(
            mu = mu,
            upper_bounds = new_inputs[0],
            lower_bounds= new_inputs[1],
            n_analyses = num_analyses,
            target_power = target_power,
            target_alpha = target_alpha,
            alpha_prime = new_sim_trial[1],
            beta_prime = 1-new_sim_trial[2],
            n_power09 = new_inputs[2]
        )

        new_data = trieste.data.Dataset(
            query_points = new_x, 
            observations = new_y
        )

        ask_tell.tell(new_data=new_data)

        if (_i+1) % 5 == 0:
            print(f"Loop {_i+1} completed.")
    return (num_repeats,)


@app.cell
def _(ask_tell):
    ask_tell.to_result()
    return


@app.cell
def _(ask_tell, tf):
    min_idx = tf.squeeze(tf.argmin(
        ask_tell.to_result().try_get_final_dataset().observations.numpy()
    ))
    return (min_idx,)


@app.cell
def _(min_idx):
    min_idx
    return


@app.cell
def _(ask_tell, min_idx):
    ask_tell.to_result().try_get_final_dataset().observations[min_idx]
    return


@app.cell
def _(ask_tell, min_idx):
    ask_tell.to_result().try_get_final_dataset().query_points[min_idx].numpy()
    return


@app.cell
def _(assumed_variance, important_diff_delta, sim):
    sim.group_sequential_designs(
        upper_bounds = [1.80548856, -3.6671459 ,  2.25116924],
        lower_bounds = [5.97277306, -1.99148641,  2.25116924],
        n_patients = 4,
        alt_hypothesis = important_diff_delta,
        variance = assumed_variance
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GPR model assessment
    """)
    return


@app.cell
def _(ask_tell, gpflow):
    gpflow.utilities.print_summary(ask_tell.model.model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GPR model with low likelihood variance
    """)
    return


@app.cell
def _(GaussianProcessRegression, gpflow):
    def build_model_specific(X, Y, kernel, likelihood):

        gpr = gpflow.models.GPR(
            data = (X, Y),
            kernel = kernel,
            likelihood = likelihood
        )

        gpflow.utilities.print_summary(gpr, fmt="notebook")

        return GaussianProcessRegression(gpr)

    return (build_model_specific,)


@app.cell
def _(gpflow):
    kernel = gpflow.kernels.SquaredExponential()
    kernel.variance = gpflow.Parameter(value = 100, trainable = False)
    kernel.lengthscales = gpflow.Parameter(value = [1, 1, 1, 1, 1, 50], trainable = True)

    likelihood = gpflow.likelihoods.Gaussian()
    likelihood.variance = gpflow.Parameter(value = 1e-1, trainable = False)
    return kernel, likelihood


@app.cell
def _(build_model_specific, design_matrix, kernel, likelihood, output_vals):
    specific_bayes_opt_model = build_model_specific(
        X = design_matrix, 
        Y = output_vals,
        kernel = kernel,
        likelihood = likelihood
    )
    return (specific_bayes_opt_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The search space will remain the same.
    """)
    return


@app.cell
def _(initial_data, search_space, specific_bayes_opt_model, trieste):
    specific_ask_tell = trieste.ask_tell_optimization.AskTellOptimizer(
        search_space = search_space,
        datasets = initial_data,
        models = specific_bayes_opt_model,
        acquisition_rule = trieste.acquisition.rule.EfficientGlobalOptimization(
            optimizer = 
              trieste.acquisition.optimizer.generate_continuous_optimizer(
                num_optimization_runs = 5000
            )
        )
    )
    return (specific_ask_tell,)


@app.cell
def _(
    assumed_variance,
    fmt_bd,
    generate_x_y,
    important_diff_delta,
    mu,
    num_analyses,
    num_repeats,
    sim,
    specific_ask_tell,
    target_alpha,
    target_power,
    trieste,
):
    # takes approximately 3-5 minutes to run 50 times
    # takes approximately 45 minutes to run 500 times
    # num_repeats = 50 defined above

    for _i in range(num_repeats):
        spec_x_results = specific_ask_tell.ask()

        spec_new_inputs = fmt_bd.format_boundaries_after_ask(
            n_analyses = num_analyses,
            result = spec_x_results
        )

        spec_new_sim_trial = sim.group_sequential_designs(
            n_analyses = num_analyses,
            upper_bounds = spec_new_inputs[0],
            lower_bounds = spec_new_inputs[1],
            n_patients = spec_new_inputs[2],
            null_hypothesis = 0,
            alt_hypothesis = important_diff_delta,
            variance = assumed_variance
        )

        spec_new_x, spec_new_y = generate_x_y(
            mu = mu,
            upper_bounds = spec_new_inputs[0],
            lower_bounds= spec_new_inputs[1],
            n_analyses = num_analyses,
            target_power = target_power,
            target_alpha = target_alpha,
            alpha_prime = spec_new_sim_trial[1],
            beta_prime = 1-spec_new_sim_trial[2],
            n_power09 = spec_new_inputs[2]
        )

        spec_new_data = trieste.data.Dataset(
            query_points = spec_new_x, 
            observations = spec_new_y
        )

        specific_ask_tell.tell(new_data=spec_new_data)

        if (_i+1) % 5 == 0:
            print(f"Loop {_i+1} completed.")
    return


@app.cell
def _(specific_ask_tell):
    specific_ask_tell.to_result()
    return


@app.cell
def _(specific_ask_tell, tf):
    new_min_idx = tf.squeeze(tf.argmin(
        specific_ask_tell.to_result().try_get_final_dataset().observations.numpy()
    ))
    return (new_min_idx,)


@app.cell
def _(new_min_idx):
    new_min_idx
    return


@app.cell
def _(new_min_idx, specific_ask_tell):
    specific_ask_tell.to_result().try_get_final_dataset().observations[new_min_idx]
    return


@app.cell
def _(new_min_idx, specific_ask_tell):
    specific_ask_tell.to_result().try_get_final_dataset().query_points[new_min_idx].numpy()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GP model assessment
    """)
    return


@app.cell
def _(gpflow, specific_ask_tell):
    gpflow.utilities.print_summary(specific_ask_tell.model.model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Kernel exploration
    """)
    return


@app.cell
def _(gpflow):
    mat_kernel = gpflow.kernels.Matern52()
    mat_kernel.variance = gpflow.Parameter(value = 100, trainable = True)
    mat_kernel.lengthscales = gpflow.Parameter(value = [1, 1, 1, 1, 1, 50], trainable = True)

    mat_likelihood = gpflow.likelihoods.Gaussian()
    mat_likelihood.variance = gpflow.Parameter(value = 1e-1, trainable = False)
    return mat_kernel, mat_likelihood


@app.cell
def _(
    build_model_specific,
    design_matrix,
    mat_kernel,
    mat_likelihood,
    output_vals,
):
    mat_bayes_opt_model = build_model_specific(
        X = design_matrix, 
        Y = output_vals,
        kernel = mat_kernel,
        likelihood = mat_likelihood
    )
    return (mat_bayes_opt_model,)


@app.cell
def _(initial_data, mat_bayes_opt_model, search_space, trieste):
    mat_ask_tell = trieste.ask_tell_optimization.AskTellOptimizer(
        search_space = search_space,
        datasets = initial_data,
        models = mat_bayes_opt_model,
        acquisition_rule = trieste.acquisition.rule.EfficientGlobalOptimization(
            optimizer = 
              trieste.acquisition.optimizer.generate_continuous_optimizer(
                num_optimization_runs = 5000
            )
        )
    )
    return (mat_ask_tell,)


@app.cell
def _(
    assumed_variance,
    fmt_bd,
    generate_x_y,
    important_diff_delta,
    mat_ask_tell,
    mu,
    num_analyses,
    num_repeats,
    sim,
    target_alpha,
    target_power,
    trieste,
):
    # takes approximately 3-4 minutes to run 50 times
    # takes approximately 105 minutes to run 500 times
    # num_repeats = 50 defined above

    for _i in range(num_repeats):
        mat_x_results = mat_ask_tell.ask()

        mat_new_inputs = fmt_bd.format_boundaries_after_ask(
            n_analyses = num_analyses,
            result = mat_x_results
        )

        mat_new_sim_trial = sim.group_sequential_designs(
            n_analyses = num_analyses,
            upper_bounds = mat_new_inputs[0],
            lower_bounds = mat_new_inputs[1],
            n_patients = mat_new_inputs[2],
            null_hypothesis = 0,
            alt_hypothesis = important_diff_delta,
            variance = assumed_variance
        )

        mat_new_x, mat_new_y = generate_x_y(
            mu = mu,
            upper_bounds = mat_new_inputs[0],
            lower_bounds= mat_new_inputs[1],
            n_analyses = num_analyses,
            target_power = target_power,
            target_alpha = target_alpha,
            alpha_prime = mat_new_sim_trial[1],
            beta_prime = 1-mat_new_sim_trial[2],
            n_power09 = mat_new_inputs[2]
        )

        mat_new_data = trieste.data.Dataset(
            query_points = mat_new_x, 
            observations = mat_new_y
        )

        mat_ask_tell.tell(new_data=mat_new_data)

        if (_i+1) % 5 == 0:
            print(f"Loop {_i+1} completed.")
    return


@app.cell
def _(mat_ask_tell):
    mat_ask_tell.to_result()
    return


@app.cell
def _(mat_ask_tell, tf):
    mat_min_idx = tf.squeeze(tf.argmin(
        mat_ask_tell.to_result().try_get_final_dataset().observations.numpy()
    ))
    return (mat_min_idx,)


@app.cell
def _(mat_min_idx):
    mat_min_idx
    return


@app.cell
def _(mat_ask_tell, mat_min_idx):
    mat_ask_tell.to_result().try_get_final_dataset().observations[mat_min_idx]
    return


@app.cell
def _(mat_ask_tell, mat_min_idx):
    mat_ask_tell.to_result().try_get_final_dataset().query_points[mat_min_idx].numpy()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GP model assessement
    """)
    return


@app.cell
def _(gpflow, mat_ask_tell):
    gpflow.utilities.print_summary(mat_ask_tell.model.model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Penalty function without step
    """)
    return


@app.cell
def _(fn_min, fp, gen_input, np, ss):
    # to add a new penalty function, we need to create a new generate_x_y function
    # create a function that generates the points x and y that will be
    # included in the design matrix X and Y
    def new_generate_x_y(
            mu,
            upper_bounds,
            lower_bounds,
            n_analyses,
            target_power,
            target_alpha,
            alpha_prime,
            beta_prime,
            n_power09):

        # 2. Generate the GPR input values
        # note that the input includes the sample size at power 0.9
        x = gen_input.generate_gpr_input(
            n_analyses = n_analyses,
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
            n_patients=n_power09)

        # 3. Generate maximum expected sample size and feasibility penalty
        max_ess_new = ss.max_ess(
            n_analyses=n_analyses,
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
            n_patients=n_power09)

        penalty = fp.smooth_penalty(
            mu = mu,
            power=target_power,
            alpha=target_alpha,
            beta_prime=beta_prime,
            alpha_prime=alpha_prime
        )


        # 4. Calculate the function value (GPR output)
        y = fn_min.function_to_minimize(max_ess_val=max_ess_new, penalty=penalty)

        return (np.array([x]), np.array([[y]]))

    return (new_generate_x_y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First 3 points for Bayes opt
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Point 1
    """)
    return


@app.cell
def _(
    mu,
    new_generate_x_y,
    num_analyses,
    poc_n_power09,
    poc_power,
    poc_simulation,
    target_alpha,
    target_power,
):
    n_x1, n_y1 = new_generate_x_y(
        mu = mu,
        upper_bounds = poc_simulation[0],
        lower_bounds = poc_simulation[1],
        n_analyses = num_analyses,
        target_power = target_power,
        target_alpha = target_alpha,
        alpha_prime = poc_simulation[3],
        beta_prime = 1-poc_power,
        n_power09 = poc_n_power09
    )
    return n_x1, n_y1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Point 2
    """)
    return


@app.cell
def _(
    mu,
    new_generate_x_y,
    num_analyses,
    of_n_power09,
    of_power,
    of_simulation,
    target_alpha,
    target_power,
):
    n_x2, n_y2 = new_generate_x_y(
        mu = mu,
        upper_bounds = of_simulation[0],
        lower_bounds = of_simulation[1],
        n_analyses = num_analyses,
        target_power = target_power,
        target_alpha = target_alpha,
        alpha_prime = of_simulation[3],
        beta_prime = 1-of_power,
        n_power09 = of_n_power09
    )
    return n_x2, n_y2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Point 3
    """)
    return


@app.cell
def _(
    mu,
    new_generate_x_y,
    num_analyses,
    target_alpha,
    target_power,
    tri_n_power09,
    tri_power,
    tri_simulation,
):
    n_x3, n_y3 = new_generate_x_y(
        mu = mu,
        upper_bounds = tri_simulation[0],
        lower_bounds = tri_simulation[1],
        n_analyses = num_analyses,
        target_power = target_power,
        target_alpha = target_alpha,
        alpha_prime = tri_simulation[3],
        beta_prime = 1-tri_power,
        n_power09 = tri_n_power09
    )
    return n_x3, n_y3


@app.cell
def _(n_x1, n_x2, n_x3, np):
    n_design_matrix = np.concatenate((n_x1, n_x2, n_x3))
    n_design_matrix
    return (n_design_matrix,)


@app.cell
def _(n_y1, n_y2, n_y3, np):
    n_output_vals = np.concatenate((n_y1, n_y2, n_y3))
    n_output_vals
    return (n_output_vals,)


@app.cell
def _(n_design_matrix, n_output_vals, trieste):
    smooth_initial_data = trieste.data.Dataset(
        query_points = n_design_matrix, 
        observations = n_output_vals
    )
    return (smooth_initial_data,)


@app.cell
def _(gpflow):
    sm_mat_kernel = gpflow.kernels.Matern52()
    sm_mat_kernel.variance = gpflow.Parameter(value = 100, trainable = True)
    sm_mat_kernel.lengthscales = gpflow.Parameter(value = [1, 1, 1, 1, 1, 10], trainable = True)

    sm_mat_likelihood = gpflow.likelihoods.Gaussian()
    sm_mat_likelihood.variance = gpflow.Parameter(value = 1, trainable = False)
    return sm_mat_kernel, sm_mat_likelihood


@app.cell
def _(
    build_model_specific,
    n_design_matrix,
    n_output_vals,
    sm_mat_kernel,
    sm_mat_likelihood,
):
    sm_mat_bayes_opt_model = build_model_specific(
        X = n_design_matrix, 
        Y = n_output_vals,
        kernel = sm_mat_kernel,
        likelihood = sm_mat_likelihood
    )
    return (sm_mat_bayes_opt_model,)


@app.cell
def _(search_space, sm_mat_bayes_opt_model, smooth_initial_data, trieste):
    mat_smooth_ask_tell = trieste.ask_tell_optimization.AskTellOptimizer(
        search_space = search_space,
        datasets = smooth_initial_data,
        models = sm_mat_bayes_opt_model,
        acquisition_rule = trieste.acquisition.rule.EfficientGlobalOptimization(
            optimizer = 
              trieste.acquisition.optimizer.generate_continuous_optimizer(
                num_optimization_runs = 5000
            )
        )
    )
    return (mat_smooth_ask_tell,)


@app.cell
def _(
    assumed_variance,
    fmt_bd,
    generate_x_y,
    important_diff_delta,
    mat_smooth_ask_tell,
    mu,
    num_analyses,
    num_repeats,
    sim,
    target_alpha,
    target_power,
    trieste,
):
    # takes approximately 3-4 minutes to run 50 times
    # takes approximately 110 minutes to run 500 times
    # num_repeats = 50 defined above

    for _i in range(num_repeats):
        sm_mat_x_results = mat_smooth_ask_tell.ask()

        sm_mat_new_inputs = fmt_bd.format_boundaries_after_ask(
            n_analyses = num_analyses,
            result = sm_mat_x_results
        )

        sm_mat_new_sim_trial = sim.group_sequential_designs(
            n_analyses = num_analyses,
            upper_bounds = sm_mat_new_inputs[0],
            lower_bounds = sm_mat_new_inputs[1],
            n_patients = sm_mat_new_inputs[2],
            null_hypothesis = 0,
            alt_hypothesis = important_diff_delta,
            variance = assumed_variance
        )

        sm_mat_new_x, sm_mat_new_y = generate_x_y(
            mu = mu,
            upper_bounds = sm_mat_new_inputs[0],
            lower_bounds= sm_mat_new_inputs[1],
            n_analyses = num_analyses,
            target_power = target_power,
            target_alpha = target_alpha,
            alpha_prime = sm_mat_new_sim_trial[1],
            beta_prime = 1-sm_mat_new_sim_trial[2],
            n_power09 = sm_mat_new_inputs[2]
        )

        sm_mat_new_data = trieste.data.Dataset(
            query_points = sm_mat_new_x, 
            observations = sm_mat_new_y
        )

        mat_smooth_ask_tell.tell(new_data=sm_mat_new_data)

        if (_i+1) % 5 == 0:
            print(f"Loop {_i+1} completed.")
    return


@app.cell
def _(mat_smooth_ask_tell):
    mat_smooth_ask_tell.to_result()
    return


@app.cell
def _(mat_smooth_ask_tell, tf):
    sm_mat_min_idx = tf.squeeze(tf.argmin(
        mat_smooth_ask_tell.to_result().try_get_final_dataset().observations.numpy()
    ))
    return (sm_mat_min_idx,)


@app.cell
def _(sm_mat_min_idx):
    sm_mat_min_idx
    return


@app.cell
def _(mat_smooth_ask_tell, sm_mat_min_idx):
    mat_smooth_ask_tell.to_result().try_get_final_dataset().observations[sm_mat_min_idx]
    return


@app.cell
def _(mat_smooth_ask_tell, sm_mat_min_idx):
    mat_smooth_ask_tell.to_result().try_get_final_dataset().query_points[sm_mat_min_idx].numpy()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GP model assessement
    """)
    return


@app.cell
def _(gpflow, mat_smooth_ask_tell):
    gpflow.utilities.print_summary(mat_smooth_ask_tell.model.model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Single boundaries with enforced monotonicity
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
