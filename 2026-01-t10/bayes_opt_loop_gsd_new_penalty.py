import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Setup
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order to install the Python package with the necessary group sequential design functions,

    1. Start the Docker container as usual.
    2. Enter the Docker container terminal by typing
        - `docker exec -it bayes-opt-marimo-nb-1 sh`
        - where `bayes-opt-marimo-nb-1` is the name of the marimo container
    3. Within the terminal, enter the command
        - `pip install /tf/pyGroupSequentialDesigns/`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Published package imports
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Personal package imports
    """)
    return


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
    # Bayesian optimization workflow default values
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
    # Simulate initial points for GPR
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create a helper function
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
            calculated_power,
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
            alpha=target_alpha,
            beta=(1-target_power),
            beta_prime=(1-calculated_power),
            alpha_prime=alpha_prime
        )


        # 4. Calculate the function value (GPR output)
        y = fn_min.function_to_minimize(max_ess_val=max_ess_new, penalty=penalty)

        return (np.array([x]), np.array([[y]]))
    return (generate_x_y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Point 1
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
        calculated_power = poc_power,
        n_power09 = poc_n_power09
    )
    return x1, y1


@app.cell
def _(x1):
    x1
    return


@app.cell
def _(y1):
    y1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Point 2
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
        calculated_power = of_power,
        n_power09 = of_n_power09
    )
    return x2, y2


@app.cell
def _(x2):
    x2
    return


@app.cell
def _(y2):
    y2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Point 3
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
    tri_n_power09, tri_beta_prime = ss.find_sample_size(
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
        calculated_power = tri_beta_prime,
        n_power09 = tri_n_power09
    )
    return x3, y3


@app.cell
def _(x3):
    x3
    return


@app.cell
def _(y3):
    y3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bayesian optimization loop
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Enter the initial points
    """)
    return


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
    ## Center and scale
    """)
    return


@app.cell
def _(np):
    def normalize_forward(data, mean=None, std=None):
        if (mean is None) and (std is None):
            mean = np.mean(data)
            std = np.std(data)

        return (
            (data - mean) / std,
            mean,
            std
        )
    return (normalize_forward,)


@app.function
def normalize_backward(norm_data, mean, std):
    return (norm_data * std) + mean


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Test the functions
    """)
    return


@app.cell
def _(design_matrix, normalize_forward):
    test_data, mean, std = normalize_forward(design_matrix)
    test_data
    return mean, std, test_data


@app.cell
def _(mean, std, test_data):
    normalize_backward(test_data, mean, std)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Normalize and loop
    """)
    return


@app.cell
def _(design_matrix, normalize_forward):
    (normed_design_matix, 
     design_matrix_mean, 
     design_matrix_std) = normalize_forward(design_matrix)
    return design_matrix_mean, design_matrix_std, normed_design_matix


@app.cell
def _(normalize_forward, output_vals):
    (normed_output_vals, 
     output_vals_mean, 
     output_vals_std) = normalize_forward(output_vals)
    return normed_output_vals, output_vals_mean, output_vals_std


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
def _(build_model, normed_design_matix, normed_output_vals):
    bayes_opt_model = build_model(
        X = normed_design_matix, 
        Y = normed_output_vals
    )
    return (bayes_opt_model,)


@app.cell
def _(normed_design_matix, normed_output_vals, trieste):
    # create a dataset that works well with trieste
    initial_data = trieste.data.Dataset(
        query_points = normed_design_matix, 
        observations = normed_output_vals
    )
    return (initial_data,)


@app.cell
def _():
    # normalize the search space
    x_search_space = [-20, -20, -20, 20, 20]
    y_search_space = 1000
    return x_search_space, y_search_space


@app.cell
def _(
    design_matrix_mean,
    design_matrix_std,
    normalize_forward,
    x_search_space,
):
    normalize_forward(x_search_space,
                      design_matrix_mean,
                      design_matrix_std)
    return


@app.cell
def _(normalize_forward, output_vals_mean, output_vals_std, y_search_space):
    normalize_forward(y_search_space,
                      output_vals_mean,
                      output_vals_std)
    return


@app.cell
def _(Box):
    # create the search space using trieste Box function
    search_space = Box(
        lower = [-5, -5, -5, -5, -5, 2], 
        upper = [5, 5, 5, 5, 5, 1000]
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
    design_matrix_mean,
    design_matrix_std,
    fmt_bd,
    generate_x_y,
    important_diff_delta,
    mu,
    normalize_forward,
    num_analyses,
    output_vals_mean,
    output_vals_std,
    sim,
    target_alpha,
    target_power,
    trieste,
):
    num_repeats = 1000

    for i in range(num_repeats):
        normed_results = ask_tell.ask()

        unnormed_results_x = normalize_backward(
            normed_results,
            design_matrix_mean,
            design_matrix_std
        )    

        unnormed_new_inputs = fmt_bd.format_boundaries_after_ask(
            n_analyses = num_analyses,
            result = unnormed_results_x
        )

        new_sim_trial = sim.group_sequential_designs(
            n_analyses = num_analyses,
            upper_bounds = unnormed_new_inputs[0],
            lower_bounds = unnormed_new_inputs[1],
            n_patients = unnormed_new_inputs[2],
            null_hypothesis = 0,
            alt_hypothesis = important_diff_delta,
            variance = assumed_variance
        )

        new_x, new_y = generate_x_y(
            mu = mu,
            upper_bounds = unnormed_new_inputs[0],
            lower_bounds= unnormed_new_inputs[1],
            n_analyses = num_analyses,
            target_power = target_power,
            target_alpha = target_alpha,
            alpha_prime = new_sim_trial[1],
            calculated_power = new_sim_trial[2],
            n_power09 = unnormed_new_inputs[2]
        )

        normed_new_x, _, _ = normalize_forward(
            new_x,
            design_matrix_mean,
            design_matrix_std
        )

        normed_new_y, _, _ = normalize_forward(
            new_y,
            output_vals_mean,
            output_vals_std
        )

        new_data = trieste.data.Dataset(
            query_points = normed_new_x, 
            observations = normed_new_y
        )

        ask_tell.tell(new_data=new_data)
    return


@app.cell
def _(ask_tell):
    ask_tell.to_result().try_get_final_dataset().observations.numpy().shape
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
def _(initial_data):
    initial_data
    return


@app.cell
def _(ask_tell, design_matrix_mean, design_matrix_std, min_idx):
    normalize_backward(
        ask_tell.to_result(
        ).try_get_final_dataset().query_points[min_idx].numpy()[0:5],
        design_matrix_mean,
        design_matrix_std
    )
    return


@app.cell
def _(ask_tell, min_idx, output_vals_mean, output_vals_std):
    normalize_backward(
        ask_tell.to_result().try_get_final_dataset().query_points[min_idx].numpy()[5],
        output_vals_mean,
        output_vals_std
    )
    return


@app.cell
def _(assumed_variance, important_diff_delta, sim):
    sim.group_sequential_designs(
        upper_bounds = [6.23615582, -3.99476778, 40.10007342],
        lower_bounds = [37.55283927, 34.79535977,  25.70870463],
        n_patients = 49698.61542851937,
        alt_hypothesis = important_diff_delta,
        variance = assumed_variance
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There remain two issues:
    1. There is no restriction on the lower bounds being less than the upper bounds.
    2. There is no restriction that each subsequent bound is less than or equal to the last.

    To try:
    - Keep everything on the unnormalized scale
    - Create a new penalty function that is not a step, but rather a line
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
