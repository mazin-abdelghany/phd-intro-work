import marimo

__generated_with = "0.19.11"
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

    return Box, GaussianProcessRegression, gpflow, np, pd, tf, trieste


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
    # Bayes opt defaults - three stage design
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
    ## Generate X, Y helper function
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

        penalty = fp.smooth_penalty(
            mu = mu,
            power=target_power,
            alpha=target_alpha,
            beta_prime=beta_prime,
            alpha_prime=alpha_prime
        )

        # 4. Calculate the function value (GPR output)
        y = fn_min.function_to_minimize(max_ess_val=max_ess_new/mu, penalty=penalty)

        return (np.array([x]), np.array([[y]]))

    return (generate_x_y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The above penalty function is modified by dividing the maximum expected sample size by $\mu$, which is the sample size for a single stage design. This change was made because the Bayesian optimization algorithm was favoring minimization of the sample size parameter  most (always outputting values at the lower bound of the search space).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First 3 points for Bayes opt initialization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pocock - point 1
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
    return x1, y1


@app.cell
def _(x1):
    x1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### O'Brien-Fleming - point 2
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
    return x2, y2


@app.cell
def _(x2):
    x2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Triangular - point 3
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
    return x3, y3


@app.cell
def _(x3):
    x3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generate design matrix
    """)
    return


@app.cell
def _(np, x1, x2, x3):
    design_matrix = np.concatenate((x1, x2, x3))
    design_matrix
    return (design_matrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generate penalty output
    """)
    return


@app.cell
def _(np, y1, y2, y3):
    output_vals = np.concatenate((y1, y2, y3))
    output_vals
    return (output_vals,)


@app.cell
def _(design_matrix, output_vals, trieste):
    initial_data = trieste.data.Dataset(
        query_points = design_matrix, 
        observations = output_vals
    )
    return (initial_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GP regression model
    """)
    return


@app.cell
def _(GaussianProcessRegression, gpflow):
    def build_model(X, Y, kernel, likelihood):

        gpr = gpflow.models.GPR(
            data = (X, Y),
            kernel = kernel,
            likelihood = likelihood
        )

        gpflow.utilities.print_summary(gpr, fmt="notebook")

        return GaussianProcessRegression(gpr)

    return (build_model,)


@app.cell
def _(gpflow):
    kernel = gpflow.kernels.Matern52()
    #kernel.variance = gpflow.Parameter(value = 100, trainable = True)
    kernel.lengthscales = gpflow.Parameter(value = [1, 1, 1, 1, 1, 1], trainable = True)

    likelihood = gpflow.likelihoods.Gaussian()
    #likelihood.variance = gpflow.Parameter(value = 10, trainable = False)
    return kernel, likelihood


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Starting the above kernel variance and likelihood variance at their default values to avoid Choleschy decomposition issues during `ask_tell` initialization.
    """)
    return


@app.cell
def _(build_model, design_matrix, kernel, likelihood, output_vals):
    bayes_opt_model = build_model(
        X = design_matrix, 
        Y = output_vals,
        kernel = kernel,
        likelihood = likelihood
    )
    return (bayes_opt_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the search space
    """)
    return


@app.cell
def _(Box):
    search_space = Box(
        lower = [-6, -6, -6, -6, -6, 2], 
        upper = [6, 6, 6, 6, 6, 60]
    )
    return (search_space,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initialize `ask_tell`
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayes opt loop
    """)
    return


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
    num_repeats = 1000
    when_to_print = 50

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

        if (_i+1) % when_to_print == 0:
            print(f"Loop {_i+1} completed.")
    return


@app.cell
def _(ask_tell):
    ask_tell.to_result().try_get_final_dataset().observations.shape
    return


@app.cell
def _(ask_tell, pd):
    pd.DataFrame(ask_tell.to_result().try_get_final_dataset().observations.numpy())
    return


@app.cell
def _(ask_tell, pd):
    penalty = pd.DataFrame(ask_tell.to_result().try_get_final_dataset().observations.numpy(),
                           columns=["penalty"])
    return (penalty,)


@app.cell
def _(ask_tell, pd):
    pd.DataFrame(ask_tell.to_result().try_get_final_dataset().observations.numpy()).to_csv(
        path_or_buf="penalty.csv",
        sep=",")
    return


@app.cell
def _(ask_tell, pd):
    pd.DataFrame(ask_tell.to_result().try_get_final_dataset().query_points.numpy())
    return


@app.cell
def _(ask_tell, pd):
    pd.DataFrame(ask_tell.to_result().try_get_final_dataset().query_points.numpy()).to_csv(
        path_or_buf="boundaries.csv",
        sep=","
    )
    return


@app.cell
def _(ask_tell, pd):
    boundaries = pd.DataFrame(ask_tell.to_result().try_get_final_dataset().query_points.numpy(),
                             columns=["upper1", "upper2", "upper3", "lower1", "lower2", "n_per_group"])
    return (boundaries,)


@app.cell
def _(boundaries):
    boundaries.mean()
    return


@app.cell
def _(boundaries):
    boundaries.median()
    return


@app.cell
def _(boundaries):
    boundaries.round(1).mode()
    return


@app.cell
def _(boundaries, pd, penalty):
    complete_data = pd.concat([penalty, boundaries], axis=1)
    return (complete_data,)


@app.cell
def _(complete_data):
    complete_data[complete_data["penalty"] < 1].round(3)
    return


@app.cell
def _(ask_tell, tf):
    min_idx_with_start_vals = tf.squeeze(tf.argmin(
        ask_tell.to_result().try_get_final_dataset().observations.numpy()
    ))
    return (min_idx_with_start_vals,)


@app.cell
def _(min_idx_with_start_vals):
    min_idx_with_start_vals
    return


@app.cell
def _(ask_tell, tf):
    min_idx = tf.squeeze(tf.argmin(
        ask_tell.to_result().try_get_final_dataset().observations.numpy()[3:]
    ))
    return (min_idx,)


@app.cell
def _(min_idx):
    min_idx
    return


@app.cell
def _(ask_tell, min_idx):
    ask_tell.to_result().try_get_final_dataset().observations[3:][min_idx]
    return


@app.cell
def _(ask_tell, min_idx):
    ask_tell.to_result().try_get_final_dataset().query_points[3:][min_idx].numpy()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GP model assessment
    """)
    return


@app.cell
def _(ask_tell, gpflow):
    gpflow.utilities.print_summary(ask_tell.model.model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A few example runs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 1000 iterations, box -6 to 6 and 2 to 60

    array([ 3.22596414,  1.49986183,  3.60818439, -5.65356059, -3.28889629,
           23.05013904])
    """)
    return


@app.cell
def _(assumed_variance, important_diff_delta, num_analyses, sim):
    sim.group_sequential_designs(
        n_analyses = num_analyses,
        upper_bounds = [3.60818439, 3.22596414, 1.49986183],
        lower_bounds = [-5.65356059, -3.28889629, 1.49986183],
        n_patients = 23.05013904,
        alt_hypothesis = important_diff_delta,
        variance = assumed_variance
    )
    return


@app.cell
def _(ask_tell):
    ask_tell.to_result().try_get_final_dataset().query_points[329]
    return


@app.cell
def _(assumed_variance, important_diff_delta, num_analyses, sim):
    sim.group_sequential_designs(
        n_analyses = num_analyses,
        upper_bounds = [2.51500741, 1.38512042, 0.26230428],
        lower_bounds = [-5.94931019, -5.35651067,  0.26230428],
        n_patients = 48.05902488,
        alt_hypothesis = important_diff_delta,
        variance = assumed_variance
    )
    return


@app.cell
def _(ask_tell):
    ask_tell.to_result().try_get_final_dataset().query_points[341]
    return


@app.cell
def _(assumed_variance, important_diff_delta, num_analyses, sim):
    sim.group_sequential_designs(
        n_analyses = num_analyses,
        upper_bounds = [4.24428477, 1.99677109, 0.90883888],
        lower_bounds = [-5.12147386, -4.69678167, 0.90883888],
        n_patients = 56.99484308,
        alt_hypothesis = important_diff_delta,
        variance = assumed_variance
    )
    return


@app.cell
def _(ask_tell):
    ask_tell.to_result().try_get_final_dataset().query_points[764]
    return


@app.cell
def _(assumed_variance, important_diff_delta, num_analyses, sim):
    sim.group_sequential_designs(
        n_analyses = num_analyses,
        upper_bounds = [2.21887673,  2.00449016,  0.08102463],
        lower_bounds = [-4.74744761, -2.41228833, 0.08102463],
        n_patients = 55.113109,
        alt_hypothesis = important_diff_delta,
        variance = assumed_variance
    )
    return


if __name__ == "__main__":
    app.run()
