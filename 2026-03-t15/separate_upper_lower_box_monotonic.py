# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.20.2",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
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
    import matplotlib.pyplot as plt

    # imports for GP regression (Step 3)
    import gpflow

    # imports for Bayes opt (Step 4-6)
    import trieste
    from trieste.space import Box
    from trieste.models.gpflow.models import GaussianProcessRegression
    import tensorflow as tf

    return Box, GaussianProcessRegression, gpflow, np, pd, plt, tf, trieste


@app.cell
def _():
    from py_group_sequential_designs import generate_boundaries as bd
    from py_group_sequential_designs import feasibility_penalty as fp
    from py_group_sequential_designs import boundary_manipulations as fmt_bd
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
def _(fmt_bd, fn_min, fp, gen_input, np, num_analyses, ss):
    # this function contains a penalty for non-monotonicity
    def generate_x_y_new(
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
        if fmt_bd.check_monotonicity(n_analyses=num_analyses, bounds=x) == False:
            y = 25
        else: 
            y = fn_min.function_to_minimize(max_ess_val=max_ess_new/mu, penalty=penalty)

        return (np.array([x]), np.array([[y]]))

    return (generate_x_y_new,)


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
    generate_x_y_new,
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

    x1, y1 = generate_x_y_new(
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
    generate_x_y_new,
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

    x2, y2 = generate_x_y_new(
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
    generate_x_y_new,
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

    x3, y3 = generate_x_y_new(
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


@app.cell
def _(plt):
    # get 10 colors to use
    colors = plt.cm.tab20.colors
    return (colors,)


@app.cell
def _(colors, design_matrix, np, plt):
    _fig, _ax = plt.subplots()

    for _i, _row in enumerate(design_matrix):
        _ax.plot([1,2,3], _row[0:3], color = colors[_i])
        _ax.plot([1,2,3], np.append(_row[3:5], _row[2]), color = colors[_i])

    _fig
    return


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

    kernel.lengthscales = gpflow.Parameter(value = [1, 1, 1, 1, 1, 1], trainable = True)

    likelihood = gpflow.likelihoods.Gaussian()
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
        lower = [ 0, 0, 0, -3, -3, 10], 
        upper = [ 3, 3, 3,  0,  0, 30]
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
    ## Bayes opt loop - with monotonicity penalty
    """)
    return


@app.cell
def _(
    ask_tell,
    assumed_variance,
    fmt_bd,
    generate_x_y_new,
    important_diff_delta,
    mu,
    num_analyses,
    sim,
    target_alpha,
    target_power,
    trieste,
):
    num_repeats = 500
    when_to_print = 50

    for _i in range(num_repeats):
        x_results1 = ask_tell.ask()

        new_inputs1 = fmt_bd.format_boundaries_after_ask(
            n_analyses = num_analyses,
            result = x_results1
        )

        new_sim_trial1 = sim.group_sequential_designs(
            n_analyses = num_analyses,
            upper_bounds = new_inputs1[0],
            lower_bounds = new_inputs1[1],
            n_patients = new_inputs1[2],
            null_hypothesis = 0,
            alt_hypothesis = important_diff_delta,
            variance = assumed_variance
        )

        new_x1, new_y1 = generate_x_y_new(
            mu = mu,
            upper_bounds = new_inputs1[0],
            lower_bounds= new_inputs1[1],
            n_analyses = num_analyses,
            target_power = target_power,
            target_alpha = target_alpha,
            alpha_prime = new_sim_trial1[1],
            beta_prime = 1-new_sim_trial1[2],
            n_power09 = new_inputs1[2]
        )

        new_data1 = trieste.data.Dataset(
            query_points = new_x1, 
            observations = new_y1
        )

        ask_tell.tell(new_data=new_data1)

        if (_i+1) % when_to_print == 0:
            print(f"\nLoop {_i+1} completed.", end="")
        elif (_i > when_to_print) and ((_i+1) % 5 == 0):
            print(".", end="")
    return


@app.cell
def _(ask_tell):
    not_25 = ask_tell.to_result().try_get_final_dataset().observations != 25
    return (not_25,)


@app.cell
def _(ask_tell, plt):
    plt.hist(ask_tell.to_result().try_get_final_dataset().observations, bins=50)
    return


@app.cell
def _(ask_tell, plt):
    plt.figure(figsize=(12, 5))
    plt.plot(ask_tell.to_result().try_get_final_dataset().observations)
    return


@app.cell
def _(ask_tell, not_25, plt):
    plt.figure(figsize=(12, 5))
    plt.plot(ask_tell.to_result().try_get_final_dataset().observations[not_25])
    return


@app.cell
def _(ask_tell, not_25, plt):
    plt.hist(ask_tell.to_result().try_get_final_dataset().observations[not_25], bins=50)
    return


@app.cell
def _(ask_tell, np, tf):
    np.where(tf.squeeze(ask_tell.to_result().try_get_final_dataset().observations > 25))
    return


@app.cell
def _(ask_tell, not_25):
    ask_tell.to_result().try_get_final_dataset().observations[not_25]
    return


@app.cell
def _(ask_tell, tf):
    lt_03 = tf.squeeze(ask_tell.to_result().try_get_final_dataset().observations < 1)
    return (lt_03,)


@app.cell
def _(lt_03, np):
    np.where(lt_03)[0]
    return


@app.cell
def _(ask_tell, fmt_bd):
    monotonic1 = []
    for _bounds in ask_tell.to_result().try_get_final_dataset().query_points:
        monotonic1.append(fmt_bd.check_monotonicity(n_analyses=3, bounds=_bounds))
    return (monotonic1,)


@app.cell
def _(lt_03, monotonic1, np):
    idx1 = np.where(np.array(monotonic1) & lt_03)
    return (idx1,)


@app.cell
def _(idx1):
    idx1
    return


@app.cell
def _(ask_tell, idx1, np):
    best_monotonic_bounds = np.array(ask_tell.to_result().try_get_final_dataset().query_points)[idx1]
    return (best_monotonic_bounds,)


@app.cell
def _(ask_tell, idx1, np):
    np.array(ask_tell.to_result().try_get_final_dataset().observations)[idx1]
    return


@app.cell
def _(best_monotonic_bounds, colors, np, plt):
    _fig, _ax = plt.subplots()

    # plot all the bounds that were monotonic and near the pocock cost
    for _i, _bounds in enumerate(best_monotonic_bounds[3:len(best_monotonic_bounds)]):
        _ax.plot([1,2,3], _bounds[0:3], color = colors[_i])
        _ax.plot([1,2,3], np.append(_bounds[3:5], _bounds[2]), color = colors[_i])

    # plot the pocock bounds for reference
    _ax.plot([1,2,3], best_monotonic_bounds[0][0:3], linewidth=3, color="blue")
    _ax.plot([1,2,3], np.append(best_monotonic_bounds[0][3:5], best_monotonic_bounds[0][2]), linewidth=3, color="blue")

    _fig
    return


@app.cell
def _(ask_tell, np):
    np.array(ask_tell.to_result().try_get_final_dataset().query_points)[922]
    return


@app.cell
def _(assumed_variance, important_diff_delta, sim):
    sim.group_sequential_designs(
        n_analyses=3,
        lower_bounds=[-1.38933787, -0.45527405, 1.52218577],
        upper_bounds=[2.99481833,  2.51223152,  1.52218577],
        n_patients=14.64947357,
        alt_hypothesis=important_diff_delta,
        variance=assumed_variance
    )
    return


@app.cell
def _(best_monotonic_bounds, mo):
    slider = mo.ui.slider(start=3, stop=len(best_monotonic_bounds)-1, step=1)
    return (slider,)


@app.cell
def _(slider):
    slider
    return


@app.cell
def _(best_monotonic_bounds, np, plt, slider):
    _fig, _ax = plt.subplots()

    # plot the pocock bounds for reference
    _ax.plot([1,2,3], best_monotonic_bounds[0][0:3], color="blue")
    _ax.plot([1,2,3], np.append(best_monotonic_bounds[0][3:5], best_monotonic_bounds[0][2]), color="blue")

    # plot the pocock bounds for reference
    _ax.plot([1,2,3], best_monotonic_bounds[1][0:3], color="green")
    _ax.plot([1,2,3], np.append(best_monotonic_bounds[1][3:5], best_monotonic_bounds[1][2]), color="green")

    # plot the pocock bounds for reference
    _ax.plot([1,2,3], best_monotonic_bounds[2][0:3], color="purple")
    _ax.plot([1,2,3], np.append(best_monotonic_bounds[2][3:5], best_monotonic_bounds[2][2]), color="purple")

    _ax.plot([1,2,3], best_monotonic_bounds[slider.value][0:3], color="orange")
    _ax.plot([1,2,3], np.append(best_monotonic_bounds[slider.value][3:5], best_monotonic_bounds[slider.value][2]), color="orange")
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
    # Save the model runs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## With monotonicity constraint
    """)
    return


@app.cell
def _(ask_tell, pd):
    bounds_monotonicity = pd.DataFrame(
        data=ask_tell.to_result().try_get_final_dataset().query_points,
        columns=["upper1","upper2","upper3","lower1","lower2","n"]
    )
    return (bounds_monotonicity,)


@app.cell
def _(ask_tell, pd):
    penalty_monotonicity = pd.DataFrame(
        data=ask_tell.to_result().try_get_final_dataset().observations,
        columns=["penalty"]
    )
    return (penalty_monotonicity,)


@app.cell
def _(bounds_monotonicity, pd, penalty_monotonicity):
    monotonicity = pd.concat([bounds_monotonicity, penalty_monotonicity], axis=1)
    return (monotonicity,)


@app.cell
def _(monotonicity):
    monotonicity.to_csv(path_or_buf="/tf/2026-03-t15/separate_upper_lower_box_monotonic.csv", index=False)
    return


if __name__ == "__main__":
    app.run()
