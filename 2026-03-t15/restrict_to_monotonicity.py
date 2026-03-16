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
    from trieste.experimental.plotting import plot_regret

    return Box, GaussianProcessRegression, gpflow, np, plt, stats, tf, trieste


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


@app.cell
def _(design_matrix, np, plt):
    _fig, _ax = plt.subplots()
    colors=["red","orange","purple", "green"]

    for _i, _row in enumerate(design_matrix):
        _ax.plot([1,2,3], _row[0:3], color = colors[_i])
        _ax.plot([1,2,3], np.append(_row[3:5], _row[2]), color = colors[_i])

    _fig
    return (colors,)


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```
    [
     [ 1.99218512e+00, 1.99218512e+00, 1.99218512e+00, -1.99218512e+00, -1.99218512e+00, 1.99629112e+01],
     [ 2.96112971e+00, 2.09383490e+00, 1.70960903e+00, -2.96112971e+00, -2.09383490e+00, 1.75540990e+01],
     [ 2.11957748e+00, 1.87345951e+00, 1.83560794e+00, 6.28553399e-16, 1.12407571e+00, 2.13211873e+01]
    ]
    ```
    """)
    return


@app.cell
def _(np, stats):
    _bounds = [2.11957748e+00,  1.87345951e+00,  1.83560794e+00, 6.28553399e-16,  1.12407571e+00]
    lower = np.empty(6)
    upper = np.empty(6)

    for _i, _bound in enumerate(_bounds):
        lower[_i] = stats.norm.ppf(q=[0.3,0.7], loc=_bound, scale=0.5)[0]
        upper[_i] = stats.norm.ppf(q=[0.3,0.7], loc=_bound, scale=0.5)[1]

    # add a single nan so it works with the plotting loops below
    lower[5] = np.nan
    upper[5] = np.nan

    print(lower)
    print(upper)
    return lower, upper


@app.cell
def _(lower, np, plt, stats, upper):
    _fig, _axes = plt.subplots(nrows=3, ncols=3, figsize=(12,8))
    xx = np.linspace(start=-4, stop=4, num=200)

    pocock_bounds = [ 1.99218512e+00, 1.99218512e+00, 1.99218512e+00, -1.99218512e+00, -1.99218512e+00, 1.99218512e+00]
    of_bounds = [ 2.96112971e+00, 2.09383490e+00, 1.70960903e+00, -2.96112971e+00, -2.09383490e+00, 1.70960903e+00]
    tri_bounds = [ 2.11957748e+00, 1.87345951e+00, 1.83560794e+00, 6.28553399e-16, 1.12407571e+00, 1.83560794e+00]
    row_labels = ["Analysis 1", "Analysis 2", "Analysis 3"]
    col_labels = ["Pocock", "O'Brien-Fleming", "Triangular"]

    for _i in range(3):
        for _j in range(3):
            _axes[_i][_j].plot(xx, stats.norm.pdf(x=xx))

    # plot the pocock bounds
    for _i in range(3):
        _axes[_i,0].axvline(x=pocock_bounds[_i], color = "green")
        _axes[_i,0].axvline(x=pocock_bounds[_i+3], color = "green")

    # plot the obrien-fleming bounds
    for _i in range(3):
        _axes[_i,1].axvline(x=of_bounds[_i], color = "green")
        _axes[_i,1].axvline(x=of_bounds[_i+3], color = "green")

    # plot the triangular bounds
    for _i in range(3):
        _axes[_i,2].axvline(x=tri_bounds[_i], color = "green")
        _axes[_i,2].axvline(x=tri_bounds[_i+3], color = "green")

    # plot possible boundaries
    for _j in range(3):
        for _k in range(3):
            _axes[_k,_j].axvline(x=lower[_k], color = "orange")
            _axes[_k,_j].axvline(x=upper[_k], color = "orange")
            _axes[_k,_j].axvline(x=lower[_k+3], color = "purple")
            _axes[_k,_j].axvline(x=upper[_k+3], color = "purple")

    # plot the titles
    for _ax, _col in zip(_axes[0], col_labels):
        _ax.set_title(_col)
    for _ax, _row in zip(_axes[:,0], row_labels):
        _ax.set_ylabel(_row)

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(Box):
    # the search space is defined assuming a normal distribution with mean at triangular bound value
    # and standard deviation of 0.5 (essentially a prior) and the bounds contain approximate
    # probability density of 0.4
    search_space = Box(
        lower = [1.85737722,  1.61125925,  1.57340768, -0.26220026,  0.86187545, 10], 
        upper = [2.38177774, 2.13565977, 2.0978082,  0.26220026, 1.38627597, 30]
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
    num_repeats = 100
    when_to_print = 10

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
def _(ask_tell, tf):
    lt_3 = tf.squeeze(ask_tell.to_result().try_get_final_dataset().observations < 0.3)
    return (lt_3,)


@app.cell
def _(lt_3, np):
    np.where(lt_3)[0]
    return


@app.cell
def _(fmt_bd, np):
    fmt_bd.format_boundaries_after_ask(n_analyses=3, result=np.array([[ 2.11957748e+00,  1.87345951e+00,  1.83560794e+00,
             6.28553399e-16,  1.12407571e+00,  2.13211873e+01]]))
    return


@app.cell
def _(fmt_bd, np):
    # create a function that checks the monotonicity of bounds
    def check_monotonicity(n_analyses, bounds):

        # first format the boundaries from the ask_tell interface
        # this takes bounds such as [upper1, upper2, upper3, lower1, lower2, n]
        # and outputs a list of lists [[upper1, upper2, upper3], [lower1, lower2, lower3], n]
        fmt_bounds = fmt_bd.format_boundaries_after_ask(
            n_analyses=n_analyses, 
            result=np.array([bounds])
        )

        # take the first two indices from the list, these are the upper and lower bounds
        upper = fmt_bounds[0]
        lower = fmt_bounds[1]

        # loop through the bounds
        for _i in range(len(upper)-1):

            # if we are not at the last stage
            if (_i != len(upper)-1):
                # a design is invalid if the upper bounds are not monotonicly decreasing
                if upper[_i] < upper[_i+1]: return False
                # a design is invalid if the lower bounds are not monotonicly increasing
                if lower[_i] > lower[_i+1]: return False
                # a design is invalid if the upper bound is not greater than the lower bound
                if upper[_i] <= lower[_i]: return False

            # at the last stage, design is invalid if the upper bound is to the lower bound
            # as this is a one-sided statistical test
            else:
                if upper[_i] != lower[_i]: return False


        return True

    return (check_monotonicity,)


@app.cell
def _(ask_tell, check_monotonicity):
    monotonic = []
    for _bounds in ask_tell.to_result().try_get_final_dataset().query_points:
        monotonic.append(check_monotonicity(n_analyses=3, bounds=_bounds))
    return (monotonic,)


@app.cell
def _(lt_3, monotonic, np):
    idx = np.where(np.array(monotonic) & lt_3)
    return (idx,)


@app.cell
def _(ask_tell, idx, np):
    best_monotonic_bounds = np.array(ask_tell.to_result().try_get_final_dataset().query_points)[idx]
    return (best_monotonic_bounds,)


@app.cell
def _(ask_tell, idx, np):
    np.array(ask_tell.to_result().try_get_final_dataset().observations)[idx]
    return


@app.cell
def _(best_monotonic_bounds, colors, np, plt):
    _fig, _ax = plt.subplots()

    for _i, _bounds in enumerate(best_monotonic_bounds[1:4]):
        _ax.plot([1,2,3], _bounds[0:3], color = colors[_i])
        _ax.plot([1,2,3], np.append(_bounds[3:5], _bounds[2]), color = colors[_i])

    _ax.plot([1,2,3], best_monotonic_bounds[0][0:3], linewidth=3, color="blue")
    _ax.plot([1,2,3], np.append(best_monotonic_bounds[0][3:5], best_monotonic_bounds[0][2]), linewidth=3, color="blue")

    _fig
    return


@app.cell
def _():
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


if __name__ == "__main__":
    app.run()
