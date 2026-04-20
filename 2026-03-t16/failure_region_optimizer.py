import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bayes opt loop design with failure regions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Our optimization problem has an issue: designs that are not monotonic are not statistically possible. However, while exploring the search space, there is no restriction on monotonicity. In initial testing, penalizing monotonicity does improve model fit, but >90% of designs are still discarded because of non-monotonicity.

    In order to solve this issue, rather than penalizing monotonicity, we will define this as a failure region. The penalty function will return finite values for feasible designs and will return `np.nan` for designs that are statistically impossible.

    The Bayesian optimization loop will then fit two models: (1) a Gaussian process regression (GPR) model for the feasible designs and their objective function and (2) a variational Gaussian process (VPR) classification model with Bernoulli likelihood to model the failure region.

    The steps are as follows:
    1. Generate the 3 initial points, Pocock, O'Brien-Fleming, and triagular bounds and penalty values.
    2. The objective function $f(\cdot)\in\mathbb{R}^+$ takes as input the study design $D$ and outputs $y=\{f(D),1\}$ if the design is feasible and $y=\{\texttt{np.nan}, 0\}$ if the study design is statistically impossible.
    3. Fit the two models (1) GPR for the real-valued $f(D)$ and (2) a VPR for the indicator $\{0,1\}$ for statistical possibility.
    4. Run the Bayesian optimization loop

    The goal of this type of model is to increase the number of feasible designs output by the model in a way that differs from only penalizing statistically impossible designs with large $f(D)$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simplified example of failure region objective function output
    """)
    return


@app.cell
def _(np):
    # a simplified example of step 2's objective function
    def objective_function(D):
        if np.isfinite(D):
            return (3, 1)
        else:
            return (np.nan, 0)

    return (objective_function,)


@app.cell
def _(objective_function):
    # example output for a feasible design D1 = 3
    D1 = 3
    objective_function(D1)
    return


@app.cell
def _(np, objective_function):
    # example output for a statistically impossible desing D2 = np.nan
    D2 = np.nan
    objective_function(D2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The $\{0,1\}$ output is used to train the classification model and the real value is used to train the regression model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Setup required imports
    """)
    return


@app.cell
def _():
    # scientific computing imports
    import numpy as np
    import pandas as pd
    from scipy import stats
    from scipy import optimize
    import matplotlib.pyplot as plt

    return np, pd, plt


@app.cell
def _():
    import tensorflow as tf

    return (tf,)


@app.cell
def _():
    # imports for GP regression
    import gpflow
    from gpflow.keras import tf_keras

    return gpflow, tf_keras


@app.cell
def _():
    import trieste
    from trieste.space import Box

    # for the two models to be used
    from trieste.models.gpflow.models import GaussianProcessRegression
    from trieste.models.gpflow.models import VariationalGaussianProcess

    # to combine the two models into a dictionary
    from trieste.models import TrainableProbabilisticModel
    from trieste.types import Tag

    # to select the Adam optimizer
    from trieste.models.optimizer import BatchOptimizer

    # to create custom aquisition function
    from trieste.acquisition.rule import EfficientGlobalOptimization
    from trieste.acquisition import (
        SingleModelAcquisitionBuilder,
        ExpectedImprovement,
        Product,
    )

    return (
        BatchOptimizer,
        Box,
        EfficientGlobalOptimization,
        ExpectedImprovement,
        GaussianProcessRegression,
        Product,
        SingleModelAcquisitionBuilder,
        Tag,
        TrainableProbabilisticModel,
        VariationalGaussianProcess,
        trieste,
    )


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

    return bd, fmt_bd, fn_min, fp, gen_input, sim, ss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Set the defaults for this Bayes opt run
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
    # Create an objective function that returns NaN as failure
    """)
    return


@app.cell
def _(fmt_bd, fn_min, fp, gen_input, np, num_analyses, ss, tf):
    # this function contains a penalty for non-monotonicity
    def observe_new_xy(
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

        # if the input value are not monotonic, don't bother with
        # all the below calculations
        if fmt_bd.check_monotonicity(n_analyses=num_analyses, bounds=x) == False:
            y = np.nan

            return {
                "failure" : {
                    "x" : np.array([x]),
                    "y" : np.array([[tf.cast(np.isfinite(y), tf.float64)]])
                }
            }

        else:
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

            y = fn_min.function_to_minimize(max_ess_val=max_ess_new/mu, penalty=penalty)

            return {
                "objective" : {
                    "x" : np.array([x]),
                    "y" : np.array([[y]])
                },
                "failure" : {
                    "x" : np.array([x]),
                    "y" : np.array([[tf.cast(np.isfinite(y), tf.float64)]])
                }
            }

    return (observe_new_xy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Generate initial input/output dataset
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Point 1 - Pocock
    """)
    return


@app.cell
def _(
    assumed_variance,
    bd,
    important_diff_delta,
    mu,
    num_analyses,
    observe_new_xy,
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

    point1 = observe_new_xy(
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
    return (point1,)


@app.cell
def _(point1):
    point1
    return


@app.cell
def _(point1):
    point1["objective"]
    return


@app.cell
def _(point1):
    point1["objective"]["x"]
    return


@app.cell
def _(point1):
    point1["failure"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Point 2 - O'Brein-Fleming
    """)
    return


@app.cell
def _(
    assumed_variance,
    bd,
    important_diff_delta,
    mu,
    num_analyses,
    observe_new_xy,
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

    point2 = observe_new_xy(
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
    return (point2,)


@app.cell
def _(point2):
    point2["objective"]
    return


@app.cell
def _(point2):
    point2["failure"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Point 3 - Triangular
    """)
    return


@app.cell
def _(
    assumed_variance,
    bd,
    important_diff_delta,
    mu,
    num_analyses,
    observe_new_xy,
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

    point3 = observe_new_xy(
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
    return (point3,)


@app.cell
def _(point3):
    point3["objective"]
    return


@app.cell
def _(point3):
    point3["failure"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order for the functions below to work, we need  a Trieste dataset as a dictionary with:

    ```
    initial_data = {
        "objective": trieste.data.Dataset(query_points = x, observations = y),
        "failure"  : trieste.data.Dataset(query_points = x, observations = y)
    }
    ```

    where

    ```
    x = np.array(
        [
         [x1],
         [x2]
        ]
    )
    ```

    and y similarly.
    """)
    return


@app.cell
def _(point1):
    point1["objective"]["x"]
    return


@app.cell
def _(np, point1, point2, point3):
    objective_query_points = np.concatenate(
        (
            point1["objective"]["x"],
            point2["objective"]["x"],
            point3["objective"]["x"]
        )
    )
    return (objective_query_points,)


@app.cell
def _(np, point1, point2, point3):
    objective_observations = np.concatenate(
        (
            point1["objective"]["y"],
            point2["objective"]["y"],
            point3["objective"]["y"]
        )
    )
    return (objective_observations,)


@app.cell
def _(np, point1, point2, point3):
    failure_query_points = np.concatenate(
        (
            point1["failure"]["x"],
            point2["failure"]["x"],
            point3["failure"]["x"]
        )
    )
    return (failure_query_points,)


@app.cell
def _(np, point1, point2, point3):
    failure_observations = np.concatenate(
        (
            point1["failure"]["y"],
            point2["failure"]["y"],
            point3["failure"]["y"]
        )
    )
    return (failure_observations,)


@app.cell
def _(
    failure_observations,
    failure_query_points,
    objective_observations,
    objective_query_points,
    trieste,
):
    initial_data = {
        "objective" : trieste.data.Dataset(query_points = objective_query_points, observations = objective_observations),
        "failure"   : trieste.data.Dataset(query_points = failure_query_points, observations = failure_observations)
    }
    return (initial_data,)


@app.cell
def _(initial_data):
    initial_data["objective"]
    return


@app.cell
def _(initial_data):
    initial_data["failure"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Build the models for objective and failure
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GPR
    """)
    return


@app.cell
def _(gpflow):
    def build_gpr_model(X, Y, kernel, likelihood):

        gpr = gpflow.models.GPR(
            data = (X, Y),
            kernel = kernel,
            likelihood = likelihood
        )

        gpflow.utilities.print_summary(gpr)

        return gpr

    return (build_gpr_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GPR kernel
    """)
    return


@app.cell
def _(gpflow):
    gpr_kernel = gpflow.kernels.Matern52()

    # there are 6 dimension in the input requiring 6-d lengthscales
    # upper1, upper2, upper3, lower1, lower2, n
    gpr_kernel.lengthscales = gpflow.Parameter(value = [1, 1, 1, 1, 1, 1], trainable = True)

    gpr_likelihood = gpflow.likelihoods.Gaussian()
    return gpr_kernel, gpr_likelihood


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build the GPR model
    """)
    return


@app.cell
def _(build_gpr_model, gpr_kernel, gpr_likelihood, initial_data):
    gpr_model = build_gpr_model(
        X = initial_data["objective"].query_points, 
        Y = initial_data["objective"].observations,
        kernel = gpr_kernel,
        likelihood = gpr_likelihood
    )
    return (gpr_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## VGP
    """)
    return


@app.cell
def _(gpflow):
    def build_vgp_model(X, Y, kernel, likelihood):

        vgp = gpflow.models.VGP(
            data = (X, Y),
            kernel = kernel,
            likelihood = likelihood
        )

        gpflow.utilities.print_summary(vgp)

        return vgp

    return (build_vgp_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## VGP kernel
    """)
    return


@app.cell
def _(gpflow):
    vgp_kernel = gpflow.kernels.SquaredExponential()

    # there are 6 dimension in the input requiring 6-d lengthscales
    # upper1, upper2, upper3, lower1, lower2, n
    vgp_kernel.lengthscales = gpflow.Parameter(value = [1, 1, 1, 1, 1, 1], trainable = True)

    vgp_likelihood = gpflow.likelihoods.Bernoulli()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build VGP model
    """)
    return


@app.cell
def _(build_vgp_model, gpr_kernel, gpr_likelihood, initial_data):
    vgp_model = build_vgp_model(
        X = initial_data["failure"].query_points, 
        Y = initial_data["failure"].observations,
        kernel = gpr_kernel,
        likelihood = gpr_likelihood
    ) 
    return (vgp_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Create a custom aquisition function with these two models
    """)
    return


@app.cell
def _(
    BatchOptimizer,
    GaussianProcessRegression,
    Tag,
    TrainableProbabilisticModel,
    VariationalGaussianProcess,
    gpr_model,
    tf_keras,
    vgp_model,
):
    # combine the models
    models: dict[Tag, TrainableProbabilisticModel] = {

        "objective": GaussianProcessRegression(model = gpr_model),

        "failure"  : VariationalGaussianProcess(
            model = vgp_model,
            optimizer = BatchOptimizer(
                tf_keras.optimizers.Adam(learning_rate = 1e-3)
            ),
            use_natgrads = True
        )

    }
    return (models,)


@app.cell
def _(
    EfficientGlobalOptimization,
    ExpectedImprovement,
    Product,
    SingleModelAcquisitionBuilder,
    tf,
    trieste,
):
    # custom aquisition
    class ProbabilityOfValidity(SingleModelAcquisitionBuilder):
        def prepare_acquisition_function(self, model, dataset=None):
            def acquisition(at):
                mean, _ = model.predict_y(tf.squeeze(at, -2))
                return mean

            return acquisition


    ei = ExpectedImprovement()
    pov = ProbabilityOfValidity()
    acq_fn = Product(ei.using("objective"), pov.using("failure"))
    rule = EfficientGlobalOptimization(
        acq_fn,
        optimizer = trieste.acquisition.optimizer.generate_continuous_optimizer(
                num_optimization_runs = 5000
            )
    )  # type: ignore
    return (rule,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Initialize `ask_tell` interface
    """)
    return


@app.cell
def _(Box):
    # define the search space
    search_space = Box(
        lower = [-3,-3,-3,-3,-3, 10], 
        upper = [ 3, 3, 3, 3, 3, 30]
    )
    return (search_space,)


@app.cell
def _(
    initial_data,
    models: "dict[Tag, TrainableProbabilisticModel]",
    rule,
    search_space,
    trieste,
):
    ask_tell = trieste.ask_tell_optimization.AskTellOptimizer(
        search_space = search_space,
        datasets = initial_data,
        models = models,
        acquisition_rule = rule
    )
    return (ask_tell,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Run the Bayes opt loop
    """)
    return


@app.cell
def _(
    ask_tell,
    assumed_variance,
    fmt_bd,
    important_diff_delta,
    mu,
    np,
    num_analyses,
    observe_new_xy,
    sim,
    target_alpha,
    target_power,
    tf,
    trieste,
):
    num_repeats = 1000
    when_to_print = 50

    for _i in range(num_repeats):

        # get the next set of query points
        x_results = ask_tell.ask()

        # format them for calculations
        new_inputs = fmt_bd.format_boundaries_after_ask(
            n_analyses = num_analyses,
            result = x_results
        )

        # initialize an empty dictionary for the `.tell()` data
        new_data = dict()

        # if the results are not monotonic, then don't bother with the below
        if fmt_bd.check_monotonicity(n_analyses=num_analyses, bounds=x_results[0]) == False:

            # observation for the objective is NaN
            # create the new data object as a Trieste dataset
            new_data["failure"] = trieste.data.Dataset(
                query_points = x_results, 
                observations = np.array(
                    tf.cast(False, tf.float64), ndmin=2
                )
            )

            new_data["objective"] = trieste.data.Dataset(
                query_points=np.reshape(np.array([]), newshape=(0, x_results.shape[1])), 
                observations=np.reshape(np.array([]), newshape=(0, 1))
            )


        else:

            new_sim_trial = sim.group_sequential_designs(
                n_analyses = num_analyses,
                upper_bounds = new_inputs[0],
                lower_bounds = new_inputs[1],
                n_patients = new_inputs[2],
                null_hypothesis = 0,
                alt_hypothesis = important_diff_delta,
                variance = assumed_variance
            )

            new_point = observe_new_xy(
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

            new_data["objective"] = trieste.data.Dataset(
                query_points = new_point["objective"]["x"], 
                observations = new_point["objective"]["y"]
            )

            new_data["failure"] = trieste.data.Dataset(
                query_points = new_point["failure"]["x"],
                observations = new_point["failure"]["y"]
            )

        # tell our Bayes opt loop
        ask_tell.tell(new_data=new_data)

        # print our progress
        if (_i+1) % when_to_print == 0:
            print(f"\nLoop {_i+1} completed.", end="")
        elif (_i > when_to_print) and ((_i+1) % 5 == 0):
            print(".", end="")
    return


@app.cell
def _(ask_tell):
    ask_tell.to_result().try_get_final_datasets()["objective"]
    return


@app.cell
def _(ask_tell):
    ask_tell.to_result().try_get_final_datasets()["failure"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Summarize the GPR and VGP models
    """)
    return


@app.cell
def _(ask_tell, gpflow):
    # print out the model fits
    for tag, trieste_model in ask_tell.models.items():
        print(f"Summary for {tag}:")
        gpflow.utilities.print_summary(trieste_model.model)
    return


@app.cell
def _(assumed_variance, important_diff_delta, num_analyses, sim):
    sim.group_sequential_designs(
        n_analyses = num_analyses,
        upper_bounds = [ 2.42065196e+00,  1.97464463e+00,  1.92162758e+00],
        lower_bounds = [-2.24402125e+00,  9.87346244e-02,  1.92162758e+00],
        n_patients = 2.11819086e+01,
        null_hypothesis = 0,
        alt_hypothesis = important_diff_delta,
        variance = assumed_variance
    )
    return


@app.cell
def _(point3):
    point3["objective"]["x"]
    return


@app.cell
def _(plt):
    plt.plot([1, 2, 3], [2.42065196e+00,  1.97464463e+00,  1.92162758e+00])
    plt.plot([1,2,3], [-2.24402125e+00,  9.87346244e-02,  1.92162758e+00])
    plt.plot([1,2,3], [2.11957748e+00, 1.87345951e+00, 1.83560794e+00],color="green")
    plt.plot([1,2,3], [6.28553399e-16, 1.12407571e+00, 1.83560794e+00],color="green")
    return


@app.cell
def _(ask_tell, pd):
    failure_region_successes = pd.DataFrame(
        data=ask_tell.to_result().try_get_final_datasets()["objective"].query_points,
        columns=["upper1","upper2","upper3","lower1","lower2","n"]
    )
    return (failure_region_successes,)


@app.cell
def _(ask_tell, pd):
    failure_region_successes_penalty = pd.DataFrame(
        data=ask_tell.to_result().try_get_final_datasets()["objective"].observations,
        columns=["penalty"]
    )
    return (failure_region_successes_penalty,)


@app.cell
def _(failure_region_successes, failure_region_successes_penalty, pd):
    failure_region_bounds = pd.concat([failure_region_successes, failure_region_successes_penalty], axis=1)
    return (failure_region_bounds,)


@app.cell
def _(failure_region_bounds):
    failure_region_bounds.to_csv(path_or_buf="/workspace/2026-03-t16/failure_region_bounds.csv", index=False)
    return


@app.cell
def _(ask_tell, pd):
    failures = pd.DataFrame(
        data=ask_tell.to_result().try_get_final_datasets()["failure"].query_points,
        columns=["upper1","upper2","upper3","lower1","lower2","n"]
    )
    return (failures,)


@app.cell
def _(ask_tell, pd):
    failures_output = pd.DataFrame(
        data=ask_tell.to_result().try_get_final_datasets()["failure"].observations,
        columns=["output"]
    )
    return (failures_output,)


@app.cell
def _(failures, failures_output, pd):
    failure_bounds = pd.concat([failures, failures_output], axis=1)
    return (failure_bounds,)


@app.cell
def _(failure_bounds):
    failure_bounds.to_csv(path_or_buf="/workspace/2026-03-t16/failure_bounds.csv", index=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
