import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf

    return np, plt, tf


@app.cell
def _(plt):
    plt.rcParams["figure.dpi"] = 500
    return


@app.cell
def _(np, tf):
    np.random.seed(1793)
    tf.random.set_seed(1793)
    return


@app.cell
def _():
    from trieste.objectives import Branin
    from trieste.experimental.plotting import plot_function_plotly

    return Branin, plot_function_plotly


@app.cell
def _(Branin):
    branin = Branin.objective
    search_space = Branin.search_space
    return branin, search_space


@app.cell
def _(search_space):
    search_space
    return


@app.cell
def _(branin, np, tf):
    # threshold is arbitrary, but has to be within the range of the function
    threshold = 80.0


    # define a modified branin function
    def thresholded_branin(x):
        y = np.array(branin(x))
        y[y > threshold] = np.nan
        return tf.convert_to_tensor(y.reshape(-1, 1), x.dtype)

    return threshold, thresholded_branin


@app.cell
def _(plot_function_plotly, search_space, thresholded_branin):
    fig = plot_function_plotly(
        thresholded_branin, search_space.lower, search_space.upper
    )
    fig.show()
    return


@app.cell
def _(np, thresholded_branin):
    thresholded_branin(x=np.array([0.9,0.9]))
    return


@app.cell
def _(branin, search_space):
    import trieste

    observer = trieste.objectives.utils.mk_observer(branin)

    num_initial_points = 6
    initial_query_points = search_space.sample_halton(num_initial_points)
    initial_data = observer(initial_query_points)
    return initial_data, num_initial_points, observer, trieste


@app.cell
def _(initial_data, search_space):
    import gpflow
    from trieste.models.gpflow import GaussianProcessRegression, build_gpr


    gpflow_model = build_gpr(initial_data, search_space, likelihood_variance=1e-7)
    model = GaussianProcessRegression(gpflow_model)
    return GaussianProcessRegression, build_gpr, model


@app.cell
def _(initial_data, model, observer, search_space, threshold, trieste):
    from trieste.acquisition.rule import EfficientGlobalOptimization
    from trieste.acquisition.function import ExpectedFeasibility

    # Bichon criterion
    delta = 1

    # set up the acquisition rule and initialize the Bayesian optimizer
    acq = ExpectedFeasibility(threshold, delta=delta)
    rule = EfficientGlobalOptimization(builder=acq)  # type: ignore
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

    num_steps = 10
    result = bo.optimize(num_steps, initial_data, model, rule)
    return bo, result, rule


@app.cell
def _(num_initial_points, plt, search_space, tf, thresholded_branin):
    from trieste.experimental.plotting import plot_bo_points, plot_function_2d
    import tensorflow_probability as tfp


    def excursion_probability(x, model, threshold=80):
        mean, variance = model.model.predict_f(x)
        normal = tfp.distributions.Normal(tf.cast(0, x.dtype), tf.cast(1, x.dtype))
        threshold = tf.cast(threshold, x.dtype)

        if tf.size(threshold) == 1:
            t = (mean - threshold) / tf.sqrt(variance)
            return normal.cdf(t)
        else:
            t0 = (mean - threshold[0]) / tf.sqrt(variance)
            t1 = (mean - threshold[1]) / tf.sqrt(variance)
            return normal.cdf(t1) - normal.cdf(t0)


    def plot_excursion_probability(
        title, model=None, query_points=None, threshold=80.0
    ):
        if model is None:
            objective_function = thresholded_branin
        else:

            def objective_function(x):
                return excursion_probability(x, model, threshold)

        _, ax = plot_function_2d(
            objective_function,
            search_space.lower - 0.01,
            search_space.upper + 0.01,
            contour=True,
            colorbar=True,
            figsize=(10, 6),
            title=[title],
            xlabel="$X_1$",
            ylabel="$X_2$",
            fill=True,
        )
        if query_points is not None:
            plot_bo_points(query_points, ax[0, 0], num_initial_points)


    plot_excursion_probability("Excursion set, Branin function")

    plt.show()
    return (plot_excursion_probability,)


@app.cell
def _(
    GaussianProcessRegression,
    build_gpr,
    initial_data,
    num_initial_points,
    plot_excursion_probability,
    plt,
    result,
    search_space,
):
    # extracting the data to illustrate the points
    dataset = result.try_get_final_dataset()
    query_points = dataset.query_points.numpy()
    observations = dataset.observations.numpy()

    # fitting the model only to the initial data
    gpflow_model1 = build_gpr(initial_data, search_space, likelihood_variance=1e-7)
    initial_model = GaussianProcessRegression(gpflow_model1)
    initial_model.optimize(initial_data)

    plot_excursion_probability(
        "Probability of excursion, initial data",
        initial_model,
        query_points[:num_initial_points,],
    )
    plt.show()
    return dataset, query_points


@app.cell
def _(plot_excursion_probability, plt, query_points, result):
    updated_model = result.try_get_final_model()

    plot_excursion_probability(
        "Updated probability of excursion", updated_model, query_points
    )

    plt.show()
    return


@app.cell
def _(bo, dataset, model, plot_excursion_probability, plt, rule):
    num_steps1 = 10
    result1 = bo.optimize(num_steps1, dataset, model, rule)

    final_model = result1.try_get_final_model()
    dataset1 = result1.try_get_final_dataset()
    query_points1 = dataset1.query_points.numpy()

    plot_excursion_probability(
        "Final probability of excursion", final_model, query_points1
    )

    plt.show()
    return


if __name__ == "__main__":
    app.run()
