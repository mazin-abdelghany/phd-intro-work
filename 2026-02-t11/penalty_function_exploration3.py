import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo._runtime.context.get_context().marimo_config["runtime"]["output_max_bytes"] = 10000000000
    return


@app.cell
def _():
    import numpy as np
    import tensorflow as tf
    import plotly.graph_objects as go
    from scipy import stats
    return go, np, tf


@app.cell
def _():
    import matplotlib.pyplot as plt
    from matplotlib import cm
    return cm, plt


@app.cell
def _():
    import gpflow
    return (gpflow,)


@app.cell
def _():
    import trieste
    from trieste.space import Box
    from trieste.models.gpflow.models import GaussianProcessRegression
    from trieste.experimental.plotting import plot_regret
    return Box, GaussianProcessRegression, trieste


@app.cell
def _():
    from py_group_sequential_designs import sample_size as ss
    return (ss,)


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
    return


@app.function
def davids_surface(
        alpha_prime,
        beta_prime,
        alpha,
        beta):

    return 150 * ((alpha_prime - alpha)**2 + (beta_prime - beta)**2)


@app.cell
def _(np):
    beta_range = np.arange(start=0.001, stop=0.99, step=0.001)
    return (beta_range,)


@app.cell
def _(beta_range):
    alpha_range = beta_range.copy()
    return (alpha_range,)


@app.cell
def _(alpha_range, beta_range, np):
    davids_surface_vals = np.empty(shape=(len(alpha_range), len(beta_range)))

    for _i, _alphas in enumerate(alpha_range):
        for _j, _betas in enumerate(beta_range):
            davids_surface_vals[_i,_j] = davids_surface(
                alpha = 0.05, beta = 0.1, 
                beta_prime=_betas, alpha_prime=_alphas
            )
    return (davids_surface_vals,)


@app.cell
def _(alpha_range, beta_range, np):
    Alpha, Beta = np.meshgrid(alpha_range, beta_range)
    return Alpha, Beta


@app.cell
def _(Alpha, Beta, cm, davids_surface_vals, plt):
    # purple to green colormap
    cmap = cm.PRGn

    # initialize the figure
    _fig, _ax = plt.subplots()

    # contour colors
    _cset1 = _ax.contourf(
        Alpha, Beta, davids_surface_vals, levels = 100,
        cmap = cmap.resampled(49)
    )

    # contour lines
    _ax.contour(
        Alpha, Beta, davids_surface_vals, 
        levels = 10, colors = 'k'
    )

    # set the plot characteristics
    _ax.set_xlim(-0.1, 1.1)
    _ax.set_ylim(-0.1, 1.1)
    _ax.set_xlabel("alpha")
    _ax.set_ylabel("beta")

    # add the colorbar to the plot
    _fig.colorbar(_cset1, ax=_ax,
                      label="Penalty")

    plt.show()
    return


@app.cell
def _(Alpha, Beta, davids_surface_vals, go):
    _fig3d = go.Figure(data = go.Surface(z=davids_surface_vals, x=Alpha, y=Beta))
    _fig3d.update_scenes(
        xaxis_title_text="alpha",
        yaxis_title_text="beta",
        zaxis_title_text="penalty"
    )
    _fig3d.show()
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
def _(Box):
    search_space = Box(lower=[0, 0], upper=[1,1])
    return (search_space,)


@app.cell
def _(search_space):
    initial_query = search_space.sample_sobol(3)
    return (initial_query,)


@app.cell
def _(initial_query):
    initial_query
    return


@app.cell
def _(np):
    initial_observ = np.empty((3,1))
    return (initial_observ,)


@app.cell
def _(initial_observ, initial_query):
    for _i, _queries in enumerate(initial_query):
        _alpha_prime = _queries[0]
        _beta_prime = _queries[1]

        initial_observ[_i] = davids_surface(
            alpha=0.05,
            beta=0.1,
            alpha_prime=_alpha_prime,
            beta_prime=_beta_prime
        )
    return


@app.cell
def _(initial_observ):
    initial_observ
    return


@app.cell
def _(gpflow):
    kernel = gpflow.kernels.SquaredExponential()
    kernel.variance = gpflow.Parameter(value = 2500, trainable = True)
    kernel.lengthscales = gpflow.Parameter(value = [0.01, 100], trainable = True)

    likelihood = gpflow.likelihoods.Gaussian()
    likelihood.variance = gpflow.Parameter(value = 1e-5, trainable = False)
    return kernel, likelihood


@app.cell
def _(build_model, initial_observ, initial_query, kernel, likelihood):
    bayes_opt_model = build_model(
        X = initial_query, 
        Y = initial_observ,
        kernel = kernel,
        likelihood = likelihood
    )
    return (bayes_opt_model,)


@app.cell
def _(initial_observ, initial_query, trieste):
    # create a dataset that works well with trieste
    initial_data = trieste.data.Dataset(
        query_points = initial_query, 
        observations = initial_observ
    )
    return (initial_data,)


@app.cell
def _(bayes_opt_model, initial_data, search_space, trieste):
    ask_tell = trieste.ask_tell_optimization.AskTellOptimizer(
        search_space = search_space,
        datasets = initial_data,
        models = bayes_opt_model
    )
    return (ask_tell,)


@app.cell
def _(ask_tell, np, trieste):
    # takes approximately ~9 minutes to run
    num_repeats = 1000

    for i in range(num_repeats):
        query = ask_tell.ask()

        observ = davids_surface(
            alpha=0.05,
            beta=0.1,
            alpha_prime=query[0][0],
            beta_prime=query[0][1]
        )

        new_data = trieste.data.Dataset(
            query_points = query, 
            observations = np.array([[observ]])
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
def _(ask_tell, gpflow):
    gpflow.utilities.print_summary(ask_tell.model.model)
    return


if __name__ == "__main__":
    app.run()
