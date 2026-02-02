import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Set up to use `gplite` in Jupyter Notebooks
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Set up the magic commands for R.
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext rpy2.ipython
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Improve image quality of `matplotlib` output.
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %config InlineBackend.figure_format='retina'
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Load the required R libraries.
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%R
    # library(gplite)
    # library(ggplot2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Previous `gplite` example
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Generate the required data.
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%R -o x -o y
    # x <- seq(from = -2, to = 2, by = 1)
    # y <- x^2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Initialize, run the model, and plot.
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%R
    # # Specify the GP model we want to use:
    # gp_empty <- gp_init(
    #   
    #   # A squared exponential (aka Gaussian aka RBF) kernel
    #   cfs = cf_sexp(
    #     vars = NULL,
    #     lscale = 0.3,
    #     magn = 1,
    #     prior_lscale = prior_logunif(),
    #     prior_magn = prior_logunif(),
    #     normalize = FALSE
    #   ),  
    #   
    #   # Assume Gaussian distributed errors
    #   lik = lik_gaussian(
    #     sigma = 0.5, 
    #     prior_sigma = prior_logunif()
    #   ), 
    #   
    #   # Use the full covariance (i.e., do not approximate)
    #   method = method_full() 
    #   
    # )
    # 
    # # Now fit the model to the data:
    # gp_optimized <- gp_optim(gp_empty, x, y, verbose = FALSE)
    # 
    # # compute the predictive mean and variance in a grid of points
    # xt   <- seq(-4, 4, len=150)
    # pred <- gp_pred(gp_optimized, xt, var = T)
    # 
    # # visualize
    # mu <- pred$mean
    # lb <- pred$mean - 2*sqrt(pred$var)
    # ub <- pred$mean + 2*sqrt(pred$var)
    # 
    # ggplot() + 
    #   geom_ribbon(aes(x=xt, ymin=lb, ymax=ub), fill='lightgray') +
    #   geom_line(aes(x=xt, y=mu), linewidth = 0.5) +
    #   geom_point(aes(x=x, y=y), size=2) +
    #   xlab('x') + ylab('y')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GPFlow
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    import gpflow
    import tensorflow as tf
    import tensorflow_probability
    return gpflow, np, plt, tensorflow_probability


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Generate the same data for `gpflow` as for `gplite`.
    """)
    return


@app.cell
def _(np):
    x = np.arange(start = -2., stop = 3., step = 1.).reshape(5,1)

    y = x**2
    return x, y


@app.cell
def _(x):
    x
    return


@app.cell
def _(y):
    y
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot the data.
    """)
    return


@app.cell
def _(plt, x, y):
    _fig, _ax = plt.subplots()
    _ax.scatter(x, y, zorder=2)
    _ax.grid(True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recapitulate `gp_lite` example
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us try to recapitulate the above `gp_lite` prediction graph using `gpflow`. The settings for `gp_lite` above were:
    * a squared exponential kernel
        * the length scale initial value was 0.3
        * the magnitude (function variance) initial value was 1
        * the prior on the length scale was log uniform
        * the prior on the magnitude was log uniform
    * Gaussian distributed error
        * the error initial value was 0.5
        * the prior on the error was log uniform
    * Full covariance
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    TensorFlow Probabilities does not have a log uniform distribution. Therefore, we will need to generate that distribution ourselves.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### No priors
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us first take the simple case fitting the model without priors.
    """)
    return


@app.cell
def _(gpflow):
    # starting with a squared exponential distribution
    # with variance 1 and length scale 0.3 as above
    se_kernel = gpflow.kernels.SquaredExponential(
        variance = 1,
        lengthscales = 0.3
    )

    # setting the prior on the likelihood
    likelihood = gpflow.likelihoods.Gaussian(
        variance = 0.5**2
    )
    return likelihood, se_kernel


@app.cell
def _(se_kernel):
    se_kernel
    return


@app.cell
def _(likelihood):
    likelihood
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Fit the model and optimize.
    """)
    return


@app.cell
def _(gpflow, likelihood, se_kernel, x, y):
    nm_model = gpflow.models.GPR(data=(x, y), kernel=se_kernel, likelihood=likelihood)
    _opt = gpflow.optimizers.Scipy()
    # optimize using Nelder-Mead
    _opt.minimize(nm_model.training_loss, nm_model.trainable_variables, method='Nelder-Mead')
    return (nm_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Make predictions and plot.
    """)
    return


@app.cell
def _(nm_model, np, plt, x, y):
    _xx = np.linspace(-4, 4, 100).reshape(100, 1)
    _mean, _var = nm_model.predict_f(_xx)
    _fig, _ax = plt.subplots()
    _ax.scatter(x, y)
    _ax.plot(_xx, _mean, lw=2)
    _ax.fill_between(_xx[:, 0], _mean[:, 0] - 1.96 * np.sqrt(_var[:, 0]), _mean[:, 0] + 1.96 * np.sqrt(_var[:, 0]), color='C0', alpha=0.2)
    return


@app.cell
def _(nm_model):
    nm_model
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Uniform priors
    """)
    return


@app.cell
def _(gpflow, tensorflow_probability):
    # starting with a squared exponential distribution
    # with variance 1 and length scale 0.3 as above
    se_kernel_1 = gpflow.kernels.SquaredExponential(variance=1, lengthscales=0.3)
    se_kernel_1.variance.prior = tensorflow_probability.distributions.Uniform(low=gpflow.utilities.to_default_float(0), high=gpflow.utilities.to_default_float(1))
    # setting a uniform prior on both
    se_kernel_1.lengthscales.prior = tensorflow_probability.distributions.Uniform(low=gpflow.utilities.to_default_float(0), high=gpflow.utilities.to_default_float(1))
    return (se_kernel_1,)


@app.cell
def _(gpflow, tensorflow_probability):
    # setting the prior on the likelihood
    likelihood_1 = gpflow.likelihoods.Gaussian(variance=0.5 ** 2)
    # set a prior
    likelihood_1.variance.prior = tensorflow_probability.distributions.Uniform(low=gpflow.utilities.to_default_float(0), high=gpflow.utilities.to_default_float(1))
    return (likelihood_1,)


@app.cell
def _(se_kernel_1):
    se_kernel_1
    return


@app.cell
def _(likelihood_1):
    likelihood_1
    return


@app.cell
def _(gpflow, likelihood_1, se_kernel_1, x, y):
    nm_model_1 = gpflow.models.GPR(data=(x, y), kernel=se_kernel_1, likelihood=likelihood_1)
    return (nm_model_1,)


@app.cell
def _(gpflow, nm_model_1):
    _opt = gpflow.optimizers.Scipy()
    _opt.minimize(nm_model_1.training_loss, nm_model_1.trainable_variables, method='Nelder-Mead')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Make predictions using the model
    """)
    return


@app.cell
def _(nm_model_1, np, plt, x, y):
    _xx = np.linspace(-4, 4, 100).reshape(100, 1)
    _mean, _var = nm_model_1.predict_f(_xx)
    _fig, _ax = plt.subplots()
    _ax.scatter(x, y)
    _ax.plot(_xx, _mean, lw=2)
    _ax.fill_between(_xx[:, 0], _mean[:, 0] - 1.96 * np.sqrt(_var[:, 0]), _mean[:, 0] + 1.96 * np.sqrt(_var[:, 0]), color='C0', alpha=0.2)
    return


@app.cell
def _(nm_model_1):
    nm_model_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The uniform distribution restricts the hyperparameters to $(0,1)$, which reduces---most notably---the length-scale parameter making the above graph less "wiggly".
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Log uniform priors
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Create a log uniform distribution.
    """)
    return


@app.cell
def _(np, tensorflow_probability):
    log_uniform = tensorflow_probability.distributions.TransformedDistribution(
        distribution = tensorflow_probability.distributions.Uniform(low = np.log(0.1), high = np.log(1)),
        bijector = tensorflow_probability.bijectors.Exp()
    )
    return (log_uniform,)


@app.cell
def _(log_uniform, plt):
    plt.hist(log_uniform.sample(10000))
    return


@app.cell
def _(np):
    def log_uniform_dist(x, low = 0.1, high = 1):
        return 1 / (x * np.log(high/low))
    return (log_uniform_dist,)


@app.cell
def _(log_uniform_dist, np, plt):
    x_plot = np.linspace(0.1, 1, 100)
    plt.plot(x_plot, log_uniform_dist(x_plot, low = 0.1, high = 1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can use this new distribution as the priors.
    """)
    return


@app.cell
def _(gpflow, log_uniform, x, y):
    # starting with a squared exponential distribution
    # with variance 1 and length scale 0.3 as above
    se_kernel_2 = gpflow.kernels.SquaredExponential(variance=1, lengthscales=0.3)
    se_kernel_2.variance.prior = log_uniform
    se_kernel_2.lengthscales.prior = log_uniform
    likelihood_2 = gpflow.likelihoods.Gaussian(variance=0.5 ** 2)
    likelihood_2.variance.prior = log_uniform
    # setting a uniform prior on both
    nm_model_2 = gpflow.models.GPR(data=(x, y), kernel=se_kernel_2, likelihood=likelihood_2)
    # setting the prior on the likelihood
    # set a prior
    nm_model_2
    return (nm_model_2,)


@app.cell
def _(gpflow, nm_model_2, np, plt, x, y):
    _opt = gpflow.optimizers.Scipy()
    _opt.minimize(nm_model_2.training_loss, nm_model_2.trainable_variables, method='Nelder-Mead')
    _xx = np.linspace(-4, 4, 100).reshape(100, 1)
    _mean, _var = nm_model_2.predict_f(_xx)
    _fig, _ax = plt.subplots()
    _ax.scatter(x, y)
    _ax.plot(_xx, _mean, lw=2)
    _ax.fill_between(_xx[:, 0], _mean[:, 0] - 1.96 * np.sqrt(_var[:, 0]), _mean[:, 0] + 1.96 * np.sqrt(_var[:, 0]), color='C0', alpha=0.2)
    return


@app.cell
def _(nm_model_2):
    nm_model_2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Most of the shape of the above is driven by the length scale parameter.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A different optimizer
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using all the same initial parameter values, but selecting a different optimizer, we can assess how this affects the model fit.
    """)
    return


@app.cell
def _(gpflow, np, plt, x, y):
    se_kernel_3 = gpflow.kernels.SquaredExponential(variance=1, lengthscales=0.3)
    likelihood_3 = gpflow.likelihoods.Gaussian(variance=0.5 ** 2)
    nm_model_3 = gpflow.models.GPR(data=(x, y), kernel=se_kernel_3, likelihood=likelihood_3)
    _opt = gpflow.optimizers.Scipy()
    _opt.minimize(nm_model_3.training_loss, nm_model_3.trainable_variables, method='CG')
    _xx = np.linspace(-4, 4, 100).reshape(100, 1)
    _mean, _var = nm_model_3.predict_f(_xx)
    _fig, _ax = plt.subplots()
    _ax.scatter(x, y)
    _ax.plot(_xx, _mean, lw=2)
    _ax.fill_between(_xx[:, 0], _mean[:, 0] - 1.96 * np.sqrt(_var[:, 0]), _mean[:, 0] + 1.96 * np.sqrt(_var[:, 0]), color='C0', alpha=0.2)
    return (nm_model_3,)


@app.cell
def _(nm_model_3):
    nm_model_3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Though this model fits the points very well, it could be argued that this is not the optimal solution because the model predits high precision far away from the points given. Prior distributions on the parameters could solve this "issue" by regularizing the parameters.
    """)
    return


if __name__ == "__main__":
    app.run()
