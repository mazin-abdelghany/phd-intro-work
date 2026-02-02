import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # run setup for notebook
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext rpy2.ipython
    # magic command not supported in marimo; please file an issue to add support
    # %config InlineBackend.figure_format='retina'
    return


@app.cell
def _():
    # import necessary libraries
    import numpy as np
    import matplotlib.pyplot as plt

    import gpflow as gf
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    return gf, np, plt, tf, tfd, tfp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Generate data to fit.
    """)
    return


@app.cell
def _(np):
    # x axis values
    x = np.linspace(start = 0, stop = 2 * np.pi, num = 100)

    # random noise
    rng = np.random.default_rng()

    # divide by 5 so that the sine wave is still readable
    noise = rng.normal(size = 100) / 4

    # y is sin(x) plus random noise
    y = np.sin(x) + noise
    return x, y


@app.cell
def _(tf, x, y):
    # make the variables into tensors for compatibility below
    # also need to shape them into column vectors
    x_tensor = tf.convert_to_tensor(x.reshape(100, 1) , dtype = "float64")
    y_tensor = tf.convert_to_tensor(y.reshape(100, 1), dtype = "float64")
    return x_tensor, y_tensor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plot the data.
    """)
    return


@app.cell
def _(plt, x, y):
    _fig, _ax = plt.subplots()
    _ax.scatter(x, y, zorder=2)
    _ax.grid(True, alpha=0.5, linestyle='--')
    _ax.set_xlabel('$x$')
    _ax.set_ylabel('$y$')
    _ax.set_title('Sine of $x$ with noise')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploring transformations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Initialize a GP regression model.
    """)
    return


@app.cell
def _(gf):
    # initialize the kernel and likelihood
    se_kernel = gf.kernels.SquaredExponential()
    gaus_likelihood = gf.likelihoods.Gaussian()

    # remove the transforms from the variables
    se_kernel.variance = gf.Parameter(value = 1, transform = None)
    se_kernel.lengthscales = gf.Parameter(value = 1, transform = None)

    gaus_likelihood.variance = gf.Parameter(value = 1, transform = None)
    return gaus_likelihood, se_kernel


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Print the kernel and likelihood.
    """)
    return


@app.cell
def _(se_kernel):
    se_kernel
    return


@app.cell
def _(gaus_likelihood):
    gaus_likelihood
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can see from the output above that the default transforms (Softplus) have been removed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we can assess how this identity transform affects model fitting.
    """)
    return


@app.cell
def _(gaus_likelihood, gf, se_kernel, x, y):
    # initialize the model
    se_model = gf.models.GPR(
        data = (x.reshape(100,1), y.reshape(100,1)),
        kernel = se_kernel,
        likelihood = gaus_likelihood
    )
    return (se_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Optimize the model using the conjugate gradient method.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below fails with error.
    """)
    return


@app.cell
def _(gf, se_model):
    optimizer = gf.optimizers.Scipy()
    optimizer.minimize(

        # a closure that re-evaluates the model, returning the loss to be minimized.
        closure = se_model.training_loss,

        # the list (tuple) of variables to be optimized
        variables = se_model.trainable_variables,

        method = "CG"
    )
    return (optimizer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see that without the transform, the optimization fails. Let's reset the kernel and likelihood trainable variables, but rather than use transforms, let us set priors on the variables to ensure that values remain appropriate during training.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Generate a half normal distribution.
    """)
    return


@app.cell
def _(gf, tfd):
    half_normal = tfd.HalfNormal(
        # default float is tf.float64
        scale = gf.utilities.to_default_float(30)
    )
    return (half_normal,)


@app.cell
def _(half_normal):
    # confirm that float64 is generated
    half_normal.sample(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot the density.
    """)
    return


@app.cell
def _(half_normal, plt):
    _fig, _ax = plt.subplots()
    _ax.hist(half_normal.sample(10000), density=True, bins=100, zorder=2)
    _ax.grid(True, alpha=0.5, linestyle='--')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use the generated half normal distribution as the prior on the parameters.
    """)
    return


@app.cell
def _(gf, half_normal):
    # initialize the kernel and likelihood
    se_kernel_1 = gf.kernels.SquaredExponential()
    gaus_likelihood_1 = gf.likelihoods.Gaussian()
    se_kernel_1.variance = gf.Parameter(value=1, transform=None, prior=half_normal, prior_on='unconstrained')
    # remove the transforms from the variables
    se_kernel_1.lengthscales = gf.Parameter(value=1, transform=None, prior=half_normal, prior_on='unconstrained')
    gaus_likelihood_1.variance = gf.Parameter(value=1, transform=None, prior=half_normal, prior_on='unconstrained')
    return gaus_likelihood_1, se_kernel_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Print the kernel and likelihood.
    """)
    return


@app.cell
def _(se_kernel_1):
    se_kernel_1
    return


@app.cell
def _(gaus_likelihood_1):
    gaus_likelihood_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Fit the model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below fails with error.
    """)
    return


@app.cell
def _(gaus_likelihood_1, gf, optimizer, se_kernel_1, x_tensor, y_tensor):
    # initialize the model
    se_model_1 = gf.models.GPR(data=(x_tensor, y_tensor), kernel=se_kernel_1, likelihood=gaus_likelihood_1)
    optimizer.minimize(closure=se_model_1.training_loss, variables=se_model_1.trainable_variables, method='CG')  # a closure that re-evaluates the model, returning the loss to be minimized.  # the list (tuple) of variables to be optimized
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The above still breaks even though the priors restrict the values of the trainable variables to positive reals. This occurs whether the prior is set on the constrained or unconstrained parameter values. In order for the  code to run correctly, the transform needs to remain.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Marginal likelihood of different SE kernels
    """)
    return


@app.cell
def _(gf, half_normal):
    # initialize the kernel and likelihood
    se_kernel_2 = gf.kernels.SquaredExponential()
    gaus_likelihood_2 = gf.likelihoods.Gaussian()
    se_kernel_2.variance.prior = half_normal
    # DO NOT REMOVE THE TRANSFORMS
    # add the priors
    se_kernel_2.lengthscales.prior = half_normal
    gaus_likelihood_2.variance.prior = half_normal
    return gaus_likelihood_2, se_kernel_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Print the kernel and likelihood.
    """)
    return


@app.cell
def _(se_kernel_2):
    se_kernel_2
    return


@app.cell
def _(gaus_likelihood_2):
    gaus_likelihood_2
    return


@app.cell
def _(gaus_likelihood_2, gf, optimizer, se_kernel_2, x_tensor, y_tensor):
    # initialize the model
    se_model_2 = gf.models.GPR(data=(x_tensor, y_tensor), kernel=se_kernel_2, likelihood=gaus_likelihood_2)
    optimizer.minimize(closure=se_model_2.training_loss, variables=se_model_2.trainable_variables, method='CG')  # a closure that re-evaluates the model, returning the loss to be minimized.  # the list (tuple) of variables to be optimized
    return (se_model_2,)


@app.cell
def _(np, plt, se_model_2, x, y):
    _xx = np.linspace(0, 2 * np.pi, 100).reshape(100, 1)
    _mean, _var = se_model_2.predict_f(_xx)
    _fig, _ax = plt.subplots()
    _ax.scatter(x, y, zorder=2)
    _ax.plot(_xx, _mean, lw=2)
    _ax.fill_between(_xx[:, 0], _mean[:, 0] - 1.96 * np.sqrt(_var[:, 0]), _mean[:, 0] + 1.96 * np.sqrt(_var[:, 0]), color='C0', alpha=0.2, zorder=2)
    _ax.grid(True, alpha=0.5, linestyle='--')
    plt.show()
    return


@app.cell
def _(se_model_2):
    se_model_2
    return


@app.cell
def _(se_model_2):
    se_model_2.log_marginal_likelihood()
    return


@app.cell
def _(gf, optimizer, x_tensor, y_tensor):
    # initialize the model with no constraints
    se_model_3 = gf.models.GPR(data=(x_tensor, y_tensor), kernel=gf.kernels.SquaredExponential(), likelihood=gf.likelihoods.Gaussian())
    optimizer.minimize(closure=se_model_3.training_loss, variables=se_model_3.trainable_variables, method='CG')  # a closure that re-evaluates the model, returning the loss to be minimized.  # the list (tuple) of variables to be optimized
    return (se_model_3,)


@app.cell
def _(np, plt, se_model_3, x, y):
    _xx = np.linspace(0, 2 * np.pi, 100).reshape(100, 1)
    _mean, _var = se_model_3.predict_f(_xx)
    _fig, _ax = plt.subplots()
    _ax.scatter(x, y, zorder=2)
    _ax.plot(_xx, _mean, lw=2)
    _ax.fill_between(_xx[:, 0], _mean[:, 0] - 1.96 * np.sqrt(_var[:, 0]), _mean[:, 0] + 1.96 * np.sqrt(_var[:, 0]), color='C0', alpha=0.2, zorder=2)
    _ax.grid(True, alpha=0.5, linestyle='--')
    plt.show()
    return


@app.cell
def _(se_model_3):
    se_model_3
    return


@app.cell
def _(se_model_3):
    se_model_3.log_marginal_likelihood()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Bad model to see how much `log_marginal_likelihood` changes.
    """)
    return


@app.cell
def _(gf, x_tensor, y_tensor):
    # initialize the model with variables that cannot be trained
    se_kernel_3 = gf.kernels.SquaredExponential()
    se_kernel_3.variance = gf.Parameter(value=0.1, trainable=False)
    se_kernel_3.lengthscales = gf.Parameter(value=0.1, trainable=False)
    se_model_4 = gf.models.GPR(data=(x_tensor, y_tensor), kernel=se_kernel_3, likelihood=gf.likelihoods.Gaussian())
    return (se_model_4,)


@app.cell
def _(se_model_4):
    se_model_4
    return


@app.cell
def _(optimizer, se_model_4):
    optimizer.minimize(closure=se_model_4.training_loss, variables=se_model_4.trainable_variables, method='CG')  # a closure that re-evaluates the model, returning the loss to be minimized.  # the list (tuple) of variables to be optimized
    return


@app.cell
def _(np, plt, se_model_4, x, y):
    _xx = np.linspace(0, 2 * np.pi, 100).reshape(100, 1)
    _mean, _var = se_model_4.predict_f(_xx)
    _fig, _ax = plt.subplots()
    _ax.scatter(x, y, zorder=2)
    _ax.plot(_xx, _mean, lw=2)
    _ax.fill_between(_xx[:, 0], _mean[:, 0] - 1.96 * np.sqrt(_var[:, 0]), _mean[:, 0] + 1.96 * np.sqrt(_var[:, 0]), color='C0', alpha=0.2, zorder=2)
    _ax.grid(True, alpha=0.5, linestyle='--')
    plt.show()
    return


@app.cell
def _(se_model_4):
    se_model_4
    return


@app.cell
def _(se_model_4):
    se_model_4.log_marginal_likelihood()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Different covariance functions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$\texttt{Matérn} = k_{M}(x) = \frac{\sigma^2}{\Gamma(\nu)2^{\nu-1}} \left(\frac{\sqrt{2 \nu} x}{l}\right)^{\nu} K_{\nu}\left(\frac{\sqrt{2 \nu} x}{l}\right)$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From [Domino datalabs](https://domino.ai/blog/fitting-gaussian-process-models-python)

    Though in general all the parameters are non-negative real-valued, when $\nu = p + \frac{1}{2}$ for integer-valued $p$, the function can be expressed partly as a polynomial function of order $p$ and generates realizations that are $p$-times differentiable, so values $\nu \in \left\{\frac{3}{2}, \frac{5}{2}\right\}$ are most common.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the below Matérn32, $\nu = \frac{3}{2}$

    $$
    \begin{align}
    \nu &= p + \frac{1}{2}\\
    \frac{3}{2} &=  p + \frac{1}{2}\\
    p &= \frac{3}{2} - \frac{1}{2} = 1
    \end{align}
    $$

    Therefore, **Matérn32 is once-differentiable.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matérn32 $\left(\nu = \frac{3}{2}\right)$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fully optimized
    """)
    return


@app.cell
def _(gf, x_tensor, y_tensor):
    _m32_kernel = gf.kernels.Matern32()
    m32_model = gf.models.GPR(data=(x_tensor, y_tensor), kernel=_m32_kernel, likelihood=gf.likelihoods.Gaussian())
    return (m32_model,)


@app.cell
def _(m32_model):
    m32_model
    return


@app.cell
def _(m32_model, optimizer):
    optimizer.minimize(

        # a closure that re-evaluates the model, returning the loss to be minimized.
        closure = m32_model.training_loss,

        # the list (tuple) of variables to be optimized
        variables = m32_model.trainable_variables,

        method = "CG"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Print the trained values.
    """)
    return


@app.cell
def _(m32_model):
    m32_model
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Print the `log_marginal_likelihood`.
    """)
    return


@app.cell
def _(m32_model):
    m32_model.log_marginal_likelihood()
    return


@app.cell
def _(m32_model, np, plt, x, y):
    _xx = np.linspace(0, 2 * np.pi, 100).reshape(100, 1)
    _mean, _var = m32_model.predict_f(_xx)
    _fig, _ax = plt.subplots()
    _ax.scatter(x, y, zorder=2)
    _ax.plot(_xx, _mean, lw=2)
    _ax.fill_between(_xx[:, 0], _mean[:, 0] - 1.96 * np.sqrt(_var[:, 0]), _mean[:, 0] + 1.96 * np.sqrt(_var[:, 0]), color='C0', alpha=0.2, zorder=2)
    _ax.grid(True, alpha=0.5, linestyle='--')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The above function appears less smooth than when the squared exponential is used, which is expected.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can explore how each of the variables affects the final predictions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Variance set as 0.001
    """)
    return


@app.cell
def _(gf, x_tensor, y_tensor):
    _m32_kernel = gf.kernels.Matern32()
    _m32_kernel.variance = gf.Parameter(value=0.001, trainable=False)
    _m32_kernel.lengthscales = gf.Parameter(value=1, trainable=True)
    m32_model_1 = gf.models.GPR(data=(x_tensor, y_tensor), kernel=_m32_kernel, likelihood=gf.likelihoods.Gaussian())
    return (m32_model_1,)


@app.cell
def _(m32_model_1):
    m32_model_1
    return


@app.cell
def _(m32_model_1, optimizer):
    optimizer.minimize(closure=m32_model_1.training_loss, variables=m32_model_1.trainable_variables, method='CG')  # a closure that re-evaluates the model, returning the loss to be minimized.  # the list (tuple) of variables to be optimized
    return


@app.cell
def _(m32_model_1):
    m32_model_1
    return


@app.cell
def _(m32_model_1):
    m32_model_1.log_marginal_likelihood()
    return


@app.cell
def _(m32_model_1, np, plt, x, y):
    _xx = np.linspace(0, 2 * np.pi, 100).reshape(100, 1)
    _mean, _var = m32_model_1.predict_f(_xx)
    _fig, _ax = plt.subplots()
    _ax.scatter(x, y, zorder=2)
    _ax.plot(_xx, _mean, lw=2)
    _ax.fill_between(_xx[:, 0], _mean[:, 0] - 1.96 * np.sqrt(_var[:, 0]), _mean[:, 0] + 1.96 * np.sqrt(_var[:, 0]), color='C0', alpha=0.2, zorder=2)
    _ax.grid(True, alpha=0.5, linestyle='--')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Length scale set as 100
    """)
    return


@app.cell
def _(gf, tfp, x_tensor, y_tensor):
    _m32_kernel = gf.kernels.Matern32()
    _m32_kernel.variance = gf.Parameter(value=1, trainable=True, transform=tfp.bijectors.Softplus())
    _m32_kernel.lengthscales = gf.Parameter(value=0.001, trainable=False)
    m32_model_2 = gf.models.GPR(data=(x_tensor, y_tensor), kernel=_m32_kernel, likelihood=gf.likelihoods.Gaussian())
    return (m32_model_2,)


@app.cell
def _(m32_model_2):
    m32_model_2
    return


@app.cell
def _(m32_model_2, optimizer):
    optimizer.minimize(closure=m32_model_2.training_loss, variables=m32_model_2.trainable_variables, method='CG')  # a closure that re-evaluates the model, returning the loss to be minimized.  # the list (tuple) of variables to be optimized
    return


@app.cell
def _(m32_model_2):
    m32_model_2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What's the difference between log marginal likelihood and log posterior density?
    """)
    return


@app.cell
def _(m32_model_2):
    m32_model_2.log_marginal_likelihood()
    return


@app.cell
def _(m32_model_2):
    m32_model_2.log_posterior_density()
    return


@app.cell
def _(m32_model_2, np, plt, x, y):
    _xx = np.linspace(0, 2 * np.pi, 100).reshape(100, 1)
    _mean, _var = m32_model_2.predict_f(_xx)
    _fig, _ax = plt.subplots()
    _ax.scatter(x, y, zorder=2)
    _ax.plot(_xx, _mean, lw=2)
    _ax.fill_between(_xx[:, 0], _mean[:, 0] - 1.96 * np.sqrt(_var[:, 0]), _mean[:, 0] + 1.96 * np.sqrt(_var[:, 0]), color='C0', alpha=0.2, zorder=2)
    _ax.grid(True, alpha=0.5, linestyle='--')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hyperparameters as random variables
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Again From [Domino datalabs](https://domino.ai/blog/fitting-gaussian-process-models-python):

    You might have noticed that there is nothing particularly Bayesian about what we have done here. No priors have been specified, and we have just performed maximum likelihood to obtain a solution. However, priors can be assigned as variable attributes, using any one of GPflow's set of distribution classes, as appropriate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Is this above technically true?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### How does `predict_f_samples` return samples of the latent function when there is no Bayesian process?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order to better understand this, let us briefly code up a Gaussian Process model from scratch.

    We are trying to model a real process $f(\mathbf{x})$ where $f$ is a function that takes a vector $\mathbf{x}$ as input and gives a vector $\mathbf{y}$ as output.

    A Gaussian Process is a **distribution over this function** and is typically specified by two other functions, specifically:
    - a mean function $m(\mathbf{x})$ and
    - a covariance function $k(\mathbf{x}, \mathbf{x'})$:

    $$
    f(\mathbf{x}) \sim GP(\,\,m(\mathbf{x}), k(\mathbf{x},\mathbf{x'})\,\,)
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### First insight
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The specification of the covariance function implies a distribution over functions. Let us take the example of the squared exponential function (**modeling without noise**):
    $$k_{\textrm{SE}}(x, x') = \sigma^2_f\exp\left(-\frac{(x - x')^2}{2\ell^2}\right)$$

    There are two parameters in the above function:
    - $\sigma^2_f$, **the signal variance**, controls how far away the function draws are "allowed" to go away from the mean (i.e., in the $y$ direction)
    - $l$, the **length-scale**, controls how quickly the correlation between two points changes, which encodes how quickly the function is "allowed" to change in the $x$ direction.
    """)
    return


@app.cell
def _(np):
    def squaredExponential(x, x_prime, signal_variance, length_scale):
        return signal_variance * np.exp( -( (x - x_prime)**2 / (2 * length_scale**2) ) )
    return (squaredExponential,)


@app.cell
def _(np, squaredExponential):
    # distances away from a value x
    plot_vals = np.linspace(start = -4, stop = 4, num = 500)

    # calculating the kernel value for each distance away from x = 0
    # signal_variance = 1, length_scale = 1
    y1 = squaredExponential(x = 0, x_prime = plot_vals,
                            signal_variance = 1, length_scale = 1)

    # signal_variance = 0.1, length_scale = 1
    y2 = squaredExponential(x = 0, x_prime = plot_vals,
                            signal_variance = 0.5, length_scale = 1)

    # signal_variance = 1, length_scale = 1
    y3 = squaredExponential(x = 0, x_prime = plot_vals,
                            signal_variance = 0.5, length_scale = 1)

    # signal_variance = 1, length_scale = 0.1
    y4 = squaredExponential(x = 0, x_prime = plot_vals,
                            signal_variance = 0.5, length_scale = 0.3)
    return plot_vals, y1, y2, y3, y4


@app.cell
def _(plot_vals, plt, y1, y2, y3, y4):
    _fig, _ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True)
    _ax[0].plot(plot_vals, y1, label='$\\sigma_f^2=1$')
    _ax[0].plot(plot_vals, y2, label='$\\sigma_f^2=0.5$')
    _ax[0].set_title('Varying signal variance, $l=1$')
    _ax[0].legend()
    _ax[1].plot(plot_vals, y3, color='orange', label='$l=1$')
    _ax[1].plot(plot_vals, y4, color='purple', label='$l=0.3$')
    _ax[1].set_title('Varying length-scale, $\\sigma^2_f=1$')
    _ax[1].legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To see the distribution over functions that the covariance matrix is specifying, we:
    1. choose a number, $n$, of input points ($\mathbf{x} \in \mathbb{R}^n$)
    2. calculate the corresponding covariance matrix **elementwise** (generates a matrix $\mathbf{K}\in\mathbb{R}^{n\times n}$)
    3. we generate a random Gaussian vector with this covariance matrix

    Explicitly, the function value $f_1$ will be assigned, assuming 0 mean function, as:
    $$
    f_1 = N(\mathbf{0}, \texttt{covariance\_matrix})
    $$

    Note! The 0 is $\mathbf{0}$, which is a vector of zeros of length $n$ and the covariance matrix is of dimension $nxn$.
    """)
    return


@app.cell
def _(np, squaredExponential):
    x_1 = np.linspace(start=-2, stop=4, num=9)
    z = np.empty(shape=(len(x_1), len(x_1)))
    for _i in range(len(x_1)):
        for j in range(len(x_1)):
            z[_i, j] = squaredExponential(x_1[_i], x_1[j], 1, 1)
    z
    return x_1, z


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It turns out that there is a universal function (`ufunc`) in Numpy that can complete the elementwise operations---``numpy.ufunc.outer``. We can redefine the `squaredExponential` function above applying this universal function.
    """)
    return


@app.cell
def _(np):
    def squaredExponential_1(x, x_prime, signal_variance, length_scale):
        return signal_variance * np.exp(-(np.subtract.outer(x, x_prime) ** 2 / (2 * length_scale ** 2)))
    return (squaredExponential_1,)


@app.cell
def _(squaredExponential_1, x_1):
    covariance_matrix = squaredExponential_1(x_1, x_1, signal_variance=1, length_scale=1)
    return (covariance_matrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This new function gives the same output, but is likely optimized for speed.
    """)
    return


@app.cell
def _(covariance_matrix, np, z):
    np.array_equal(z, covariance_matrix)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We have now defined the distribution of $f(\mathbf{x})$!
    $$
    f(\mathbf{x}) = N(\mathbf{0}, \texttt{covariance\_matrix})
    $$


    Now, we can generate samples!
    """)
    return


@app.cell
def _(covariance_matrix, np, tf, tfd, x_1):
    # define a multivariate normal as above
    f = tfd.MultivariateNormalTriL(loc=np.zeros(len(x_1)), scale_tril=tf.linalg.cholesky(covariance_matrix))
    return (f,)


@app.cell
def _(f):
    # draw 10 samples from it
    f_samples = f.sample(10)
    return (f_samples,)


@app.cell
def _(f_samples, plt, x_1):
    for _i in range(f_samples.shape[0]):
        plt.plot(x_1, f_samples[_i])
    return


@app.cell
def _(np, squaredExponential_1, tf, tfd):
    # assuming a much finer and longer x space
    x_2 = np.linspace(start=-5, stop=5, num=500)
    covariance_matrix1 = squaredExponential_1(x_2, x_2, signal_variance=1, length_scale=1)
    # calculate the covariance matrix
    covariance_matrix2 = squaredExponential_1(x_2, x_2, signal_variance=0.3, length_scale=1)
    covariance_matrix3 = squaredExponential_1(x_2, x_2, signal_variance=1, length_scale=0.1)
    f1 = tfd.MultivariateNormalTriL(loc=np.zeros(len(x_2)), scale_tril=tf.linalg.cholesky(covariance_matrix1 + tf.eye(len(x_2), dtype=tf.float64) * 1e-06))
    f2 = tfd.MultivariateNormalTriL(loc=np.zeros(len(x_2)), scale_tril=tf.linalg.cholesky(covariance_matrix2 + tf.eye(len(x_2), dtype=tf.float64) * 1e-06))
    # define a multivariate normal
    f3 = tfd.MultivariateNormalTriL(loc=np.zeros(len(x_2)), scale_tril=tf.linalg.cholesky(covariance_matrix3 + tf.eye(len(x_2), dtype=tf.float64) * 1e-06))
    f1 = f1.sample(10)  # mean vector of 0s
    f2 = f2.sample(10)
    # draw 10 samples from it
    f3 = f3.sample(10)  # triangular matrix for Cholesky decomp  # with jitter added for numerical stability at float64
    return f1, f2, f3, x_2


@app.cell
def _(f1, f2, f3, np, plt, x_2):
    _fig, _ax = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for _i in range(f1.shape[0]):
        _ax[0].plot(x_2, f1[_i])
        _ax[1].plot(x_2, f2[_i])
        _ax[2].plot(x_2, f3[_i])
    _ax[0].fill_between(x_2, 0 - 1.96 * np.sqrt(1), 0 + 1.96 * np.sqrt(1), alpha=0.15, color='grey')
    _ax[1].fill_between(x_2, 0 - 1.96 * np.sqrt(0.3), 0 + 1.96 * np.sqrt(0.3), alpha=0.15, color='grey')
    _ax[2].fill_between(x_2, 0 - 1.96 * np.sqrt(1), 0 + 1.96 * np.sqrt(1), alpha=0.15, color='grey')
    _ax[0].set_title('signal_variance = 1, length_scale = 1')
    _ax[1].set_title('signal_variance = 0.3, length_scale = 1')
    _ax[2].set_title('signal_variance = 1, length_scale = 0.1')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can recapulate the above with `GPflow`.
    """)
    return


@app.cell
def _(gf, np, x_2):
    se_empty1 = gf.models.GPR(data=(np.zeros((10, 1)), np.zeros((10, 1))), kernel=gf.kernels.SquaredExponential(variance=1, lengthscales=1))
    se_empty2 = gf.models.GPR(data=(np.zeros((10, 1)), np.zeros((10, 1))), kernel=gf.kernels.SquaredExponential(variance=0.3, lengthscales=1))
    se_empty3 = gf.models.GPR(data=(np.zeros((10, 1)), np.zeros((10, 1))), kernel=gf.kernels.SquaredExponential(variance=1, lengthscales=0.1))
    preds1 = se_empty1.predict_f_samples(x_2.reshape(len(x_2), 1), 10)
    preds2 = se_empty2.predict_f_samples(x_2.reshape(len(x_2), 1), 10)
    preds3 = se_empty3.predict_f_samples(x_2.reshape(len(x_2), 1), 10)
    return preds1, preds2, preds3


@app.cell
def _(np, plt, preds1, preds2, preds3, x_2):
    _fig, _ax = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for _i in range(preds1.shape[0]):
        _ax[0].plot(x_2, preds1[_i])
        _ax[1].plot(x_2, preds2[_i])
        _ax[2].plot(x_2, preds3[_i])
    _ax[0].fill_between(x_2, 0 - 1.96 * np.sqrt(1), 0 + 1.96 * np.sqrt(1), alpha=0.15, color='grey')
    _ax[1].fill_between(x_2, 0 - 1.96 * np.sqrt(0.3), 0 + 1.96 * np.sqrt(0.3), alpha=0.15, color='grey')
    _ax[2].fill_between(x_2, 0 - 1.96 * np.sqrt(1), 0 + 1.96 * np.sqrt(1), alpha=0.15, color='grey')
    _ax[0].set_title('signal_variance = 1, length_scale = 1')
    _ax[1].set_title('signal_variance = 0.3, length_scale = 1')
    _ax[2].set_title('signal_variance = 1, length_scale = 0.1')
    plt.show()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
