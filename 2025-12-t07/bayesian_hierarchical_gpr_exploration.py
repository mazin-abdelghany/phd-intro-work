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
    # import numpy and matplot lib
    import numpy as np
    import matplotlib.pyplot as plt
    return np, plt


@app.cell
def _():
    # import tensorflow
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    return tf, tfd, tfp


@app.cell
def _():
    # import gpflow
    import gpflow as gf
    from gpflow import set_trainable
    return (gf,)


@app.cell
def _(gf, np):
    # gpflow recommended config
    gf.config.set_default_float(np.float64)
    gf.config.set_default_jitter(1e-4)
    gf.config.set_default_summary_fmt("notebook")

    # generate function to convert to float64 for 
    # tfp to play nicely with gpflow in 64-bit
    f64 = gf.utilities.to_default_float
    return (f64,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sampling hyperparameters in Gaussian process regression
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Build the GPR model.
    """)
    return


@app.cell
def _(np, plt):
    # randomized data
    rng = np.random.RandomState(42)

    N = 15


    def synthetic_data(num: int, rng: np.random.RandomState):
        X = rng.rand(num, 1)
        Y = np.sin(12 * X) + 0.66 * np.cos(25 * X) + rng.randn(num, 1) * 0.1 + 3
        return X, Y


    data = (X, Y) = synthetic_data(N, rng)

    plt.figure(figsize=(12, 6))
    plt.plot(X, Y, "kx", mew=2)
    plt.xlabel("$X$")
    plt.ylabel("$Y$")
    plt.title("toy data")
    plt.show()
    return X, Y, data


@app.cell
def _(data, gf):
    # fit the model
    kernel = gf.kernels.Matern52(lengthscales=0.3)

    # the mean_functions.Linear takes A and b where
    # y_i = A x_i + b
    mean_function = gf.mean_functions.Linear(1.0, 0.0)

    model = gf.models.GPR(data, kernel, mean_function, noise_variance=0.01)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Print the model.
    """)
    return


@app.cell
def _(model):
    model
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Optimize using maximum likelihood.
    """)
    return


@app.cell
def _(gf, model):
    # initialize the optimizer
    optimizer = gf.optimizers.Scipy()

    # minimize
    optimizer.minimize(
    
        # a closure that re-evaluates the model, returning the loss to be minimized.
        closure = model.training_loss,
    
        # the list (tuple) of variables to be optimized
        variables = model.trainable_variables,
    
        method = "CG"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Print the optimized model.
    """)
    return


@app.cell
def _(model):
    model
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot the GP model.
    """)
    return


@app.cell
def _(X, Y, model, np, plt):
    xx = np.linspace(0, 2, 500).reshape(500, 1)
    mean, _var = model.predict_f(xx)
    _fig, _ax = plt.subplots()
    _ax.scatter(X, Y, zorder=2)
    _ax.plot(xx, mean, lw=2)
    _ax.fill_between(xx[:, 0], mean[:, 0] - 1.96 * np.sqrt(_var[:, 0]), mean[:, 0] + 1.96 * np.sqrt(_var[:, 0]), color='C0', alpha=0.2, zorder=2)
    _ax.grid(True, alpha=0.5, linestyle='--')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Add priors to the hyperparameters
    """)
    return


@app.cell
def _(f64, model, tfd):
    model.kernel.lengthscales.prior = tfd.Gamma(f64(1.0), f64(1.0))
    model.kernel.variance.prior     = tfd.Gamma(f64(1.0), f64(1.0))
    model.likelihood.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))

    model.mean_function.A.prior     = tfd.Normal(f64(0.0), f64(10.0))
    model.mean_function.b.prior     = tfd.Normal(f64(0.0), f64(10.0))
    return


@app.cell
def _():
    # burn in and samples
    n_burnin = 500
    n_samples = 10000
    return n_burnin, n_samples


@app.cell
def _(gf, model):
    # initialize the HMC helper
    hmc_helper = gf.optimizers.SamplingHelper(
        target_log_prob_fn = model.log_posterior_density,

        # parameters have the priors not the variables
        # model fitting involves trainable_variables
        parameters = model.trainable_parameters
    )
    return (hmc_helper,)


@app.cell
def _(f64, hmc_helper, tfp):
    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn = hmc_helper.target_log_prob_fn,
        num_leapfrog_steps = 10,
        step_size = 0.01
    )

    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc,
        num_adaptation_steps = 10,
        target_accept_prob = f64(0.75),
        adaptation_rate = 0.1
    )
    return (adaptive_hmc,)


@app.cell
def _(adaptive_hmc, hmc_helper, n_burnin, n_samples, tf, tfp):
    @tf.function
    def run_chain_fn():
        return tfp.mcmc.sample_chain(
            num_results = n_samples,
            num_burnin_steps = n_burnin,
            current_state = hmc_helper.current_state,
            kernel = adaptive_hmc,
            trace_fn = lambda _, pkr: pkr.inner_results.is_accepted
        )
    return (run_chain_fn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In addition to returning the chain state, this function supports tracing of auxiliary variables used by the kernel. The traced values are selected by specifying `trace_fn`. By default, all kernel results are traced but in the future the default will be changed to no results being traced, so plan accordingly. See below for some examples of this feature.

    `trace_fn` is a callable that takes in the current chain state and the previous kernel results and return a **Tensor** or a nested collection of **Tensor**s that is then traced along with the chain state.
    """)
    return


@app.cell
def _(run_chain_fn):
    samples, traces = run_chain_fn()
    return samples, traces


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When this function runs are the chains for each of these parameters related? Is this being taken into account?
    """)
    return


@app.cell
def _(samples):
    samples
    return


@app.cell
def _(traces):
    traces
    return


@app.cell
def _(np, traces):
    np.sum(traces)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    All the Hamiltonian MCMC sampling takes place in an unconstrained space (where constrained parameters have been mapped via a bijector to an unconstrained space). This makes the optimization, as required in the gradient step, much easier.
    """)
    return


@app.cell
def _(hmc_helper, samples):
    parameter_samples = hmc_helper.convert_to_constrained_values(samples)
    return (parameter_samples,)


@app.cell
def _(gf, model):
    # get the parameter names that were trained
    param_to_name = {_param: name for name, _param in gf.utilities.parameter_dict(model).items()}
    return (param_to_name,)


@app.cell
def _(param_to_name):
    param_to_name
    return


@app.cell
def _(model, param_to_name, parameter_samples, plt, tf):
    plt.figure(figsize=(8, 4))
    for _val, _param in zip(parameter_samples, model.trainable_parameters):
        plt.plot(tf.squeeze(_val), label=param_to_name[_param])
    plt.legend(bbox_to_anchor=(1, 1))
    plt.ylabel('Constrained parameter values')
    plt.show()
    return


@app.cell
def _(model):
    model.trainable_parameters
    return


@app.cell
def _(model, np, param_to_name, parameter_samples, plt):
    # list of maximum likelihood estimates in the correct order
    mle = [0.07944876238377, 0.8535550317390823, 0.016521567744902554, -0.79271744, 3.37838]
    _fig, axes = plt.subplots(1, len(param_to_name), figsize=(15, 3), constrained_layout=True)
    for _ax, _val, _param, mle in zip(axes, parameter_samples, model.trainable_parameters, mle):
        _ax.hist(np.stack(_val).flatten(), bins=20)
        _ax.set_title(param_to_name[_param])
        _ax.axvline(mle, 0, 1, color='r')
    _fig.suptitle('Constrained parameter values')
    plt.show()
    return


@app.cell
def _(X, Y, hmc_helper, model, n_samples, np, plt, samples):
    xx_1 = np.linspace(-0.1, 1.1, 100)[:, None]
    plt.figure(figsize=(12, 6))
    for i in range(0, n_samples, 200):
        for _var, _var_samples in zip(hmc_helper.current_state, samples):
            _var.assign(_var_samples[i])
        _f = model.predict_f_samples(xx_1, 1)
        plt.plot(xx_1, _f[0, :, :], 'C0', lw=2, alpha=0.3)
    plt.plot(X, Y, 'kx', mew=2)
    plt.xlim(xx_1.min(), xx_1.max())
    plt.ylim(0, 6)
    plt.xlabel('$x$')
    plt.ylabel('$f|X,Y$')
    plt.title('Posterior GP samples')
    plt.show()
    return (xx_1,)


@app.cell
def _(X, Y, hmc_helper, model, plt, samples, xx_1):
    for _var, _var_samples in zip(hmc_helper.current_state, samples):
        _var.assign(_var_samples[8300])
    _f = model.predict_f_samples(xx_1, 1)
    plt.plot(xx_1, _f[0, :, :], 'C0', lw=2, alpha=0.3)
    plt.plot(X, Y, 'kx', mew=2)
    plt.xlim(xx_1.min(), xx_1.max())
    plt.ylim(0, 6)
    plt.xlabel('$x$')
    plt.ylabel('$f$ given $X,Y$')
    plt.title('Posterior GP samples')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
