# import numpy and matplot lib
import numpy as np
import matplotlib.pyplot as plt

# import tensorflow
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# import gpflow
import gpflow as gf
from gpflow import set_trainable

# gpflow recommended config
gf.config.set_default_float(np.float64)
gf.config.set_default_jitter(1e-4)
gf.config.set_default_summary_fmt("notebook")

# generate function to convert to float64 for 
# tfp to play nicely with gpflow in 64-bit
f64 = gf.utilities.to_default_float

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

# fit the model
kernel = gf.kernels.Matern52(lengthscales=0.3)

# the mean_functions.Linear takes A and b where
# y_i = A x_i + b
mean_function = gf.mean_functions.Linear(1.0, 0.0)

model = gf.models.GPR(data, kernel, mean_function, noise_variance=0.01)

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

xx = np.linspace(0, 2, 500).reshape(500,1)
mean, var = model.predict_f(xx)

fig, ax = plt.subplots()
ax.scatter(X, Y, zorder = 2)
ax.plot(xx, mean, lw = 2)
ax.fill_between(
    xx[:,0],
    mean[:,0] - 1.96 * np.sqrt(var[:,0]),
    mean[:,0] + 1.96 * np.sqrt(var[:,0]),
    color = 'C0',
    alpha = 0.2,
    zorder = 2
)
ax.grid(True, alpha = 0.5, linestyle = "--")

model.kernel.lengthscales.prior = tfd.Gamma(f64(1.0), f64(1.0))
model.kernel.variance.prior     = tfd.Gamma(f64(1.0), f64(1.0))
model.likelihood.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))

model.mean_function.A.prior     = tfd.Normal(f64(0.0), f64(10.0))
model.mean_function.b.prior     = tfd.Normal(f64(0.0), f64(10.0))

# burn in and samples
n_burnin = 500
n_samples = 10000

# initialize the HMC helper
hmc_helper = gf.optimizers.SamplingHelper(
    target_log_prob_fn = model.log_posterior_density,

    # parameters have the priors not the variables
    # model fitting involves trainable_variables
    parameters = model.trainable_parameters
)

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

@tf.function
def run_chain_fn():
    return tfp.mcmc.sample_chain(
        num_results = n_samples,
        num_burnin_steps = n_burnin,
        current_state = hmc_helper.current_state,
        kernel = adaptive_hmc,
        trace_fn = lambda _, pkr: pkr.inner_results.is_accepted
    )

samples, traces = run_chain_fn()

samples

traces

np.sum(traces)

parameter_samples = hmc_helper.convert_to_constrained_values(samples)

param_to_name

plt.figure(figsize = (8,4))

for val, param in zip(parameter_samples, model.trainable_parameters):
    plt.plot(tf.squeeze(val), label = param_to_name[param])

plt.legend(bbox_to_anchor = (1, 1))
plt.ylabel("Constrained parameter values")
plt.show()

# list of maximum likelihood estimates in the correct order
mle = [0.07944876238377, 0.8535550317390823, 0.016521567744902554, -0.79271744, 3.37838]

fig, axes = plt.subplots(
    1, len(param_to_name), figsize=(15, 3), constrained_layout=True
)

for ax, val, param, mle in zip(axes, parameter_samples, model.trainable_parameters, mle):
    ax.hist(np.stack(val).flatten(), bins=20)
    ax.set_title(param_to_name[param])
    ax.axvline(mle, 0, 1, color = 'r')

fig.suptitle("Constrained parameter values")
plt.show()

# plot the function posterior
xx = np.linspace(-0.1, 1.1, 100)[:, None]
plt.figure(figsize=(12, 6))

for i in range(0, n_samples, 200):
    for var, var_samples in zip(hmc_helper.current_state, samples):
        var.assign(var_samples[i])
    f = model.predict_f_samples(xx, 1)
    plt.plot(xx, f[0, :, :], "C0", lw=2, alpha=0.3)

plt.plot(X, Y, "kx", mew=2)
plt.xlim(xx.min(), xx.max())
plt.ylim(0, 6)
plt.xlabel("$x$")
plt.ylabel("$f|X,Y$")
plt.title("Posterior GP samples")

plt.show()

for var, var_samples in zip(hmc_helper.current_state, samples):
    var.assign(var_samples[8300])
f = model.predict_f_samples(xx, 1)
plt.plot(xx, f[0, :, :], "C0", lw=2, alpha=0.3)

plt.plot(X, Y, "kx", mew=2)
plt.xlim(xx.min(), xx.max())
plt.ylim(0, 6)
plt.xlabel("$x$")
plt.ylabel("$f$ given $X,Y$")
plt.title("Posterior GP samples")

plt.show()