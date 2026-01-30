import numpy as np
import matplotlib.pyplot as plt

import gpflow
import tensorflow as tf
import tensorflow_probability 

x = np.arange(start = -2., stop = 3., step = 1.).reshape(5,1)

y = x**2

x

y

fig, ax = plt.subplots()
ax.scatter(x, y, zorder=2)
ax.grid(True)
plt.show()

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

se_kernel

likelihood

nm_model = gpflow.models.GPR(
    data = (x, y),
    kernel = se_kernel,
    likelihood = likelihood
)

# optimize using Nelder-Mead
opt = gpflow.optimizers.Scipy()
opt.minimize(
    nm_model.training_loss,
    nm_model.trainable_variables,
    method = 'Nelder-Mead'
)

xx = np.linspace(-4, 4, 100).reshape(100,1)
mean, var = nm_model.predict_f(xx)

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(xx, mean, lw = 2)
ax.fill_between(
    xx[:,0],
    mean[:,0] - 1.96 * np.sqrt(var[:,0]),
    mean[:,0] + 1.96 * np.sqrt(var[:,0]),
    color = 'C0',
    alpha = 0.2
)

nm_model

# starting with a squared exponential distribution
# with variance 1 and length scale 0.3 as above
se_kernel = gpflow.kernels.SquaredExponential(
    variance = 1,
    lengthscales = 0.3
)

# setting a uniform prior on both
se_kernel.variance.prior = tensorflow_probability.distributions.Uniform(
    low = gpflow.utilities.to_default_float(0),
    high = gpflow.utilities.to_default_float(1)
)
se_kernel.lengthscales.prior = tensorflow_probability.distributions.Uniform(
    low = gpflow.utilities.to_default_float(0),
    high = gpflow.utilities.to_default_float(1)
)

# setting the prior on the likelihood
likelihood = gpflow.likelihoods.Gaussian(
    variance = 0.5**2
)

# set a prior
likelihood.variance.prior = tensorflow_probability.distributions.Uniform(
    low = gpflow.utilities.to_default_float(0),
    high = gpflow.utilities.to_default_float(1)
)

se_kernel
likelihood

nm_model = gpflow.models.GPR(
    data = (x, y),
    kernel = se_kernel,
    likelihood = likelihood
)

# optimize using Nelder-Mead
opt = gpflow.optimizers.Scipy()
opt.minimize(
    nm_model.training_loss,
    nm_model.trainable_variables,
    method = 'Nelder-Mead'
)

xx = np.linspace(-4, 4, 100).reshape(100,1)
mean, var = nm_model.predict_f(xx)

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(xx, mean, lw = 2)
ax.fill_between(
    xx[:,0],
    mean[:,0] - 1.96 * np.sqrt(var[:,0]),
    mean[:,0] + 1.96 * np.sqrt(var[:,0]),
    color = 'C0',
    alpha = 0.2
)

nm_model

log_uniform = tensorflow_probability.distributions.TransformedDistribution(
    distribution = tensorflow_probability.distributions.Uniform(low = np.log(0.1), high = np.log(1)),
    bijector = tensorflow_probability.bijectors.Exp()
)

plt.hist(log_uniform.sample(10000))

def log_uniform_dist(x, low = 0.1, high = 1):
    return 1 / (x * np.log(high/low))

x_plot = np.linspace(0.1, 1, 100)
plt.plot(x_plot, log_uniform_dist(x_plot, low = 0.1, high = 1))

# starting with a squared exponential distribution
# with variance 1 and length scale 0.3 as above
se_kernel = gpflow.kernels.SquaredExponential(
    variance = 1,
    lengthscales = 0.3
)

# setting a uniform prior on both
se_kernel.variance.prior = log_uniform
se_kernel.lengthscales.prior = log_uniform

# setting the prior on the likelihood
likelihood = gpflow.likelihoods.Gaussian(
    variance = 0.5**2
)

# set a prior
likelihood.variance.prior = log_uniform

nm_model = gpflow.models.GPR(
    data = (x, y),
    kernel = se_kernel,
    likelihood = likelihood
)

nm_model

# optimize using Nelder-Mead
opt = gpflow.optimizers.Scipy()
opt.minimize(
    nm_model.training_loss,
    nm_model.trainable_variables,
    method = 'Nelder-Mead'
)

xx = np.linspace(-4, 4, 100).reshape(100,1)
mean, var = nm_model.predict_f(xx)

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(xx, mean, lw = 2)
ax.fill_between(
    xx[:,0],
    mean[:,0] - 1.96 * np.sqrt(var[:,0]),
    mean[:,0] + 1.96 * np.sqrt(var[:,0]),
    color = 'C0',
    alpha = 0.2
)

nm_model

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

nm_model = gpflow.models.GPR(
    data = (x, y),
    kernel = se_kernel,
    likelihood = likelihood
)

# optimize using Nelder-Mead
opt = gpflow.optimizers.Scipy()
opt.minimize(
    nm_model.training_loss,
    nm_model.trainable_variables,
    method = 'CG'
)

xx = np.linspace(-4, 4, 100).reshape(100,1)
mean, var = nm_model.predict_f(xx)

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(xx, mean, lw = 2)
ax.fill_between(
    xx[:,0],
    mean[:,0] - 1.96 * np.sqrt(var[:,0]),
    mean[:,0] + 1.96 * np.sqrt(var[:,0]),
    color = 'C0',
    alpha = 0.2
)

nm_model