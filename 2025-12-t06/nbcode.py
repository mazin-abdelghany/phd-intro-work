# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

import gpflow as gf
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# x axis values
x = np.linspace(start = 0, stop = 2 * np.pi, num = 100)

# random noise
rng = np.random.default_rng()

# divide by 5 so that the sine wave is still readable
noise = rng.normal(size = 100) / 4

# y is sin(x) plus random noise
y = np.sin(x) + noise

# make the variables into tensors for compatibility below
# also need to shape them into column vectors
x_tensor = tf.convert_to_tensor(x.reshape(100, 1) , dtype = "float64")
y_tensor = tf.convert_to_tensor(y.reshape(100, 1), dtype = "float64")

fig, ax = plt.subplots()
ax.scatter(x, y, zorder = 2)
ax.grid(True, alpha = 0.5, linestyle = "--")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title("Sine of $x$ with noise")
plt.show()

# initialize the kernel and likelihood
se_kernel = gf.kernels.SquaredExponential()
gaus_likelihood = gf.likelihoods.Gaussian()

# remove the transforms from the variables
se_kernel.variance = gf.Parameter(value = 1, transform = None)
se_kernel.lengthscales = gf.Parameter(value = 1, transform = None)

gaus_likelihood.variance = gf.Parameter(value = 1, transform = None)

# initialize the model
se_model = gf.models.GPR(
    data = (x.reshape(100,1), y.reshape(100,1)),
    kernel = se_kernel,
    likelihood = gaus_likelihood
)

optimizer = gf.optimizers.Scipy()
optimizer.minimize(
    
    # a closure that re-evaluates the model, returning the loss to be minimized.
    closure = se_model.training_loss,
    
    # the list (tuple) of variables to be optimized
    variables = se_model.trainable_variables,
    
    method = "CG"
)

half_normal = tfd.HalfNormal(
    # default float is tf.float64
    scale = gf.utilities.to_default_float(30)
)

# confirm that float64 is generated
half_normal.sample(10)

fig, ax = plt.subplots()
ax.hist(half_normal.sample(10000), density = True, bins = 100, zorder = 2)
ax.grid(True, alpha = 0.5, linestyle = "--")
plt.show()

# initialize the kernel and likelihood
se_kernel = gf.kernels.SquaredExponential()
gaus_likelihood = gf.likelihoods.Gaussian()

# remove the transforms from the variables
se_kernel.variance = gf.Parameter(
    value = 1, 
    transform = None, 
    prior = half_normal, 
    prior_on = "unconstrained"
)

se_kernel.lengthscales = gf.Parameter(
    value = 1, 
    transform = None, 
    prior = half_normal, 
    prior_on = "unconstrained"
)

gaus_likelihood.variance = gf.Parameter(
    value = 1, 
    transform = None, 
    prior = half_normal, 
    prior_on = "unconstrained"
)

# initialize the model
se_model = gf.models.GPR(
    data = (x_tensor, y_tensor),
    kernel = se_kernel,
    likelihood = gaus_likelihood
)

optimizer.minimize(
    
    # a closure that re-evaluates the model, returning the loss to be minimized.
    closure = se_model.training_loss,
    
    # the list (tuple) of variables to be optimized
    variables = se_model.trainable_variables,
    
    method = "CG"
)

# initialize the kernel and likelihood
se_kernel = gf.kernels.SquaredExponential()
gaus_likelihood = gf.likelihoods.Gaussian()

# DO NOT REMOVE THE TRANSFORMS
# add the priors
se_kernel.variance.prior = half_normal
se_kernel.lengthscales.prior = half_normal

gaus_likelihood.variance.prior = half_normal

# initialize the model
se_model = gf.models.GPR(
    data = (x_tensor, y_tensor),
    kernel = se_kernel,
    likelihood = gaus_likelihood
)

optimizer.minimize(
    
    # a closure that re-evaluates the model, returning the loss to be minimized.
    closure = se_model.training_loss,
    
    # the list (tuple) of variables to be optimized
    variables = se_model.trainable_variables,
    
    method = "CG"
)

xx = np.linspace(0, 2 * np.pi, 100).reshape(100,1)
mean, var = se_model.predict_f(xx)

fig, ax = plt.subplots()
ax.scatter(x, y, zorder = 2)
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

# initialize the model with no constraints
se_model = gf.models.GPR(
    data = (x_tensor, y_tensor),
    kernel = gf.kernels.SquaredExponential(),
    likelihood = gf.likelihoods.Gaussian()
)

optimizer.minimize(
    
    # a closure that re-evaluates the model, returning the loss to be minimized.
    closure = se_model.training_loss,
    
    # the list (tuple) of variables to be optimized
    variables = se_model.trainable_variables,
    
    method = "CG"
)

xx = np.linspace(0, 2 * np.pi, 100).reshape(100,1)
mean, var = se_model.predict_f(xx)

fig, ax = plt.subplots()
ax.scatter(x, y, zorder = 2)
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

# initialize the model with variables that cannot be trained
se_kernel = gf.kernels.SquaredExponential()
se_kernel.variance = gf.Parameter(value = 0.1, trainable = False)
se_kernel.lengthscales = gf.Parameter(value = 0.1, trainable = False)

se_model = gf.models.GPR(
    data = (x_tensor, y_tensor),
    kernel = se_kernel,
    likelihood = gf.likelihoods.Gaussian()
)

optimizer.minimize(
    
    # a closure that re-evaluates the model, returning the loss to be minimized.
    closure = se_model.training_loss,
    
    # the list (tuple) of variables to be optimized
    variables = se_model.trainable_variables,
    
    method = "CG"
)

xx = np.linspace(0, 2 * np.pi, 100).reshape(100,1)
mean, var = se_model.predict_f(xx)

fig, ax = plt.subplots()
ax.scatter(x, y, zorder = 2)
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

m32_kernel = gf.kernels.Matern32()
m32_model = gf.models.GPR(
    data = (x_tensor, y_tensor),
    kernel = m32_kernel,
    likelihood = gf.likelihoods.Gaussian()
)

optimizer.minimize(
    
    # a closure that re-evaluates the model, returning the loss to be minimized.
    closure = m32_model.training_loss,
    
    # the list (tuple) of variables to be optimized
    variables = m32_model.trainable_variables,
    
    method = "CG"
)

xx = np.linspace(0, 2 * np.pi, 100).reshape(100,1)
mean, var = m32_model.predict_f(xx)

fig, ax = plt.subplots()
ax.scatter(x, y, zorder = 2)
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

m32_kernel = gf.kernels.Matern32()
m32_kernel.variance = gf.Parameter(value = 0.001, trainable = False)
m32_kernel.lengthscales = gf.Parameter(value = 1, trainable = True)

m32_model = gf.models.GPR(
    data = (x_tensor, y_tensor),
    kernel = m32_kernel,
    likelihood = gf.likelihoods.Gaussian()
)

optimizer.minimize(
    
    # a closure that re-evaluates the model, returning the loss to be minimized.
    closure = m32_model.training_loss,
    
    # the list (tuple) of variables to be optimized
    variables = m32_model.trainable_variables,
    
    method = "CG"
)

xx = np.linspace(0, 2 * np.pi, 100).reshape(100,1)
mean, var = m32_model.predict_f(xx)

fig, ax = plt.subplots()
ax.scatter(x, y, zorder = 2)
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

m32_kernel = gf.kernels.Matern32()

m32_kernel.variance = gf.Parameter(
    value = 1, 
    trainable = True, 
    transform = tfp.bijectors.Softplus()
)

m32_kernel.lengthscales = gf.Parameter(value = 0.001, trainable = False)

m32_model = gf.models.GPR(
    data = (x_tensor, y_tensor),
    kernel = m32_kernel,
    likelihood = gf.likelihoods.Gaussian()
)

optimizer.minimize(
    
    # a closure that re-evaluates the model, returning the loss to be minimized.
    closure = m32_model.training_loss,
    
    # the list (tuple) of variables to be optimized
    variables = m32_model.trainable_variables,
    
    method = "CG"
)

xx = np.linspace(0, 2 * np.pi, 100).reshape(100,1)
mean, var = m32_model.predict_f(xx)

fig, ax = plt.subplots()
ax.scatter(x, y, zorder = 2)
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

def squaredExponential(x, x_prime, signal_variance, length_scale):
    return signal_variance * np.exp( -( (x - x_prime)**2 / (2 * length_scale**2) ) )

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

fig, ax = plt.subplots(nrows = 1, ncols = 2,
                      figsize = (12, 4), 
                       sharey = True)

ax[0].plot(plot_vals, y1, label = "$\sigma_f^2=1$")
ax[0].plot(plot_vals, y2, label = "$\sigma_f^2=0.5$")
ax[0].set_title("Varying signal variance, $l=1$")
ax[0].legend()

ax[1].plot(plot_vals, y3, color = "orange", label = "$l=1$")
ax[1].plot(plot_vals, y4, color = "purple", label = "$l=0.3$")
ax[1].set_title("Varying length-scale, $\sigma^2_f=1$")
ax[1].legend()

plt.show()

# assuming some x values
x = np.linspace(start = -2, stop = 4, num = 9)

# create an empty covariance matrix
z = np.empty(shape = (len(x), len(x)))

# calculate each value of the covariance matrix elementwise
for i in range(len(x)):
    for j in range(len(x)):
        z[i, j] = squaredExponential(x[i], x[j], 1, 1,)

# print z
z

def squaredExponential(x, x_prime, signal_variance, length_scale):
    return signal_variance * np.exp( -( np.subtract.outer(x, x_prime)**2 / (2 * length_scale**2) ) )

covariance_matrix = squaredExponential(x, x, signal_variance=1, length_scale=1)

np.array_equal(z, covariance_matrix)

# define a multivariate normal as above
f = tfd.MultivariateNormalTriL(
    loc = np.zeros(len(x)), scale_tril=tf.linalg.cholesky(covariance_matrix)
)

# draw 10 samples from it
f_samples = f.sample(10)

# plot them
for i in range(f_samples.shape[0]):
    plt.plot(x, f_samples[i])

# assuming a much finer and longer x space
x = np.linspace(start = -5, stop = 5, num = 500)

# calculate the covariance matrix
covariance_matrix1 = squaredExponential(x, x, signal_variance=1, length_scale=1)
covariance_matrix2 = squaredExponential(x, x, signal_variance=0.3, length_scale=1)
covariance_matrix3 = squaredExponential(x, x, signal_variance=1, length_scale=0.1)

# define a multivariate normal
f1 = tfd.MultivariateNormalTriL(
    # mean vector of 0s
    loc = np.zeros(len(x)),

    # triangular matrix for Cholesky decomp
    # with jitter added for numerical stability at float64
    scale_tril=tf.linalg.cholesky(
        covariance_matrix1 + (tf.eye(len(x), dtype = tf.float64) * 1e-6)
    )
)

f2 = tfd.MultivariateNormalTriL(
    loc = np.zeros(len(x)), 
    scale_tril=tf.linalg.cholesky(
        covariance_matrix2+ (tf.eye(len(x), dtype = tf.float64) * 1e-6)
    )
)

f3 = tfd.MultivariateNormalTriL(
    loc = np.zeros(len(x)), 
    scale_tril=tf.linalg.cholesky(
        covariance_matrix3+ (tf.eye(len(x), dtype = tf.float64) * 1e-6)
    )
)

# draw 10 samples from it
f1 = f1.sample(10)
f2 = f2.sample(10)
f3 = f3.sample(10)

# plot them
fig, ax = plt.subplots(1, 3, figsize=(14,4), sharey=True)
for i in range(f1.shape[0]):
    ax[0].plot(x, f1[i])
    ax[1].plot(x, f2[i])
    ax[2].plot(x, f3[i])

# error bands are generated using the signal variance
# bands are 2 standard deviations * sqrt(signal variance)
ax[0].fill_between(
    x,
    0 - 1.96 * np.sqrt(1),
    0 + 1.96 * np.sqrt(1),
    alpha = 0.15,
    color = "grey"
)

ax[1].fill_between(
    x,
    0 - 1.96 * np.sqrt(0.3),
    0 + 1.96 * np.sqrt(0.3),
    alpha = 0.15,
    color = "grey"
)

ax[2].fill_between(
    x,
    0 - 1.96 * np.sqrt(1),
    0 + 1.96 * np.sqrt(1),
    alpha = 0.15,
    color = "grey"
)

ax[0].set_title("signal_variance = 1, length_scale = 1")
ax[1].set_title("signal_variance = 0.3, length_scale = 1")
ax[2].set_title("signal_variance = 1, length_scale = 0.1")
plt.show()

se_empty1 = gf.models.GPR(
    data = (np.zeros((10, 1)), np.zeros((10, 1))),
    kernel = gf.kernels.SquaredExponential(variance=1, lengthscales=1)
)

se_empty2 = gf.models.GPR(
    data = (np.zeros((10, 1)), np.zeros((10, 1))),
    kernel = gf.kernels.SquaredExponential(variance=0.3, lengthscales=1)
)

se_empty3 = gf.models.GPR(
    data = (np.zeros((10, 1)), np.zeros((10, 1))),
    kernel = gf.kernels.SquaredExponential(variance=1, lengthscales=0.1)
)

preds1 = se_empty1.predict_f_samples(x.reshape(len(x),1), 10)
preds2 = se_empty2.predict_f_samples(x.reshape(len(x),1), 10)
preds3 = se_empty3.predict_f_samples(x.reshape(len(x),1), 10)

# plot them
fig, ax = plt.subplots(1, 3, figsize=(14,4), sharey=True)
for i in range(preds1.shape[0]):
    ax[0].plot(x, preds1[i])
    ax[1].plot(x, preds2[i])
    ax[2].plot(x, preds3[i])

# error bands are generated using the signal variance
# bands are 2 standard deviations * sqrt(signal variance)
ax[0].fill_between(
    x,
    0 - 1.96 * np.sqrt(1),
    0 + 1.96 * np.sqrt(1),
    alpha = 0.15,
    color = "grey"
)

ax[1].fill_between(
    x,
    0 - 1.96 * np.sqrt(0.3),
    0 + 1.96 * np.sqrt(0.3),
    alpha = 0.15,
    color = "grey"
)

ax[2].fill_between(
    x,
    0 - 1.96 * np.sqrt(1),
    0 + 1.96 * np.sqrt(1),
    alpha = 0.15,
    color = "grey"
)

ax[0].set_title("signal_variance = 1, length_scale = 1")
ax[1].set_title("signal_variance = 0.3, length_scale = 1")
ax[2].set_title("signal_variance = 1, length_scale = 0.1")
plt.show()

plt.show()