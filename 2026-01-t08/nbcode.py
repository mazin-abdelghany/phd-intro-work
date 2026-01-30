import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import trieste
import gpflow

from trieste.objectives import ScaledBranin
from trieste.experimental.plotting import plot_function_plotly
from trieste.space import Box

from trieste.models.gpflow import GaussianProcessRegression

np.random.seed(1793)
tf.random.set_seed(1793)

# import included objective function
scaled_branin = ScaledBranin.objective

# predefined search space
search_space = ScaledBranin.search_space  

# define the search space directly
search_space = Box([0, 0], [1, 1])  

# plot the function using trieste built-in plotting
fig = plot_function_plotly(
    scaled_branin,
    search_space.lower,
    search_space.upper,
    grid_density=20,
)
fig.update_layout(height=800, width=1000)
fig.show()

# create a function that observes the objective at certain points
observer = trieste.objectives.utils.mk_observer(scaled_branin)

# sample 5 points using Sobol design
num_initial_points = 5
initial_query_points = search_space.sample_sobol(num_initial_points)
initial_data = observer(initial_query_points)

from trieste.models.gpflow import GaussianProcessRegression

def build_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[0.2, 0.2])
    prior_scale = tf.cast(1.0, dtype=tf.float64)
    kernel.variance.prior = tfp.distributions.LogNormal(
        tf.cast(-2.0, dtype=tf.float64), prior_scale
    )
    kernel.lengthscales.prior = tfp.distributions.LogNormal(
        tf.math.log(kernel.lengthscales), prior_scale
    )
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)

    return GaussianProcessRegression(gpr, num_kernel_samples=100)


model = build_model(initial_data)

from trieste.models.gpflow import build_gpr

gpflow_model = build_gpr(initial_data, search_space, likelihood_variance=1e-7)
model = GaussianProcessRegression(gpflow_model, num_kernel_samples=100)

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

num_steps = 15
result = bo.optimize(num_steps, initial_data, model)
dataset = result.try_get_final_dataset()

query_point, observation, arg_min_idx = result.try_get_optimal_point()

print(f"query point: {query_point}")
print(f"observation: {observation}")

from trieste.experimental.plotting import plot_bo_points, plot_function_2d

query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

_, ax = plot_function_2d(
    scaled_branin,
    search_space.lower,
    search_space.upper,
    grid_density=30,
    contour=True,
)
plot_bo_points(query_points, ax[0, 0], num_initial_points, arg_min_idx)
ax[0, 0].set_xlabel(r"$x_1$")
ax[0, 0].set_xlabel(r"$x_2$")

import matplotlib.pyplot as plt
from trieste.experimental.plotting import plot_regret

suboptimality = observations - ScaledBranin.minimum.numpy()
_, ax = plt.subplots(1, 2)
plot_regret(
    suboptimality, ax[0], num_init=num_initial_points, idx_best=arg_min_idx
)
plot_bo_points(
    query_points, ax[1], num_init=num_initial_points, idx_best=arg_min_idx
)

ax[0].set_yscale("log")
ax[0].set_ylabel("Regret")
ax[0].set_ylim(0.001, 100)
ax[0].set_xlabel("# evaluations")