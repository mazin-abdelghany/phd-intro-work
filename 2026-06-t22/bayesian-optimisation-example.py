import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Setup
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import tensorflow_probability as tfp
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    import matplotlib.image as mpimg

    tfd = tfp.distributions
    return norm, np, plt, tf, tfd


@app.cell
def _(np):
    rng = np.random.default_rng(seed = 940965957842)
    return


@app.cell
def _(plt):
    plt.rcParams["figure.dpi"] = 300
    return


@app.cell
def _(np):
    def squaredExponential(x, x_prime, signal_variance, length_scale):
        return signal_variance * np.exp( -( np.subtract.outer(x, x_prime)**2 / (2 * length_scale**2) ) )

    return (squaredExponential,)


@app.cell
def _(np, plt):
    def plot_bo_iteration(x, true_obj_func, x_obs, y_obs, mu_post,
                          f_posterior_samples, expected_improvement,
                          next_x_to_observe, next_y_to_observe, iteration_count):
        """
        Generates a Matplotlib figure for a single Bayesian Optimization iteration.
        Returns the figure object.
        """
        _fig, (ax_gp, ax_acq) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        for _i in range(f_posterior_samples.shape[0]):
            ax_gp.plot(x, f_posterior_samples[_i], color='darkorange', linewidth=0.7, alpha = 0.5)

        ax_gp.plot(x, mu_post, color='purple', linewidth=2, label='Posterior mean')
        ax_gp.plot(x, true_obj_func(x), lw=2, color="black", linestyle='--', label="Objective function")
        ax_gp.plot(x_obs, y_obs, 'o', markersize=8, color='red', label='Observed points')
        ax_gp.axvline(next_x_to_observe, color='green')
        # ax_gp.plot(next_x_to_observe, next_y_to_observe, 'o', markersize=8, color='green', label='New point')

        ax_gp.set_title(f"Bayes opt iteration {iteration_count}: GP fit and objective function")
        ax_gp.set_ylim(-3.6, 3.6)
        ax_gp.set_xlim(-6, 6)
        ax_gp.grid(True, linestyle = ":", alpha = 0.5, axis = "y")
        ax_gp.legend(loc="lower right")

        # --- Acquisition Function Plot (Bottom Subplot) ---
        ax_acq.plot(x, expected_improvement, color='purple', label='Expected improvement')
        ax_acq.plot(next_x_to_observe, np.max(expected_improvement),
                    'o', markersize=8, color='green')
        ax_acq.axvline(next_x_to_observe, color='green', label='Max EI ppoint')

        ax_acq.set_title(f"Bayes opt iteration {iteration_count}: Expected improvement")
        ax_acq.grid(True, linestyle = ":", alpha = 0.5, axis = "y")
        ax_acq.set_xlim(-6, 6)
        ax_acq.set_xlabel("Input, x")
        ax_acq.set_ylabel("EI value")
        ax_acq.legend(loc='lower right')

        return _fig

    return (plot_bo_iteration,)


@app.cell(hide_code=True)
def _(np, squaredExponential, tf, tfd):
    x = np.linspace(start = -6, stop = 6, num = 500)
    jitter = (tf.eye(len(x), dtype = tf.float64) * 1e-6)

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
            covariance_matrix1 + jitter
        )
    )

    f2 = tfd.MultivariateNormalTriL(
        loc = np.zeros(len(x)),
        scale_tril=tf.linalg.cholesky(
            covariance_matrix2 + jitter
        )
    )

    f3 = tfd.MultivariateNormalTriL(
        loc = np.zeros(len(x)),
        scale_tril=tf.linalg.cholesky(
            covariance_matrix3 + jitter
        )
    )

    # draw 10 samples from it
    f1_draws = f1.sample(10)
    f2_draws = f2.sample(10)
    f3_draws = f3.sample(10)
    return f1, f1_draws, f2_draws, f3_draws, jitter, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GP priors and their hyperparameters
    """)
    return


@app.cell(hide_code=True)
def _(f1_draws, f2_draws, f3_draws, np, plt, x):
    # plot them
    _fig, _ax = plt.subplots(1, 3, figsize=(14,4), sharey=True)
    for _i in range(f1_draws.shape[0]):
        _ax[0].plot(x, f1_draws[_i])
        _ax[1].plot(x, f2_draws[_i])
        _ax[2].plot(x, f3_draws[_i])

    # error bands are generated using the signal variance
    # bands are 2 standard deviations * sqrt(signal variance)
    _ax[0].fill_between(
        x,
        0 - 1.96 * np.sqrt(1),
        0 + 1.96 * np.sqrt(1),
        alpha = 0.15,
        color = "grey"
    )

    _ax[1].fill_between(
        x,
        0 - 1.96 * np.sqrt(0.3),
        0 + 1.96 * np.sqrt(0.3),
        alpha = 0.15,
        color = "grey"
    )

    _ax[2].fill_between(
        x,
        0 - 1.96 * np.sqrt(1),
        0 + 1.96 * np.sqrt(1),
        alpha = 0.15,
        color = "grey"
    )

    for i in range(3):
        _ax[i].grid(True, linestyle=":", axis="y", alpha = 0.5)
        _ax[i].set_xlabel("input, x")
        _ax[i].set_xlim(-6, 6)

    _ax[0].set_title("signal variance = 1, length scale = 1")
    _ax[1].set_title("signal variance = 0.3, length scale = 1")
    _ax[2].set_title("signal variance = 1, length scale = 0.1")

    _ax[0].set_ylabel("output, f(x)")

    # Save the figure before showing it
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GP regression fitting
    """)
    return


@app.cell(hide_code=True)
def _(f1, plt, true_obj_func, x):
    f1_for_plot = f1.sample(40)

    fig, ax = plt.subplots(figsize=(14,4))

    # plot the samples from GP prior up to last
    for _i in range(f1_for_plot.shape[0]-1):
        ax.plot(x, f1_for_plot[_i], color = "darkorange", lw=0.7, alpha=0.5, zorder=1)

    # plot the last GP prior sample to get a label
    ax.plot(x, f1_for_plot[f1_for_plot.shape[0]-1], label = "Prior function draws", color = "darkorange", lw=0.7, alpha=0.5)

    # plot the true objective function
    ax.plot(x, true_obj_func(x), lw=2, color="black", linestyle='--', label="Objective function")

    ax.set_ylim(-3.6, 3.6)
    ax.set_xlim(-6, 6)
    ax.grid(True, linestyle = ":", alpha = 0.5, axis = "y")

    ax.set_xlabel('x')
    ax.set_ylabel("f(x)")
    ax.set_title("Gaussian process prior and objective function")
    ax.legend(loc = "lower right")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(
    jitter,
    length_scale_prior,
    noise_variance,
    np,
    plt,
    signal_variance_prior,
    squaredExponential,
    tf,
    tfd,
    true_obj_func,
    x,
):
    _fig, _ax = plt.subplots(figsize=(14,4))

    # observe 1 plot
    _ax.axvline(-5, color = "green")
    _ax.plot(-5, true_obj_func(-5), 'o', markersize=8, zorder = 3, color = "green")

    _ax.plot(x, true_obj_func(x), lw=2, color="black", linestyle='--', label="Objective function")

    _ax.set_xlim(-6, 6)
    _ax.set_ylim(-3.7, 3.7)
    _ax.grid(True, linestyle = ":", alpha = 0.5, axis = "y")

    # Calculate kernel matrices for the observed data and prediction points
    obs_obs = squaredExponential(np.array([-5]), np.array([-5]), signal_variance_prior, length_scale_prior)
    pred_obs = squaredExponential(x, np.array([-5]), signal_variance_prior, length_scale_prior)
    pred_pred = squaredExponential(x, x, signal_variance_prior, length_scale_prior)

    # Add noise variance to the observed kernel matrix for numerical stability
    obs_obs_noisy = obs_obs + noise_variance * np.eye(len(np.array([-5])))

    # Calculate the inverse of the noisy observed kernel matrix
    obs_obs_noisy_inv = np.linalg.inv(obs_obs_noisy)

    # Calculate the posterior mean
    mu = pred_obs @ obs_obs_noisy_inv @ np.array([true_obj_func(-5)])

    # Calculate the posterior covariance
    Sigma = pred_pred - pred_obs @ obs_obs_noisy_inv @ pred_obs.T

    # Add jitter for numerical stability to the posterior covariance matrix
    Sigma_stable = Sigma + jitter * np.eye(len(x))

    # Create a multivariate normal distribution for the posterior GP
    p_gp = tfd.MultivariateNormalTriL(
        loc = mu,
        scale_tril = tf.linalg.cholesky(Sigma_stable)
    )

    # Draw 10 samples (functions) from the posterior GP
    f_samples = p_gp.sample(30)

    for sample in f_samples:
        _ax.plot(x, sample, color = "darkorange", alpha = 0.5, linewidth = 0.7, zorder = 1)


    # plot the last GP prior sample to get a label
    _ax.plot(x, f_samples[0], label = "Prior function draws", color = "darkorange", lw=0.7, alpha=0.5)

    _ax.set_xlabel('x')
    _ax.set_ylabel("f(x)")
    _ax.set_title("Gaussian process prior and objective function")
    _ax.legend(loc = "lower right")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Acquisition function for optimisation
    """)
    return


@app.cell(hide_code=True)
def _(jitter, norm, np, plt, squaredExponential, tf, tfd, x):
    # Define known input (x_observed) and output (y_observed) points
    x_observed = np.array([-4, 3.])
    y_observed = np.array([2., -2.])
    noise_variance = 1e-8 # A very small noise for observations, assuming negligible measurement error

    # Use the same kernel parameters as the first subplot (signal variance = 1, length scale = 1)
    signal_variance_prior = 1.0
    length_scale_prior = 1.0

    # Calculate kernel matrices for the observed data and prediction points
    K_obs_obs = squaredExponential(x_observed, x_observed, signal_variance_prior, length_scale_prior)
    K_pred_obs = squaredExponential(x, x_observed, signal_variance_prior, length_scale_prior)
    K_pred_pred = squaredExponential(x, x, signal_variance_prior, length_scale_prior)

    # Add noise variance to the observed kernel matrix for numerical stability
    K_obs_obs_noisy = K_obs_obs + noise_variance * np.eye(len(x_observed))

    # Calculate the inverse of the noisy observed kernel matrix
    K_obs_obs_noisy_inv = np.linalg.inv(K_obs_obs_noisy)

    # Calculate the posterior mean
    mu_post = K_pred_obs @ K_obs_obs_noisy_inv @ y_observed

    # Calculate the posterior covariance
    Sigma_post = K_pred_pred - K_pred_obs @ K_obs_obs_noisy_inv @ K_pred_obs.T

    # Add jitter for numerical stability to the posterior covariance matrix
    Sigma_post_stable = Sigma_post + jitter * np.eye(len(x))

    # Create a multivariate normal distribution for the posterior GP
    posterior_gp = tfd.MultivariateNormalTriL(
        loc = mu_post,
        scale_tril = tf.linalg.cholesky(Sigma_post_stable)
    )

    # Draw 10 samples (functions) from the posterior GP
    f_posterior_samples = posterior_gp.sample(30)

    # Plotting
    _fig, (ax_gp, ax_acq) = plt.subplots(2, 1, figsize=(14,8), sharex=True) # Changed to 2 subplots, increased height

    # --- GP Plot (Top Subplot) ---
    # Plot the sampled functions from the posterior GP
    for _i in range(f_posterior_samples.shape[0]):
        ax_gp.plot(x, f_posterior_samples[_i], alpha=0.2, color='purple', linewidth=0.8)

    # Plot the known data points
    ax_gp.plot(x_observed, y_observed, 'o', markersize=8, color='darkblue', label='Known Points', zorder=3)

    # Plot the posterior mean function
    ax_gp.plot(x, mu_post, color='black', linestyle='--', linewidth=2, label='Posterior Mean')

    # Plot the error bands (2 standard deviations from the posterior mean)
    std_dev_post = np.sqrt(np.diag(Sigma_post))

    ax_gp.set_title("Gaussian Process Posterior and Samples") # New title for GP subplot
    ax_gp.set_ylabel("Output, f(x)")
    ax_gp.legend(loc='lower right')
    ax_gp.grid(True, linestyle=':', alpha=0.5, axis = "y")

    # --- Acquisition Function Plot (Bottom Subplot) ---
    # Calculate Expected Improvement (EI)
    f_best = np.max(y_observed) # Current best observed value

    # Avoid division by zero for std_dev_post where it might be extremely small or zero
    std_dev_post_safe = np.where(std_dev_post > 1e-10, std_dev_post, 1e-10)

    Z = (mu_post - f_best) / std_dev_post_safe

    expected_improvement = np.where(
        std_dev_post > 1e-10,
        (mu_post - f_best) * norm.cdf(Z) + std_dev_post * norm.pdf(Z),
        0.0
    )

    ax_acq.plot(x, expected_improvement, color='green', label='Expected Improvement')
    ax_acq.set_title("Expected Improvement Acquisition Function")
    ax_acq.set_xlabel("Input, x")
    ax_acq.set_ylabel("EI Value")
    ax_acq.set_xlim(-6, 6)
    ax_acq.grid(True, linestyle=':', alpha=0.5, axis = "y")
    ax_acq.legend(loc='lower right') # Adjust legend position for acquisition plot

    plt.gca()
    return length_scale_prior, noise_variance, signal_variance_prior


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bayesian optimisation example
    """)
    return


@app.cell(hide_code=True)
def _(np):
    def true_obj_func(x):
      return ((-1*x*np.sin(x)) + (x/3))/2

    return (true_obj_func,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Objective function
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$f(x)=\frac{\left(-x\sin x\ +\ \frac{x}{3}\right)}{2}$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayes opt loop
    """)
    return


@app.cell(hide_code=True)
def _(
    norm,
    np,
    plot_bo_iteration,
    squaredExponential,
    tf,
    tfd,
    true_obj_func,
    x,
):
    # --- Setup Hyperparameters and Initial Data ---
    x_obs = np.array([-5])
    y_obs = true_obj_func(x_obs)

    num_iterations = 10  # Set how many total iterations you want to run
    epsilon = 1e-6
    plot_dict = {}

    # Assuming x, x_obs, y_obs, squaredExponential, and true_obj_func 
    # are already defined in your environment before this block.

    for j in range(num_iterations):

        # 1. Calculate kernel matrices for current observed data and prediction points
        COV_obs_obs = squaredExponential(x_obs, x_obs, 1, 1)
        COV_pred_obs = squaredExponential(x, x_obs, 1, 1)
        COV_pred_pred = squaredExponential(x, x, 1, 1)

        # 2. Add noise variance for numerical stability
        COV_obs_obs_noisy = COV_obs_obs + epsilon * np.eye(len(x_obs))

        # 3. Calculate the inverse matrix
        COV_obs_obs_noisy_inv = np.linalg.inv(COV_obs_obs_noisy)

        # 4. Calculate posterior mean and covariance
        mu_posterior = COV_pred_obs @ COV_obs_obs_noisy_inv @ y_obs
        SIGMA_post = COV_pred_pred - COV_pred_obs @ COV_obs_obs_noisy_inv @ COV_pred_obs.T
        SIGMA_post_stable = SIGMA_post + epsilon * np.eye(len(x))

        # 5. Draw samples from the posterior GP
        post_gp = tfd.MultivariateNormalTriL(
            loc=mu_posterior,
            scale_tril=tf.linalg.cholesky(SIGMA_post_stable)
        )
        f_post_samples = post_gp.sample(40)

        # 6. Calculate Expected Improvement (EI)
        f_curr_best = np.max(y_obs)
        sd_post = np.sqrt(np.diag(SIGMA_post))
        sd_post_safe = np.where(sd_post > 1e-10, sd_post, 1e-10)

        Z_std = (mu_posterior - f_curr_best) / sd_post_safe
        EI = np.where(
            sd_post > 1e-10,
            (mu_posterior - f_curr_best) * norm.cdf(Z_std) + sd_post * norm.pdf(Z_std),
            0.0
        )

        # 7. Determine the next point to observe
        next_x_to_observe_idx = np.argmax(EI)
        next_x_to_observe = x[next_x_to_observe_idx]
        next_y_to_observe = true_obj_func(next_x_to_observe)

        # 8. Plot the current state
        # Pass 'i' or 'i+1' as the iteration number to your plotting function
        plot_dict[j] = plot_bo_iteration(
            x, true_obj_func, x_obs, y_obs, mu_posterior,
            f_post_samples, EI, next_x_to_observe, next_y_to_observe, j + 1
        )

        # 9. Append the new observation back into the dataset for the next round
        x_obs = np.append(x_obs, next_x_to_observe)
        y_obs = np.append(y_obs, next_y_to_observe)
    return num_iterations, plot_dict, x_obs, y_obs


@app.cell(hide_code=True)
def _(mo, num_iterations):
    iter = mo.ui.slider(0, num_iterations-1, 1, label="Iteration no.")
    iter
    return (iter,)


@app.cell(hide_code=True)
def _(iter, mo, np, x_obs, y_obs):
    mo.vstack(
        [
            f"--- Iteration no. {iter.value+1} ---", 
            f"Next x to observe: {np.round(x_obs[iter.value+1], 3)}",
            f"Value of f(x): {np.round(y_obs[iter.value+1],3)}"
        ]
    )
    return


@app.cell(hide_code=True)
def _(iter, plot_dict):
    plot_dict[iter.value]
    return


if __name__ == "__main__":
    app.run()
