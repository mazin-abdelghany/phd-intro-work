import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1.2 Group sequential design explanation
    """)
    return


@app.cell
def _(plt):
    _fig, _ax = plt.subplots(figsize=(9,2.5))

    # 5 stage design
    stages = [i+1 for i in range(5)]

    # upper and lower bound Z values
    upper_bounds = [5, 3, 2, 1.5, 1]
    lower_bounds = [-2, -1, 0, 0.5, 1]

    _ax.plot(stages, upper_bounds)
    _ax.plot(stages, lower_bounds)

    _ax.scatter(stages, upper_bounds, color="black", zorder=2, s=20)
    _ax.scatter(stages, lower_bounds, color="black", zorder=2, s=20)

    _ax.fill_between(stages, upper_bounds, lower_bounds, alpha=0.2, color="green")
    _ax.fill_between(stages, -6, lower_bounds, alpha=0.2, color="red")
    _ax.fill_between(stages, 8, upper_bounds, alpha=0.2, color="red")

    _ax.text(x=1.2, y=1, s="Continue trial", bbox=dict(facecolor='white'))
    _ax.text(x=2.5, y=5, s="Stop for efficacy", bbox=dict(facecolor='white'))
    _ax.text(x=2.5, y=-3.6, s="Stop for futility", bbox=dict(facecolor='white'))

    _ax.set_ylim(-6, 8)

    _ax.set_xticks([i+1 for i in range(5)])

    _ax.set_title("Group sequential bounds for a 5-stage design")
    _ax.set_xlabel("Analysis number, $k$")
    _ax.set_ylabel("Test statistic, $Z_k$")

    _fig.savefig("/tf/first_year_assessment/figures/gsd_example.png", dpi=300, bbox_inches="tight")

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1.6 GP regression figures
    """)
    return


@app.cell
def _():
    import tensorflow_probability as tfp
    import tensorflow as tf

    from scipy.stats import norm
    import matplotlib.image as mpimg

    tfd = tfp.distributions
    return norm, tf, tfd


@app.cell
def _(np, tf):
    seed = 940965957842
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)
    return


@app.cell
def _(np):
    def squaredExponential(x, x_prime, signal_variance, length_scale):
        return signal_variance * np.exp( -( np.subtract.outer(x, x_prime)**2 / (2 * length_scale**2) ) )

    return (squaredExponential,)


@app.cell
def _(np, squaredExponential, tf, tfd):
    x = np.linspace(start = -6, stop = 6, num = 500)
    jitter = (tf.eye(len(x), dtype = tf.float64) * 1e-6)

    # calculate the covariance matrix
    covariance_matrix1 = squaredExponential(x, x, signal_variance=1, length_scale=1/np.sqrt(2))
    covariance_matrix2 = squaredExponential(x, x, signal_variance=0.3, length_scale=1/np.sqrt(2))
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
    f1_draws = f1.sample(5)
    f2_draws = f2.sample(5)
    f3_draws = f3.sample(5)
    return f1, f1_draws, f2_draws, f3_draws, jitter, x


@app.cell
def _(f1_draws, f2_draws, f3_draws, np, plt, x):
    # plot them
    _fig, _ax = plt.subplots(1, 3, figsize=(14,4), sharey=True)
    for _i in range(f1_draws.shape[0]):
        _ax[0].plot(x, f1_draws[_i], linewidth=1)
        _ax[1].plot(x, f2_draws[_i], linewidth=1)
        _ax[2].plot(x, f3_draws[_i], linewidth=1)

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

    _ax[0].set_title("signal variance = 1, length scale = $\\frac{1}{\sqrt{2}}$")
    _ax[1].set_title("signal variance = 0.3, length scale = $\\frac{1}{\sqrt{2}}$")
    _ax[2].set_title("signal variance = 1, length scale = 0.1")

    _ax[0].set_ylabel("output, f(x)")

    # Save the figure before showing it
    _fig.savefig('/tf/first_year_assessment/figures/gp_func_priors.png', dpi=300, bbox_inches="tight")
    plt.gca()
    return


@app.cell
def _(np, plt, squaredExponential, tf, tfd, x):
    # conditioned Gaussian process figure
    # Define known input (x_observed) and output (y_observed) points
    x_observed = np.array([-5, -3.5, 3.])
    y_observed = np.array([1., -1., 0.5])
    noise_variance = 1e-8 # A very small noise for observations, assuming negligible measurement error

    # Use the same kernel parameters as the first subplot (signal variance = 1, length scale = 1)
    signal_variance_prior = 1.0
    length_scale_prior = 1.2

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
    _jitter = 1e-6
    Sigma_post_stable = Sigma_post + _jitter * np.eye(len(x))

    # Create a multivariate normal distribution for the posterior GP
    posterior_gp = tfd.MultivariateNormalTriL(
        loc = mu_post,
        scale_tril = tf.linalg.cholesky(Sigma_post_stable)
    )

    # Draw 10 samples (functions) from the posterior GP
    f_posterior_samples = posterior_gp.sample(10)

    # Plotting
    _fig, ax_gp = plt.subplots(figsize=(14,4)) 

    # Plot 9 of 10 the sampled functions from the posterior GP
    for _i in range(f_posterior_samples.shape[0]-1):
        ax_gp.plot(x, f_posterior_samples[_i], alpha=0.6, color='blue', linewidth=0.8)

    # plot the last sampled function to add the legend
    ax_gp.plot(x, f_posterior_samples[_i], alpha=0.6, color='blue', linewidth=0.8, label="Posterior function draws")

    # Plot the known data points
    ax_gp.plot(x_observed, y_observed, 'o', markersize=8, color='red', label='Known Points', zorder=3)

    # Plot the posterior mean function
    ax_gp.plot(x, mu_post, color='black', linestyle='--', linewidth=2, label='Posterior Mean')

    # Plot the error bands (2 standard deviations from the posterior mean)
    std_dev_post = np.sqrt(np.diag(Sigma_post))
    ax_gp.fill_between(
        x,
        mu_post - 2 * std_dev_post,
        mu_post + 2 * std_dev_post,
        alpha = 0.2,
        color = "grey",
        label = "95% Confidence Interval"
    )

    ax_gp.set_title("Gaussian Process Posterior and Samples") # New title for GP subplot
    ax_gp.set_ylabel("Output, f(x)")
    ax_gp.legend(loc='lower left')
    ax_gp.grid(True, linestyle='--', alpha=0.7)

    # Save the figure before showing it
    _fig.savefig('/tf/first_year_assessment/figures/conditioned_gp_samples.png', dpi=300, bbox_inches="tight")
    plt.gca()
    return length_scale_prior, noise_variance, signal_variance_prior


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1.7 Bayesian optimisation
    """)
    return


@app.cell
def _(np):
    def true_obj_func(x):
      return ((-1*x*np.sin(x)) + (x/3))/2

    return (true_obj_func,)


@app.cell
def _(f1, plt, true_obj_func, x):
    f1_for_plot = f1.sample(40)

    _fig, _ax = plt.subplots(figsize=(14,4))

    # plot the true objective function
    _ax.plot(x, true_obj_func(x), lw=2, color="black", linestyle='--', label="Objective function")

    # plot the samples from GP prior up to last
    for _i in range(f1_for_plot.shape[0]-1):
        _ax.plot(x, f1_for_plot[_i], color = "purple", lw=0.7, alpha=0.2, zorder=1)

    # plot the last GP prior sample to get a label
    _ax.plot(x, f1_for_plot[f1_for_plot.shape[0]-1], label = "Prior function draws", color = "purple", lw=0.7, alpha=0.2)

    _ax.set_ylim(-3.6, 3.6)
    _ax.set_xlim(-6, 6)
    _ax.grid(True, linestyle = ":", alpha = 0.5, axis = "y")

    _ax.set_xlabel('x')
    _ax.set_ylabel("f(x)")
    _ax.set_title("Gaussian process prior and objective function")
    _ax.legend(loc = "lower right")

    _fig.savefig("/tf/first_year_assessment/figures/gp_prior_and_objective_function.png", dpi=300, bbox_inches="tight")
    plt.gca()
    return


@app.cell
def _(
    jitter,
    length_scale_prior,
    noise_variance,
    norm,
    np,
    plt,
    signal_variance_prior,
    squaredExponential,
    tf,
    tfd,
    true_obj_func,
    x,
):
    _fig, _ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # observe 1 plot
    _ax[0].axvline(-5, color = "darkgreen")
    _ax[0].plot(-5, true_obj_func(-5), 'o', markersize=8, zorder = 3, color = "darkgreen", label="Randomly observed point")

    _ax[0].plot(x, true_obj_func(x), lw=2, color="black", linestyle='--', label="Objective function")

    _ax[0].set_xlim(-6, 6)
    _ax[0].set_ylim(-3.7, 3.7)
    _ax[0].grid(True, linestyle = ":", alpha = 0.5, axis = "y")

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
    f_samples = p_gp.sample(40)

    for sample in f_samples:
        _ax[0].plot(x, sample, color = "purple", alpha = 0.2, linewidth = 0.7, zorder = 1)

    # plot the last GP prior sample to get a label
    _ax[0].plot(x, f_samples[0], label = "Prior function draws", color = "purple", lw=0.7, alpha=0.2)

    _ax[0].set_xlabel('x')
    _ax[0].set_ylabel("f(x)")
    _ax[0].set_title("Gaussian process fit, objective function, and expected improvement")
    _ax[0].legend(loc = "lower right")

    # --- Acquisition Function Plot (Bottom Subplot) ---
    _std_dev_post = np.sqrt(np.diag(Sigma))

    # Calculate Expected Improvement (EI)
    _f_best = np.max(np.array([true_obj_func(-5)])) # Current best observed value

    # Avoid division by zero for std_dev_post where it might be extremely small or zero
    _std_dev_post_safe = np.where(_std_dev_post > 1e-10, _std_dev_post, 1e-10)

    _Z = (mu - _f_best) / _std_dev_post_safe

    _expected_improvement = np.where(
        _std_dev_post > 1e-10,
        (mu - _f_best) * norm.cdf(_Z) + _std_dev_post * norm.pdf(_Z),
        0.0
    )

    _ax[1].plot(x, _expected_improvement, color='darkorange', label='Expected improvement (EI)')
    _ax[1].set_xlabel("Input, x")
    _ax[1].set_ylabel("EI Value")
    _ax[1].set_xlim(-6, 6)
    _ax[1].grid(True, linestyle=':', alpha=0.5, axis = "y")
    _ax[1].legend(loc='lower right') # Adjust legend position for acquisition plot

    _fig.savefig("/tf/first_year_assessment/figures/gp_fit_w_acq_func.png", dpi=300, bbox_inches="tight")
    plt.gca()
    return


@app.cell
def _(np, plt):
    def plot_bo_iteration(x, true_obj_func, x_obs, y_obs, mu_post,
                          f_posterior_samples, expected_improvement,
                          next_x_to_observe, next_y_to_observe, iteration_count):
        """
        Generates a Matplotlib figure for a single Bayesian Optimization iteration.
        Returns the figure object.
        """
        _fig, (ax_gp, ax_acq) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

        for _i in range(f_posterior_samples.shape[0]):
            ax_gp.plot(x, f_posterior_samples[_i], color='purple', linewidth=0.7, alpha = 0.2)

        ax_gp.plot(x, mu_post, color='purple', linewidth=2, label='Posterior mean')
        ax_gp.plot(x, true_obj_func(x), lw=2, color="black", linestyle='--', label="Objective function")
        ax_gp.plot(x_obs, y_obs, 'o', markersize=8, color='darkgreen', label='Observed points')
        ax_gp.axvline(next_x_to_observe, color='darkgreen')
        # ax_gp.plot(next_x_to_observe, next_y_to_observe, 'o', markersize=8, color='green', label='New point')

        ax_gp.set_title(f"Bayes opt iteration {iteration_count}: Gaussian process fit, objective function, and expected improvement")
        ax_gp.set_ylim(-3.6, 3.6)
        ax_gp.set_xlim(-6, 6)
        ax_gp.grid(True, linestyle = ":", alpha = 0.5, axis = "y")
        ax_gp.legend(loc="lower right")

        # --- Acquisition Function Plot (Bottom Subplot) ---
        ax_acq.plot(x, expected_improvement, color='darkorange', label='Expected improvement')
        ax_acq.plot(next_x_to_observe, np.max(expected_improvement),
                    'o', markersize=8, color='darkgreen')
        ax_acq.axvline(next_x_to_observe, color='darkgreen', label='Max EI point')

        ax_acq.grid(True, linestyle = ":", alpha = 0.5, axis = "y")
        ax_acq.set_xlim(-6, 6)
        ax_acq.set_xlabel("Input, x")
        ax_acq.set_ylabel("EI value")
        ax_acq.legend(loc='lower right')

        return _fig

    return (plot_bo_iteration,)


@app.cell
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


@app.cell
def _(mo, num_iterations):
    iter = mo.ui.slider(0, num_iterations-1, 1, label="Iteration no.")
    iter
    return (iter,)


@app.cell
def _(iter, mo, np, x_obs, y_obs):
    mo.vstack(
        [
            f"--- Iteration no. {iter.value+1} ---", 
            f"Next x to observe: {np.round(x_obs[iter.value+1], 3)}",
            f"Value of f(x): {np.round(y_obs[iter.value+1],3)}"
        ]
    )
    return


@app.cell
def _(iter, plot_dict):
    plot_dict[iter.value]
    return


@app.cell
def _(plot_dict):
    save_fig = plot_dict[0]
    save_fig.savefig("/tf/first_year_assessment/figures/observe_new_point.png", dpi=300, bbox_inches="tight")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Loss function figures
    """)
    return


@app.cell
def _(np):
    alpha_range = np.linspace(0, 1, num=200)
    beta_range = np.linspace(0, 1, num=200)
    return alpha_range, beta_range


@app.cell
def _():
    target_alpha = 0.05
    target_beta = 0.1
    return (target_alpha,)


@app.cell
def _(alpha_range, target_alpha):
    loss_alpha = (alpha_range - target_alpha)**2
    return (loss_alpha,)


@app.cell
def _(alpha_range, loss_alpha, plt):
    fig, ax = plt.subplots(figsize=(9,2.5))

    ax.plot(alpha_range, loss_alpha, color = "darkorange", label="loss function")
    ax.set_title("Loss function with $\\alpha'$")
    ax.set_ylabel("Loss($\\alpha'$)")
    ax.set_xlabel("$\\alpha'$")
    ax.axvline(0.05, color = "purple", linewidth = 1, linestyle="--", label = "minimum")
    ax.legend()

    fig.savefig("/tf/first_year_assessment/figures/loss_alpha.png", dpi=300, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(alpha_range, beta_range, np):
    X, Y = np.meshgrid(alpha_range, beta_range)

    loss_ab = (X - 0.05)**2 + (Y-0.1)**2
    return X, Y, loss_ab


@app.cell
def _(X, Y, loss_ab, plt):
    _fig, _ax = plt.subplots(figsize=(10.3,2.5))

    # number of contour lines
    levels = 15

    # filled contours
    cf = _ax.contourf(X, Y, loss_ab, levels=levels, cmap="viridis")

    # contour lines
    cs = _ax.contour(X, Y, loss_ab, levels=levels, colors="black", linewidths=1.2)

    # colorbar
    _fig.colorbar(cf, ax=_ax, label="Loss($\\alpha',\\beta'$)")

    # minimum point
    _ax.plot(0.05, 0.1, marker="o", color="white", markersize=8)
    _ax.text(0.065, 0.081, " min", color="white")

    # labels
    _ax.set_title("Loss function with $\\alpha'$ and $\\beta'$")
    _ax.set_xlabel("$\\alpha'$")
    _ax.set_ylabel("$\\beta'$")

    _fig.savefig("/tf/first_year_assessment/figures/loss_alpha_beta.png", dpi=300, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reverse parameterisation exploration
    """)
    return


@app.cell
def _(np):
    def reverse_to_boundaries(params, K):
        params = np.asarray(params).flatten()
        c = params[0]

        delta_u = params[1::2][::-1]
        delta_l = params[2::2][::-1]

        upper_bounds = np.array([c + np.sum(delta_u[k:]) for k in range(K)])
        lower_bounds = np.array([c - np.sum(delta_l[k:]) for k in range(K)])

        return upper_bounds, lower_bounds

    return (reverse_to_boundaries,)


@app.cell
def _():
    #         c  u_4  l_4  u_3  l_3  u_2  l_2  u_1  l_1
    params = [3, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1]

    delta_u = params[1::2][::-1]
    delta_l = params[2::2][::-1]

    for k in range(5):
        print("k=",k+1)
        print("upper:")
        print(delta_u[k:])
        print(sum(delta_u[k:]))
        print(3+sum(delta_u[k:]))
        print("lower:")
        print(delta_l[k:])
        print("\n")
    return


@app.cell
def _(reverse_to_boundaries):
    reverse_to_boundaries(params=[3, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1], K=5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Classical boundaries and their losses

    Go to `/tf/first_year_assessment/classical_bounds_compare.py` for that figure
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MAMS design
    """)
    return


@app.cell
def _():
    mams_stages = [i+1 for i in range(3)]
    mams_lower = [-4, -0.5, 1.3]
    mams_upper = [5, 2.5, 1.3]
    return mams_lower, mams_stages, mams_upper


@app.cell
def _(mams_lower, mams_stages, mams_upper, plt):
    _fig, _ax = plt.subplots(nrows=1,ncols=4, figsize=(13,3), sharey=True)

    for t, axis in enumerate(_ax):

        axis.plot(mams_stages, mams_upper)
        axis.plot(mams_stages, mams_lower)
    
        axis.scatter(mams_stages, mams_upper, color="black", zorder=2, s=20)
        axis.scatter(mams_stages, mams_lower, color="black", zorder=2, s=20)
    
        axis.fill_between(mams_stages, mams_upper, mams_lower, alpha=0.2, color="green")
        axis.fill_between(mams_stages, -6, mams_lower, alpha=0.2, color="darkorange")
        axis.fill_between(mams_stages, 8, mams_upper, alpha=0.2, color="blue")

        if t == 0:
            axis.set_ylabel("Test statistic, $Z^{(k)}$")
            axis.text(x=1.2, y=1, s="Continue trial", bbox=dict(facecolor='white'))
            axis.text(x=1.7, y=4.5, s="Stop for efficacy", bbox=dict(facecolor='white'))
            axis.text(x=1.7, y=-3.6, s="Stop for futility", bbox=dict(facecolor='white'))
        
        axis.set_ylim(-5, 6)
        axis.set_title(f"Treatment {t+1}")
        axis.set_xticks([i+1 for i in range(3)])
        axis.set_xlabel("Analysis number, $j$")

    _ax[1].plot([1,2], [0, 1], color="black")
    _ax[1].scatter([1,2], [0, 1], color="red", zorder=3)
    _ax[1].text(x=1.7, y=-3.6, s="Group remains", bbox=dict(facecolor='white'))

    _ax[2].plot([1,2], [0, -1.5], color="black")
    _ax[2].scatter([1,2], [0, -1.5], color="red", zorder=3)
    _ax[2].text(x=1.7, y=-3.6, s="Group dropped", bbox=dict(facecolor='white'))

    _ax[3].plot([1,2], [0, 3.5], color="black")
    _ax[3].scatter([1,2], [0, 3.5], color="red", zorder=3)
    _ax[3].text(x=1.7, y=-3.6, s='"Winner"', bbox=dict(facecolor='white'))

    _fig.suptitle("MAMS design with 4 treatments and 3 stages", y=1.04)

    _fig.savefig("/tf/first_year_assessment/figures/mams_example.png", dpi=300, bbox_inches="tight")

    plt.show()
    return


if __name__ == "__main__":
    app.run()
