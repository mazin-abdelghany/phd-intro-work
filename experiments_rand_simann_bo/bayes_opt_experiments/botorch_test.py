import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


@app.cell
def _():
    import time
    import gc
    import scipy.stats as stats
    from scipy.stats import qmc
    import numpy as np
    import pandas as pd
    import torch

    return gc, np, pd, qmc, stats, time, torch


@app.cell
def _():
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import ExpectedImprovement
    from botorch.optim import optimize_acqf
    import warnings

    return (
        ExactMarginalLogLikelihood,
        ExpectedImprovement,
        SingleTaskGP,
        fit_gpytorch_mll,
        optimize_acqf,
        warnings,
    )


@app.cell
def _():
    # group sequential design assessment imports
    from py_group_sequential_designs import generate_boundaries as bd
    from py_group_sequential_designs import feasibility_penalty as fp
    from py_group_sequential_designs import boundary_manipulations as fmt_bd
    from py_group_sequential_designs import function_to_minimize as fn_min
    from py_group_sequential_designs import generate_gpr_input as gen_input
    from py_group_sequential_designs import simulate as sim
    from py_group_sequential_designs import sample_size as ss

    return bd, fmt_bd, fn_min, fp, sim, ss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Trial Design Settings
    """)
    return


@app.cell
def _(mo):
    num_analyses = mo.ui.number(label="Number of analyses = ", value=5, start=1)

    mo.vstack([num_analyses])
    return (num_analyses,)


@app.cell
def _(num_analyses, ss):
    target_alpha = 0.05
    target_power = 0.9
    delta0 = 0.
    delta1 = 1.
    sigma2 = 9.

    mu = ss.sample_size_means(
        ratio=1,
        variance=sigma2,
        power=target_power,
        alpha=target_alpha,
        delta=delta1
    )

    print(f"We are running an experiment with a trial design with {num_analyses.value} stages, with:\na target alpha of {target_alpha},\na target power of {target_power},\na null hypothesis of {delta0},\nan alternative hypothesis of {delta1},\nand an assumed variance of {sigma2}\n")
    print(f"Single-stage sample size mu = {mu:.2f}")
    return delta0, delta1, mu, sigma2, target_alpha, target_power


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reverse parameterisation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    BO vector: $\{c, \Delta u_3, \Delta \ell_3, \Delta u_2, \Delta \ell_2\}$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Objective function
    """)
    return


@app.cell
def _(fn_min, fp, sim, ss):
    def obj_f(
            mu,
            upper_bounds,
            lower_bounds,
            n_patients,
            n_analyses,
            target_power,
            target_alpha,
            null_hypothesis,
            alternative_hypothesis,
            variance):

        trial_sim = sim.group_sequential_designs(
            n_analyses = n_analyses,
            upper_bounds = upper_bounds,
            lower_bounds = lower_bounds,
            n_patients = n_patients, 
            null_hypothesis = null_hypothesis,
            alt_hypothesis = alternative_hypothesis,
            variance = variance
        )

        alpha_prime = trial_sim[0]
        beta_prime = 1-trial_sim[1]

        max_ess = ss.max_ess(
            n_analyses = n_analyses,
            upper_bounds = upper_bounds,
            lower_bounds = lower_bounds,
            n_patients = n_patients,
            null_hypothesis = null_hypothesis,
            variance = variance
        )

        penalty = fp.smooth_penalty(
            mu = mu,
            power = target_power,
            alpha = target_alpha,
            beta_prime = beta_prime,
            alpha_prime = alpha_prime
        )

        f_val = fn_min.function_to_minimize(max_ess_val = max_ess/mu, penalty = penalty)

        return (
            alpha_prime,
            1-beta_prime,
            max_ess,
            f_val
        )

    return (obj_f,)


@app.cell
def _(
    bd,
    delta0,
    delta1,
    fmt_bd,
    mu,
    np,
    num_analyses,
    obj_f,
    sigma2,
    ss,
    target_alpha,
    target_power,
):
    tri = bd.calculate_triangular_boundaries(
        n_analyses = num_analyses.value,
        alpha = target_alpha,
        delta = delta1
    )

    tri_n_patients = ss.find_sample_size(
        power_target = target_power,
        n_analyses = num_analyses.value,
        upper_bounds = tri[0],
        lower_bounds = tri[1],
        null_hypothesis = delta0,
        alt_hypothesis = delta1,
        variance = sigma2
    )[0]

    tri_alpha, tri_power, tri_max_ess, tri_obj = obj_f(
        mu = mu,
        upper_bounds = tri[0],
        lower_bounds = tri[1],
        n_analyses = num_analyses.value,
        n_patients = tri_n_patients,
        target_power = target_power,
        target_alpha = target_alpha,
        null_hypothesis = delta0,
        alternative_hypothesis = delta1,
        variance = sigma2
    )

    tri_params = fmt_bd.boundaries_to_reverse(
        upper_bounds = tri[0],
        lower_bounds = tri[1]
    )

    c0 = tri_params[0]

    print(f"Original triangular params: {np.round(np.concatenate((tri[0], tri[1])), 4)}")
    print(f"Reparameterized triangular params: {np.round(tri_params, 4)}")
    print(f"Meeting point c0 = {c0:.4f}\n")
    print(f"Triangular benchmark objective: {tri_obj:.4f}")
    print(f"Triangular alpha: {tri_alpha:.4f}")
    print(f"Triangular delta alpha: {abs(0.05-tri_alpha):.4f}")
    print(f"Triangular power: {tri_power:.4f}")
    print(f"Triangular delta beta: {abs(0.9-tri_power):.4f}")
    print(f"Triangular sample size: {tri_n_patients:.1f}")
    print(f"Triangular max ESS: {tri_max_ess:.1f}")
    return c0, tri_params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Quantities to follow
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data structure
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Experimental setup
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Search space
    """)
    return


@app.cell
def _(c0, mo, np, num_analyses, tri_params):
    search_space_boxes = ['large_box', 'large_box_5_stages', 'small_box', 'triang_box']

    space_dropdown = mo.ui.dropdown(
        options=search_space_boxes,
        value="large_box",
        label="Choose search space:"
    )

    lower_spaces = {}
    upper_spaces = {}

    for key in search_space_boxes:
        if key == "triang_box":
            lower_spaces[key] = np.array([max(0, p - 0.4) for p in tri_params] + [20])
            upper_spaces[key] = np.array([p + 0.4 for p in tri_params] + [160])
            continue

        n = num_analyses.value * 2
        lower = np.zeros(n)
        upper = np.ones(n)

        if key == "large_box":
            upper = upper * 4
            lower[0] = c0 - 3.0
            upper[0] = c0 + 3.0
        elif key == "large_box_5_stages":
            upper = upper * 2
            lower[0] = c0 - 2.0
            upper[0] = c0 + 2.0
            upper[2] = 4
        elif key == "small_box":
            lower[0] = c0 - 1.0
            upper[0] = c0 + 1.0
            upper[2] = 4.0

        if key == "large_box":
            lower[-1] = 20
            upper[-1] = 160
        elif key == "large_box_5_stages":
            lower[-1] = 20
            upper[-1] = 60

        lower_spaces[key] = lower
        upper_spaces[key] = upper
    return lower_spaces, space_dropdown, upper_spaces


@app.cell
def _(lower_spaces, mo, np, space_dropdown, upper_spaces):
    current_lower = lower_spaces[space_dropdown.value]
    current_upper = upper_spaces[space_dropdown.value]

    mo.vstack(
        [
            space_dropdown, 
            mo.md(f"**Lower value:** {np.round(current_lower, decimals=3)}"),
            mo.md(f"**Upper value:** {np.round(current_upper, decimals=3)}")
        ]
    )
    return current_lower, current_upper


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data collection setup
    """)
    return


@app.cell
def _():
    n_experiments = 50
    n_loops = 500
    return n_experiments, n_loops


@app.cell
def _(n_loops):
    space_needed_for_label = len(str(n_loops))
    label_range = 10**(space_needed_for_label)
    return (label_range,)


@app.cell
def _(num_analyses):
    upper_labels = [f"upper{i+1}" for i in range(num_analyses.value)]
    lower_labels = [f"lower{i+1}" for i in range(num_analyses.value - 1)]

    labels = upper_labels + lower_labels

    ordered_keys = ["index"] + labels + [
        "alpha", "power", "sample_size", "max_ess", 
        "obj_func", "execute_time", "seed"
    ]

    bayes_opt_results = {key: [] for key in ordered_keys}
    return bayes_opt_results, labels


@app.cell
def _(np):
    rng = np.random.default_rng(seed = 437591)
    return (rng,)


@app.cell
def _(bayes_opt_results, n_experiments, n_loops, np, rng):
    seed_list = [] 
    short_seed_list = []

    for _ in range(n_experiments):
        seed = int(np.round(rng.uniform(0, 2**32 - 1)))

        if seed in short_seed_list:
            seed = int(np.round(rng.uniform(0, 2**32 - 1)))

        short_seed_list.append(seed)

        seeds = np.repeat(seed, n_loops)
        seed_list += seeds.tolist()

    bayes_opt_results["seed"] = seed_list
    return (short_seed_list,)


@app.cell
def _(bayes_opt_results, label_range, n_experiments, n_loops):
    index_list = [
        i 
        for start in range(label_range, (n_experiments+1)*label_range, label_range)
        for i in range(start + 1, start + (n_loops+1))
    ]

    bayes_opt_results["index"] = index_list
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Experiment initiation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ============
    ## Bayes opt setup
    """)
    return


@app.cell
def _(mo):
    scale_input = mo.ui.switch(label="Min-max scale inputs")
    return (scale_input,)


@app.cell
def _(mo, scale_input):
    mo.vstack([scale_input, mo.md(f"Has value: {scale_input.value}")])
    return


@app.cell
def _(mo):
    scale_output = mo.ui.switch(label="Z-scale outputs")
    return (scale_output,)


@app.cell
def _(mo, scale_output):
    mo.vstack([scale_output, mo.md(f"Has value: {scale_output.value}")])
    return


@app.cell
def _(mo):
    num_haltons = mo.ui.number(label="Number of Halton points = ", value=500, start=100, stop=500, step=100)

    mo.vstack([num_haltons])
    return (num_haltons,)


@app.cell
def _(mo):
    do_not_train_error = mo.ui.switch(label="Do not train error")
    return (do_not_train_error,)


@app.cell
def _(do_not_train_error, mo):
    mo.vstack([do_not_train_error, mo.md(f"Has value: {do_not_train_error.value}")])
    return


@app.cell
def _(do_not_train_error, mo):
    if do_not_train_error.value:
        radio = mo.ui.radio(
            options={
                "1e-1": 1e-1,
                "1e-2": 1e-2,
                "1e-3": 1e-3,
                "1e-4": 1e-4,
                "1e-5": 1e-5,
            },
            value="1e-3",
            label="Likelihood variance",
        )
        radio
    else:
        radio = mo.ui.radio(
            options={"1" : 1},
            label="Initial value for likelihood variance",
        )
    radio
    return (radio,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## End Bayes opt setup
    ## ============
    """)
    return


@app.cell
def _(torch):
    def min_max_scale(x, x_min, x_max):
        """Transform values to [0, 1]. Works on Tensors."""
        return (x - x_min) / (x_max - x_min)

    def min_max_unscale(x_scaled, x_min, x_max):
        """Transform normalized [0, 1] values back. Works on Tensors."""
        return x_scaled * (x_max - x_min) + x_min

    def z_score_scale(x, mu_x, sigma_x):
        """Standardize input array using Z-score scale. Works on Tensors."""
        return (x - mu_x) / sigma_x
        
    return min_max_scale, min_max_unscale, z_score_scale


@app.cell
def _(
    ExactMarginalLogLikelihood,
    ExpectedImprovement,
    SingleTaskGP,
    bayes_opt_results,
    current_lower,
    current_upper,
    delta0,
    delta1,
    do_not_train_error,
    fit_gpytorch_mll,
    fmt_bd,
    gc,
    labels,
    min_max_unscale,
    mu,
    n_experiments,
    n_loops,
    np,
    num_analyses,
    num_haltons,
    obj_f,
    optimize_acqf,
    qmc,
    radio,
    scale_input,
    scale_output,
    short_seed_list,
    sigma2,
    target_alpha,
    target_power,
    time,
    torch,
    warnings,
    z_score_scale,
):
    # BoTorch uses torch.float64 by default for stability
    t_dtype = torch.float64
    
    lower_t = torch.tensor(current_lower, dtype=t_dtype)
    upper_t = torch.tensor(current_upper, dtype=t_dtype)
    
    # Suppress BoTorch optimization warnings for cleaner output
    warnings.filterwarnings("ignore", module="botorch")

    for i in range(n_experiments):

        # Set seeds
        np.random.seed(short_seed_list[i])
        torch.manual_seed(short_seed_list[i])

        ########################
        # Halton initialisation #
        ########################
        sampler = qmc.Halton(d=len(current_lower), seed=short_seed_list[i])
        # Halton points are generated in [0,1]^D
        initial_x_halton = torch.tensor(sampler.random(n=num_haltons.value), dtype=t_dtype)

        if scale_input.value:
            # The GP will train on [0,1] directly.
            # We must unscale only to evaluate the objective function.
            initial_x = initial_x_halton
            initial_points = min_max_unscale(initial_x_halton, lower_t, upper_t)
        else:
            # The GP will train on the raw, true bounds.
            # We must unscale the [0,1] Halton points to become our GP training data.
            initial_x = min_max_unscale(initial_x_halton, lower_t, upper_t)
            initial_points = initial_x

        initial_y = []

        # Evaluate initial points
        for point in initial_points:
            sample_size = point[(num_analyses.value*2)-1].item()
            bounds_raw = point[:-1].numpy()

            bounds = fmt_bd.reverse_to_boundaries(params = bounds_raw, K = num_analyses.value)

            _, _, _, initial_y_new = obj_f(
                mu = mu,
                upper_bounds = bounds[0],
                lower_bounds = bounds[1],
                n_analyses = num_analyses.value,
                n_patients = sample_size,
                target_power = target_power,
                target_alpha = target_alpha,
                null_hypothesis = delta0,
                alternative_hypothesis = delta1,
                variance = sigma2
            )

            initial_y.append(initial_y_new)

        # Track our raw training points
        train_X = initial_x
        
        # Note: BoTorch's ExpectedImprovement MAXIMIZES by default.
        # Since we want to MINIMIZE obj_f, we train the GP on the negative values.
        train_Y_raw = -torch.tensor(initial_y, dtype=t_dtype).unsqueeze(1)


        ############################
        # Start the bayes opt loop #
        ############################
        start_time = time.time()

        for j in range(n_loops):
            
            # 1. Apply Output Scaling (if enabled)
            if scale_output.value:
                y_mu = train_Y_raw.mean()
                y_sigma = train_Y_raw.std()
                train_Y = z_score_scale(train_Y_raw, y_mu, y_sigma)
            else:
                train_Y = train_Y_raw

            # 2. Build and Train GP
            model = SingleTaskGP(train_X, train_Y)
            
            if do_not_train_error.value:
                # Fix the noise variance in the Gaussian Likelihood
                model.likelihood.noise_covar.raw_noise.requires_grad_(False)
                model.likelihood.noise_covar.noise = torch.tensor([float(radio.value)], dtype=t_dtype)
            
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # 3. Define Acquisition Function
            EI = ExpectedImprovement(
                model=model, 
                best_f=train_Y.max(), # Since we negated Y, maximizing this finds the min objective
                maximize=True
            )

            # 4. Optimize Acquisition Function to find next point
            if scale_input.value:
                acqf_bounds = torch.stack([torch.zeros(len(lower_t), dtype=t_dtype), 
                                           torch.ones(len(upper_t), dtype=t_dtype)])
            else:
                acqf_bounds = torch.stack([lower_t, upper_t])

            candidates, _ = optimize_acqf(
                acq_function=EI,
                bounds=acqf_bounds,
                q=1,
                num_restarts=10, 
                raw_samples=512, 
            )
            
            x_new = candidates[0] # The proposed point in GP space

            # 5. Evaluate Objective Function
            if scale_input.value:
                x_new_eval = min_max_unscale(x_new, lower_t, upper_t)
            else:
                x_new_eval = x_new
                
            x_new_sample_size = x_new_eval[(num_analyses.value*2)-1].item()
            x_new_bounds_raw = x_new_eval[:-1].numpy()

            bounds = fmt_bd.reverse_to_boundaries(params = x_new_bounds_raw, K = num_analyses.value)
            bounds_list = np.concatenate((bounds[0], bounds[1][0:num_analyses.value-1]))

            alpha, power, max_ess, y_new = obj_f(
                mu = mu,
                upper_bounds = bounds[0],
                lower_bounds = bounds[1],
                n_analyses = num_analyses.value,
                n_patients = x_new_sample_size,
                target_power = target_power,
                target_alpha = target_alpha,
                null_hypothesis = delta0,
                alternative_hypothesis = delta1,
                variance = sigma2
            )
            
            # 6. Store Data and Update Tensors
            for _i in range(len(bounds_list)):
                bayes_opt_results[labels[_i]].append(bounds_list[_i])

            bayes_opt_results["alpha"].append(alpha)
            bayes_opt_results["power"].append(power)
            bayes_opt_results["sample_size"].append(x_new_sample_size)
            bayes_opt_results["max_ess"].append(max_ess)
            bayes_opt_results["obj_func"].append(y_new)
            
            # Append to BO tracking tensors
            train_X = torch.cat([train_X, x_new.unsqueeze(0)])
            y_new_tensor = -torch.tensor([[y_new]], dtype=t_dtype) # Negated for maximization
            train_Y_raw = torch.cat([train_Y_raw, y_new_tensor])

            if j % 25 == 0:
                print(".", end="")

        stop_time = time.time()
        execute_time = stop_time - start_time
        bayes_opt_results["execute_time"].extend([execute_time] * n_loops)

        if i % 1 == 0:
            print("\n===========================")
            print(f"= Completed experiment {i+1}. =")
            print("===========================")

        del model
        del mll
        gc.collect()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Saving data
    """)
    return


@app.cell
def _(bayes_opt_results, pd):
    pd.DataFrame(bayes_opt_results)
    return


@app.cell
def _(
    do_not_train_error,
    radio,
    n_experiments,
    n_loops,
    num_haltons,
    scale_input,
    scale_output,
):
    file_name = "bo_smooth_botorch"

    file_name += "_" + str(n_experiments) + "x" + str(n_loops)

    if scale_input.value:
        file_name += "_x_min_max"
    if scale_output.value:
        file_name += "_y_z_scaled"
    if do_not_train_error.value:
        # Uses the radio selection if noise is fixed
        file_name += "_err_" + str(radio.value)

    file_name += "_" + str(num_haltons.value) + "haltons"

    file_name += ".csv"
    return (file_name,)


@app.cell
def _(file_name):
    file_name
    return


@app.cell
def _(file_name):
    path = "/workspace/experiments_rand_simann_bo/bayes_opt_experiments/" + file_name
    return (path,)


@app.cell
def _(bayes_opt_results, path, pd):
    pd.DataFrame(bayes_opt_results).to_csv(path_or_buf=path, index=False)
    return


if __name__ == "__main__":
    app.run()
