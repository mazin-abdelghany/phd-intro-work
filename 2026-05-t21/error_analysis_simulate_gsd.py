import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # group sequential design assessment imports
    from py_group_sequential_designs import simulate as sim

    return (sim,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Trapezoidal
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import math
    from numba import njit

    #@njit
    def _numba_grid_engine(
        n_analyses, 
        upper_bounds, 
        lower_bounds, 
        n_patients_analysis, 
        means, 
        grid_points
    ):
        fut_probs = np.zeros(n_analyses)
        eff_probs = np.zeros(n_analyses)

        ####################
        # helper functions #
        ####################

        # standard normal pdf (mean=0, std=1)
        def std_norm_pdf(x):
            return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)

        # standard normal cdf (mean=0, std=1)
        def std_norm_cdf(x):
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

        ##############
        # analysis 1 #
        ##############

        # for the first stage, compute the probabilities of stopping
        # for futility and efficacy; bound - means standardizes the problem
        fut_probs[0] = std_norm_cdf(lower_bounds[0] - means[0])
        eff_probs[0] = 1.0 - std_norm_cdf(upper_bounds[0] - means[0])


        if lower_bounds[0] > upper_bounds[0]:
            print("Error: Upper bounds are lower than lower bounds.")
            return

        # z_grid is selecting a number of grid points and splitting up
        # the continuation region (ell < cont_region < u)
        z_grid = np.linspace(lower_bounds[0], upper_bounds[0], grid_points)

        # fill in the probability densities (PDF NOT CDF!) for each of
        # the selected points in the continuation region; again
        # z_grid - means standardizes
        density = np.zeros(grid_points)
        for i in range(grid_points):
            density[i] = std_norm_pdf(z_grid[i] - means[0])


        ###################
        # analyses 2 to K #
        ###################


        for k in range(1, n_analyses):

            I_prev = n_patients_analysis[k-1]
            I_curr = n_patients_analysis[k]

            # this is the covariance between the previous 
            # and current stages Cov(z_prev, z_curr)
            cov_pre_curr = np.sqrt(I_prev / I_curr)

            # the conditional standard deviation
            cond_scale = np.sqrt(1.0 - cov_pre_curr**2)

            # the conditional means at each grid point (vector of length
            # grid points)
            cond_means = means[k] + cov_pre_curr * (z_grid - means[k-1])

            fut_arg = np.zeros(grid_points)
            eff_arg = np.zeros(grid_points)
            for i in range(grid_points):
                fut_arg[i] = std_norm_cdf((lower_bounds[k] - cond_means[i]) / cond_scale) * density[i]
                eff_arg[i] = (1.0 - std_norm_cdf((upper_bounds[k] - cond_means[i]) / cond_scale)) * density[i]

            # the width of the steps along the grid as calculated above
            dz = (upper_bounds[k-1] - lower_bounds[k-1]) / (grid_points - 1)

            # Composite Trapezoidal Integration rule
            fut_probs[k] = (np.sum(fut_arg) - 0.5 * (fut_arg[0] + fut_arg[-1])) * dz
            eff_probs[k] = (np.sum(eff_arg) - 0.5 * (eff_arg[0] + eff_arg[-1])) * dz

            # Grid transition to next stage
            if k < n_analyses - 1:
                next_lower, next_upper = lower_bounds[k], upper_bounds[k]

                if next_lower > next_upper:
                    print("Error: Upper bounds are lower than lower bounds.")
                    return

                next_z_grid = np.linspace(next_lower, next_upper, grid_points)

                # the recursive density profile is defined by the 
                # Chapman-Kolmogorov-style integral
                # transition density matrix (transition_pdf) 
                # mapping a state y at stage k−1 to a state z at stage k
                transition_pdf = np.zeros((grid_points, grid_points))
                for r in range(grid_points):
                    for c in range(grid_points):
                        diff = (next_z_grid[r] - cond_means[c]) / cond_scale
                        transition_pdf[r, c] = std_norm_pdf(diff) / cond_scale

                next_matrix = transition_pdf * density
                density = (np.sum(next_matrix, axis=1) - 0.5 * (next_matrix[:, 0] + next_matrix[:, -1])) * dz
                z_grid = next_z_grid


        return fut_probs, eff_probs


    # high-level Python wrapper
    def group_sequential_designs(
        n_analyses=3,
        upper_bounds=[2.5, 2, 1.5],
        lower_bounds=[0, 0.75, 1.5],
        n_patients=20,
        null_hypothesis=0,
        alt_hypothesis=0.5,
        variance=1,
        return_table=False,
        grid_points=100
    ):

        upper_bounds = np.asarray(upper_bounds, dtype=float)
        lower_bounds = np.asarray(lower_bounds, dtype=float)
        n_patients_analysis = n_patients * np.arange(1, n_analyses + 1)

        # standardize the null and alternative hypotheses
        # sqrt(information) * theta is the Wald statistic at each stage (Z_j)
        scale = np.sqrt(n_patients_analysis / (2 * variance))
        mean_0 = null_hypothesis * scale
        mean_1 = alt_hypothesis * scale

        futility_null, efficacy_null = _numba_grid_engine(
            n_analyses, upper_bounds, lower_bounds, n_patients_analysis, mean_0, grid_points
        )

        futility_alt, efficacy_alt = _numba_grid_engine(
            n_analyses, upper_bounds, lower_bounds, n_patients_analysis, mean_1, grid_points
        )

        alpha = efficacy_null.sum()
        power = efficacy_alt.sum()
        expected_sample_size = np.sum((futility_alt + efficacy_alt) * n_patients_analysis)

        if return_table == True:
            # fill a DataFrame with the outputs
            probs_to_return = pd.DataFrame({
                "futility_null": futility_null,
                "efficacy_null": efficacy_null,
                "futility_alt": futility_alt,
                "efficacy_alt": efficacy_alt,
            }, index=[f"analysis_{i}" for i in range(1, n_analyses + 1)])

            return alpha, power, expected_sample_size, probs_to_return
        else:
            return alpha, power, expected_sample_size 


    return group_sequential_designs, math, njit, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simpson's
    """)
    return


@app.cell
def _(math, njit, np, pd):
    @njit
    def _probability_calculator(
        n_analyses, 
        upper_bounds, 
        lower_bounds, 
        n_patients_analysis, 
        means, 
        grid_points
    ):
        if grid_points % 2 == 0:
            raise Exception("Error: Simpson's rule requires an odd number of grid points.")

        fut_probs = np.zeros(n_analyses)
        eff_probs = np.zeros(n_analyses)

        # calculate the coefficients in Simpson's rule
        simpson_coeffs = np.ones(grid_points)
        for i in range (1, grid_points - 1):
            if i % 2 == 0:
                simpson_coeffs[i] = 4.
            else:
                simpson_coeffs[i] = 2.

        ####################
        # helper functions #
        ####################

        # standard normal pdf (mean=0, std=1)
        def std_norm_pdf(x):
            return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)

        # standard normal cdf (mean=0, std=1)
        def std_norm_cdf(x):
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

        ##############
        # analysis 1 #
        ##############

        # for the first stage, compute the probabilities of stopping
        # for futility and efficacy; bound - means conditions on the mean 
        fut_probs[0] = std_norm_cdf(lower_bounds[0] - means[0])
        eff_probs[0] = 1.0 - std_norm_cdf(upper_bounds[0] - means[0])

        ###################
        # analyses 2 to K #
        ###################

        if lower_bounds[0] > upper_bounds[0]:
            raise Exception("Error: Upper bounds are lower than lower bounds.")

        #---------
        # Calculate density f_1(z_1) (see Eq. 27/28 in LaTeX document)
        #---------
        # z_grid is selecting a number of grid points and splitting up
        # the continuation region (ell < cont_region < u) over which to integrate
        z_grid = np.linspace(lower_bounds[0], upper_bounds[0], grid_points)

        # fill in the probability densities (PDF NOT CDF!) for each of
        # the selected points in the continuation region; again
        # z_grid - means conditions on the mean
        density = np.zeros(grid_points)
        for i in range(grid_points):
            density[i] = std_norm_pdf(z_grid[i] - means[0])
        #---------
        # Density f_1(z_1) calculated and to be used in Stage 2 calculation
        #---------

        for k in range(1, n_analyses):

            I_prev = n_patients_analysis[k-1]
            I_curr = n_patients_analysis[k]

            # calculate the covariance between stages Cov(z_prev, z_curr)
            cov_pre_curr = np.sqrt(I_prev / I_curr)

            # the conditional standard deviation
            cond_scale = np.sqrt(1.0 - cov_pre_curr**2)

            # the width of the steps along the grid as calculated above used as
            # part of Simpson's rule calculation
            dz_prev = (upper_bounds[k-1] - lower_bounds[k-1]) / (grid_points - 1)

            # the conditional means at each grid point (mu_k|k-1)
            cond_means = means[k] + cov_pre_curr * (z_grid - means[k-1])

            #--------
            # Calculate the joint probability subdensity (see Eq. 27)
            # density is f_k-1 and the cdf calculation is intregral of g_k
            #--------
            fut_arg = np.zeros(grid_points)
            eff_arg = np.zeros(grid_points)
            for i in range(grid_points):
                fut_arg[i] = std_norm_cdf((lower_bounds[k] - cond_means[i]) / cond_scale) * density[i]
                eff_arg[i] = (1.0 - std_norm_cdf((upper_bounds[k] - cond_means[i]) / cond_scale)) * density[i]

            #--------
            # Calculate the stopping probabilities using numerical integration by
            # Simpson's rule via dot product
            #--------
            fut_probs[k] = np.dot(fut_arg, simpson_coeffs) * (dz_prev / 3.0)
            eff_probs[k] = np.dot(eff_arg, simpson_coeffs) * (dz_prev / 3.0)

            # Transition probability subdensity matrix for stages 3+ as the 
            # Stage 2 subdensity was calculated above (f_1(z_1))
            if k < n_analyses - 1:
                next_lower, next_upper = lower_bounds[k], upper_bounds[k]

                if lower_bounds[k] > upper_bounds[k]:
                    raise Exception("Error: Upper bounds are lower than lower bounds.")

                # calculating z_grid for Stage 3+ (first z_grid through loop is 
                # for z_2 grid)
                current_loop_z_grid = np.linspace(lower_bounds[k], upper_bounds[k], grid_points)

                # calculate the transition matrix that discretizes Eq. 19; this is
                # g_k from equation 19, the conditional probability density of 
                # moving from stage k-1 to k for a certain z_k
                transition_pdf = np.zeros((grid_points, grid_points))
                for r in range(grid_points):
                    for c in range(grid_points):
                        # calculate the conditional z_k for the argument to phi()
                        phi_arg = (current_loop_z_grid[r] - cond_means[c]) / cond_scale
                        # finish the g_k calculation (Eq. 23)
                        transition_pdf[r, c] = std_norm_pdf(phi_arg) / cond_scale

                # this is f_k-1 times g_k from Eq. 19
                joint_probability_matrix = transition_pdf * density

                # integrating over the rows of the joint matrix integrates out the
                # z_k-1 component
                for r in range(grid_points):
                    density[r] = np.dot(joint_probability_matrix[r, :], simpson_coeffs) * (dz_prev/3.)

                z_grid = current_loop_z_grid

        return fut_probs, eff_probs


    # high-level Python wrapper
    def group_sequential_designs_simp(
        n_analyses=3,
        upper_bounds=[2.5, 2, 1.5],
        lower_bounds=[0, 0.75, 1.5],
        n_patients=20,
        null_hypothesis=0,
        alt_hypothesis=0.5,
        variance=1,
        return_table=False,
        grid_points=101
    ):

        upper_bounds = np.asarray(upper_bounds, dtype=float)
        lower_bounds = np.asarray(lower_bounds, dtype=float)
        n_patients_analysis = n_patients * np.arange(1, n_analyses + 1)

        # standardize the null and alternative hypotheses
        # sqrt(information) * theta is the Wald statistic at each stage (Z_j)
        scale = np.sqrt(n_patients_analysis / (2 * variance))
        mean_0 = null_hypothesis * scale
        mean_1 = alt_hypothesis * scale

        futility_null, efficacy_null = _probability_calculator(
            n_analyses, upper_bounds, lower_bounds, n_patients_analysis, mean_0, grid_points
        )

        futility_alt, efficacy_alt = _probability_calculator(
            n_analyses, upper_bounds, lower_bounds, n_patients_analysis, mean_1, grid_points
        )

        alpha = efficacy_null.sum()
        power = efficacy_alt.sum()
        expected_sample_size = np.sum((futility_alt + efficacy_alt) * n_patients_analysis)

        if return_table == True:
            # fill a DataFrame with the outputs
            probs_to_return = pd.DataFrame({
                "futility_null": futility_null,
                "efficacy_null": efficacy_null,
                "futility_alt": futility_alt,
                "efficacy_alt": efficacy_alt,
            }, index=[f"analysis_{i}" for i in range(1, n_analyses + 1)])

            return alpha, power, expected_sample_size, probs_to_return
        else:
            return alpha, power, expected_sample_size

    return (group_sequential_designs_simp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Midpoint
    """)
    return


@app.cell
def _(math, np, pd):
    #@njit
    def _numba_grid_engine(
        n_analyses, 
        upper_bounds, 
        lower_bounds, 
        n_patients_analysis, 
        means, 
        grid_points
    ):
        fut_probs = np.zeros(n_analyses)
        eff_probs = np.zeros(n_analyses)

        ####################
        # helper functions #
        ####################

        # standard normal pdf (mean=0, std=1)
        def std_norm_pdf(x):
            return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)

        # standard normal cdf (mean=0, std=1)
        def std_norm_cdf(x):
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

        ##############
        # analysis 1 #
        ##############

        # for the first stage, compute the probabilities of stopping
        # for futility and efficacy; bound - means standardizes the problem
        fut_probs[0] = std_norm_cdf(lower_bounds[0] - means[0])
        eff_probs[0] = 1.0 - std_norm_cdf(upper_bounds[0] - means[0])


        if lower_bounds[0] > upper_bounds[0]:
            print("Error: Upper bounds are lower than lower bounds.")
            return

        # z_grid is selecting a number of grid points and splitting up
        # the continuation region (ell < cont_region < u)
        # midpoint
        dz = (upper_bounds[0] - lower_bounds[0]) / grid_points
        z_grid = lower_bounds[0] + (np.arange(grid_points) + 0.5) * dz

        # fill in the probability densities (PDF NOT CDF!) for each of
        # the selected points in the continuation region; again
        # z_grid - means standardizes
        density = np.zeros(grid_points)
        for i in range(grid_points):
            density[i] = std_norm_pdf(z_grid[i] - means[0])


        ###################
        # analyses 2 to K #
        ###################


        for k in range(1, n_analyses):

            I_prev = n_patients_analysis[k-1]
            I_curr = n_patients_analysis[k]

            # this is the covariance between the previous 
            # and current stages Cov(z_prev, z_curr)
            cov_pre_curr = np.sqrt(I_prev / I_curr)

            # the conditional standard deviation
            cond_scale = np.sqrt(1.0 - cov_pre_curr**2)

            # the conditional means at each grid point (vector of length
            # grid points)
            cond_means = means[k] + cov_pre_curr * (z_grid - means[k-1])

            fut_arg = np.zeros(grid_points)
            eff_arg = np.zeros(grid_points)
            for i in range(grid_points):
                fut_arg[i] = std_norm_cdf((lower_bounds[k] - cond_means[i]) / cond_scale) * density[i]
                eff_arg[i] = (1.0 - std_norm_cdf((upper_bounds[k] - cond_means[i]) / cond_scale)) * density[i]

            # the width of the steps along the grid as calculated above
            dz = (upper_bounds[k-1] - lower_bounds[k-1]) / (grid_points - 1)

            # Composite Trapezoidal Integration rule
            fut_probs[k] = (np.sum(fut_arg) - 0.5 * (fut_arg[0] + fut_arg[-1])) * dz
            eff_probs[k] = (np.sum(eff_arg) - 0.5 * (eff_arg[0] + eff_arg[-1])) * dz

            if k < n_analyses - 1:
                next_lower, next_upper = lower_bounds[k], upper_bounds[k]

                # Calculate next stage step size and midpoint grid points
                next_dz = (next_upper - next_lower) / grid_points
                next_z_grid = next_lower + (np.arange(grid_points) + 0.5) * next_dz

                # the recursive density profile is defined by the 
                # Chapman-Kolmogorov-style integral
                # transition density matrix (transition_pdf) 
                # mapping a state y at stage k−1 to a state z at stage k
                transition_pdf = np.zeros((grid_points, grid_points))
                for r in range(grid_points):
                    for c in range(grid_points):
                        diff = (next_z_grid[r] - cond_means[c]) / cond_scale
                        transition_pdf[r, c] = std_norm_pdf(diff) / cond_scale

                # Because midpoint grid cells are perfectly self-contained, 
                # the integration simplifies back to a clean matrix multiplication:
                next_matrix = transition_pdf * density
                density = np.sum(next_matrix, axis=1) * dz  # Multiplied by previous stage's dz

                # Update grid and step size for next loop iteration
                z_grid = next_z_grid
                dz = next_dz

        return fut_probs, eff_probs


    # high-level Python wrapper
    def group_sequential_designs_midpoint(
        n_analyses=3,
        upper_bounds=[2.5, 2, 1.5],
        lower_bounds=[0, 0.75, 1.5],
        n_patients=20,
        null_hypothesis=0,
        alt_hypothesis=0.5,
        variance=1,
        return_table=False,
        grid_points=100
    ):

        upper_bounds = np.asarray(upper_bounds, dtype=float)
        lower_bounds = np.asarray(lower_bounds, dtype=float)
        n_patients_analysis = n_patients * np.arange(1, n_analyses + 1)

        # standardize the null and alternative hypotheses
        # sqrt(information) * theta is the Wald statistic at each stage (Z_j)
        scale = np.sqrt(n_patients_analysis / (2 * variance))
        mean_0 = null_hypothesis * scale
        mean_1 = alt_hypothesis * scale

        futility_null, efficacy_null = _numba_grid_engine(
            n_analyses, upper_bounds, lower_bounds, n_patients_analysis, mean_0, grid_points
        )

        futility_alt, efficacy_alt = _numba_grid_engine(
            n_analyses, upper_bounds, lower_bounds, n_patients_analysis, mean_1, grid_points
        )

        alpha = efficacy_null.sum()
        power = efficacy_alt.sum()
        expected_sample_size = np.sum((futility_alt + efficacy_alt) * n_patients_analysis)

        if return_table == True:
            # fill a DataFrame with the outputs
            probs_to_return = pd.DataFrame({
                "futility_null": futility_null,
                "efficacy_null": efficacy_null,
                "futility_alt": futility_alt,
                "efficacy_alt": efficacy_alt,
            }, index=[f"analysis_{i}" for i in range(1, n_analyses + 1)])

            return alpha, power, expected_sample_size, probs_to_return
        else:
            return alpha, power, expected_sample_size 


    return (group_sequential_designs_midpoint,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Three methods compared to `scipy`
    """)
    return


@app.cell
def _(sim):
    sim.group_sequential_designs_scipy()
    return


@app.cell
def _(group_sequential_designs):
    # trapezoidal
    group_sequential_designs(grid_points=21)
    return


@app.cell
def _(group_sequential_designs_simp):
    group_sequential_designs_simp()
    return


@app.cell
def _(group_sequential_designs_midpoint):
    group_sequential_designs_midpoint()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Error analysis
    """)
    return


@app.cell
def _():
    grid = 101
    n_tests = 100
    return grid, n_tests


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Three boundaries
    """)
    return


@app.cell
def _(
    grid,
    group_sequential_designs,
    group_sequential_designs_midpoint,
    group_sequential_designs_simp,
    n_tests,
    np,
    sim,
):
    # Error tracking lists for Alpha
    trap_alpha = []
    simp_alpha = []
    mid_alpha = []

    # Error tracking lists for Power
    trap_power = []
    simp_power = []
    mid_power = []

    # Error tracking lists for ESS (Expected Sample Size)
    trap_ess = []
    simp_ess = []
    mid_ess = []

    for _i in range(n_tests):
        alpha_trap3, power_trap3, ess_trap3 = group_sequential_designs(grid_points=grid)
        alpha_simp3, power_simp3, ess_simp3 = group_sequential_designs_simp(grid_points=grid)
        alpha_scipy3, power_scipy3, ess_scipy3 = sim.group_sequential_designs_scipy()
        alpha_mid3, power_mid3, ess_mid3 = group_sequential_designs_midpoint(grid_points=grid)

        # Calculate Alpha Errors
        trap_alpha.append(abs(alpha_trap3 - alpha_scipy3))
        simp_alpha.append(abs(alpha_simp3 - alpha_scipy3))
        mid_alpha.append(abs(alpha_mid3 - alpha_scipy3))
    
        # Calculate Power Errors
        trap_power.append(abs(power_trap3 - power_scipy3))
        simp_power.append(abs(power_simp3 - power_scipy3))
        mid_power.append(abs(power_mid3 - power_scipy3))
    
        # Calculate ESS Errors
        trap_ess.append(abs(ess_trap3 - ess_scipy3))
        simp_ess.append(abs(ess_simp3 - ess_scipy3))
        mid_ess.append(abs(ess_mid3 - ess_scipy3))

    print("--- ALPHA ERRORS (3 BOUNDS) ---")
    print(f"Trapezoidal: {np.mean(trap_alpha)} +/- {np.std(trap_alpha)} | Range: [{np.min(trap_alpha)}, {np.max(trap_alpha)}]")
    print(f"Simpson's  : {np.mean(simp_alpha)} +/- {np.std(simp_alpha)} | Range: [{np.min(simp_alpha)}, {np.max(simp_alpha)}]")
    print(f"Midpoint   : {np.mean(mid_alpha)} +/- {np.std(mid_alpha)} | Range: [{np.min(mid_alpha)}, {np.max(mid_alpha)}]")
    print()
    print("--- POWER ERRORS (3 BOUNDS) ---")
    print(f"Trapezoidal: {np.mean(trap_power)} +/- {np.std(trap_power)} | Range: [{np.min(trap_power)}, {np.max(trap_power)}]")
    print(f"Simpson's  : {np.mean(simp_power)} +/- {np.std(simp_power)} | Range: [{np.min(simp_power)}, {np.max(simp_power)}]")
    print(f"Midpoint   : {np.mean(mid_power)} +/- {np.std(mid_power)} | Range: [{np.min(mid_power)}, {np.max(mid_power)}]")
    print()
    print("--- ESS ERRORS (3 BOUNDS) ---")
    print(f"Trapezoidal: {np.mean(trap_ess)} +/- {np.std(trap_ess)} | Range: [{np.min(trap_ess)}, {np.max(trap_ess)}]")
    print(f"Simpson's  : {np.mean(simp_ess)} +/- {np.std(simp_ess)} | Range: [{np.min(simp_ess)}, {np.max(simp_ess)}]")
    print(f"Midpoint   : {np.mean(mid_ess)} +/- {np.std(mid_ess)} | Range: [{np.min(mid_ess)}, {np.max(mid_ess)}]")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Five boundaries
    """)
    return


@app.cell
def _(
    grid,
    group_sequential_designs,
    group_sequential_designs_midpoint,
    group_sequential_designs_simp,
    n_tests,
    np,
    sim,
):
    bounds5 = [[3.5,   3,  2.5, 2,    1.5],
               [-3.5, -2,  0,   0.75, 1.5]]

    # Error tracking lists for Alpha
    trap1_alpha = []
    simp1_alpha = []
    mid1_alpha = []

    # Error tracking lists for Power
    trap1_power = []
    simp1_power = []
    mid1_power = []

    # Error tracking lists for ESS (Expected Sample Size)
    trap1_ess = []
    simp1_ess = []
    mid1_ess = []

    for _i in range(n_tests):
        alpha_trap5, power_trap5, ess_trap5 = group_sequential_designs(
            grid_points=grid,
            lower_bounds=bounds5[1], upper_bounds=bounds5[0]
        )
    
        alpha_simp5, power_simp5, ess_simp5 = group_sequential_designs_simp(
            grid_points=grid,
            lower_bounds=bounds5[1], upper_bounds=bounds5[0]
        )
    
        alpha_scipy5, power_scipy5, ess_scipy5 = sim.group_sequential_designs_scipy(
            lower_bounds=bounds5[1], upper_bounds=bounds5[0]
        )
    
        alpha_mid5, power_mid5, ess_mid5 = group_sequential_designs_midpoint(
            grid_points=grid,
            lower_bounds=bounds5[1], upper_bounds=bounds5[0]
        )

        # Calculate Alpha Errors
        trap1_alpha.append(abs(alpha_trap5 - alpha_scipy5))
        simp1_alpha.append(abs(alpha_simp5 - alpha_scipy5))
        mid1_alpha.append(abs(alpha_mid5 - alpha_scipy5))
    
        # Calculate Power Errors
        trap1_power.append(abs(power_trap5 - power_scipy5))
        simp1_power.append(abs(power_simp5 - power_scipy5))
        mid1_power.append(abs(power_mid5 - power_scipy5))
    
        # Calculate ESS Errors
        trap1_ess.append(abs(ess_trap5 - ess_scipy5))
        simp1_ess.append(abs(ess_simp5 - ess_scipy5))
        mid1_ess.append(abs(ess_mid5 - ess_scipy5))

    print("--- ALPHA ERRORS (5 BOUNDS) ---")
    print(f"Trapezoidal: {np.mean(trap1_alpha)} +/- {np.std(trap1_alpha)} | Range: [{np.min(trap1_alpha)}, {np.max(trap1_alpha)}]")
    print(f"Simpson's  : {np.mean(simp1_alpha)} +/- {np.std(simp1_alpha)} | Range: [{np.min(simp1_alpha)}, {np.max(simp1_alpha)}]")
    print(f"Midpoint   : {np.mean(mid1_alpha)} +/- {np.std(mid1_alpha)} | Range: [{np.min(mid1_alpha)}, {np.max(mid1_alpha)}]")
    print()
    print("--- POWER ERRORS (5 BOUNDS) ---")
    print(f"Trapezoidal: {np.mean(trap1_power)} +/- {np.std(trap1_power)} | Range: [{np.min(trap1_power)}, {np.max(trap1_power)}]")
    print(f"Simpson's  : {np.mean(simp1_power)} +/- {np.std(simp1_power)} | Range: [{np.min(simp1_power)}, {np.max(simp1_power)}]")
    print(f"Midpoint   : {np.mean(mid1_power)} +/- {np.std(mid1_power)} | Range: [{np.min(mid1_power)}, {np.max(mid1_power)}]")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Seven boundaries
    """)
    return


@app.cell
def _(
    grid,
    group_sequential_designs,
    group_sequential_designs_midpoint,
    group_sequential_designs_simp,
    n_tests,
    np,
    sim,
):
    bounds7 = [[ 5, 4,   3.5,  3,   2.5, 2,    1.5],
               [-5,-4, -3.5, -2,  0,   0.75, 1.5]]

    # Error tracking lists for Alpha
    trap2_alpha = []
    simp2_alpha = []
    mid2_alpha = []

    # Error tracking lists for Power
    trap2_power = []
    simp2_power = []
    mid2_power = []

    # Error tracking lists for ESS (Expected Sample Size)
    trap2_ess = []
    simp2_ess = []
    mid2_ess = []

    for _i in range(n_tests):
        alpha_trap7, power_trap7, ess_trap7 = group_sequential_designs(
            grid_points=grid,
            lower_bounds=bounds7[1], upper_bounds=bounds7[0]
        )
    
        alpha_simp7, power_simp7, ess_simp7 = group_sequential_designs_simp(
            grid_points=grid,
            lower_bounds=bounds7[1], upper_bounds=bounds7[0]
        )
    
        alpha_scipy7, power_scipy7, ess_scipy7 = sim.group_sequential_designs_scipy(
            lower_bounds=bounds7[1], upper_bounds=bounds7[0]
        )
    
        alpha_mid7, power_mid7, ess_mid7 = group_sequential_designs_midpoint(
            grid_points=grid,
            lower_bounds=bounds7[1], upper_bounds=bounds7[0]
        )

        # Calculate Alpha Errors
        trap2_alpha.append(abs(alpha_trap7 - alpha_scipy7))
        simp2_alpha.append(abs(alpha_simp7 - alpha_scipy7))
        mid2_alpha.append(abs(alpha_mid7 - alpha_scipy7))
    
        # Calculate Power Errors
        trap2_power.append(abs(power_trap7 - power_scipy7))
        simp2_power.append(abs(power_simp7 - power_scipy7))
        mid2_power.append(abs(power_mid7 - power_scipy7))
    
        # Calculate ESS Errors
        trap2_ess.append(abs(ess_trap7 - ess_scipy7))
        simp2_ess.append(abs(ess_simp7 - ess_scipy7))
        mid2_ess.append(abs(ess_mid7 - ess_scipy7))

    print("--- ALPHA ERRORS ---")
    print(f"Trapezoidal: {np.mean(trap2_alpha)} +/- {np.std(trap2_alpha)} | Range: [{np.min(trap2_alpha)}, {np.max(trap2_alpha)}]")
    print(f"Simpson's  : {np.mean(simp2_alpha)} +/- {np.std(simp2_alpha)} | Range: [{np.min(simp2_alpha)}, {np.max(simp2_alpha)}]")
    print(f"Midpoint   : {np.mean(mid2_alpha)} +/- {np.std(mid2_alpha)} | Range: [{np.min(mid2_alpha)}, {np.max(mid2_alpha)}]")
    print()
    print("--- POWER ERRORS ---")
    print(f"Trapezoidal: {np.mean(trap2_power)} +/- {np.std(trap2_power)} | Range: [{np.min(trap2_power)}, {np.max(trap2_power)}]")
    print(f"Simpson's  : {np.mean(simp2_power)} +/- {np.std(simp2_power)} | Range: [{np.min(simp2_power)}, {np.max(simp2_power)}]")
    print(f"Midpoint   : {np.mean(mid2_power)} +/- {np.std(mid2_power)} | Range: [{np.min(mid2_power)}, {np.max(mid2_power)}]")
    print()
    print("--- ESS ERRORS ---")
    print(f"Trapezoidal: {np.mean(trap2_ess)} +/- {np.std(trap2_ess)} | Range: [{np.min(trap2_ess)}, {np.max(trap2_ess)}]")
    print(f"Simpson's  : {np.mean(simp2_ess)} +/- {np.std(simp2_ess)} | Range: [{np.min(simp2_ess)}, {np.max(simp2_ess)}]")
    print(f"Midpoint   : {np.mean(mid2_ess)} +/- {np.std(mid2_ess)} | Range: [{np.min(mid2_ess)}, {np.max(mid2_ess)}]")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
