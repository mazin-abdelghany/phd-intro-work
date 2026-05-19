import numpy as np
import pandas as pd
import math
from numba import njit

@njit
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
    
    if lower_bounds[0] >= upper_bounds[0]:
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

        # the width of the steps along the grid as calculated above
        dz = (upper_bounds[k-1] - lower_bounds[k-1]) / (grid_points - 1)

        # the conditional means at each grid point (vector of length
        # grid points)
        cond_means = means[k] + cov_pre_curr * (z_grid - means[k-1])
        
        fut_arg = np.zeros(grid_points)
        eff_arg = np.zeros(grid_points)
        for i in range(grid_points):
            fut_arg[i] = std_norm_cdf((lower_bounds[k] - cond_means[i]) / cond_scale) * density[i]
            eff_arg[i] = (1.0 - std_norm_cdf((upper_bounds[k] - cond_means[i]) / cond_scale)) * density[i]
            
        # Composite Trapezoidal Integration rule
        fut_probs[k] = (np.sum(fut_arg) - 0.5 * (fut_arg[0] + fut_arg[-1])) * dz
        eff_probs[k] = (np.sum(eff_arg) - 0.5 * (eff_arg[0] + eff_arg[-1])) * dz
        
        # Grid transition to next stage
        if k < n_analyses - 1:
            next_lower, next_upper = lower_bounds[k], upper_bounds[k]
            
            if next_lower >= next_upper:
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
