import numpy as np
import pandas as pd
import math
from numba import njit
from scipy.stats import multivariate_normal

@njit
def _probability_calculator(
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
        # trapezoidal rule product
        #--------
        # the width of the steps along the grid 
        dz_prev = (upper_bounds[k-1] - lower_bounds[k-1]) / (grid_points - 1)

        # trapezoidal integration rule -- summing all arguments and then 
        # subtracting the extra 0.5 of first and last argument
        fut_probs[k] = (np.sum(fut_arg) - 0.5 * (fut_arg[0] + fut_arg[-1])) * dz_prev
        eff_probs[k] = (np.sum(eff_arg) - 0.5 * (eff_arg[0] + eff_arg[-1])) * dz_prev
        
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
            density = (np.sum(joint_probability_matrix, axis=1) - 0.5 * (joint_probability_matrix[:, 0] + joint_probability_matrix[:, -1])) * dz_prev
            
            z_grid = current_loop_z_grid
            
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

def group_sequential_designs_scipy(
    n_analyses=3,
    upper_bounds=[2.5, 2, 1.5],
    lower_bounds=[0, 0.75, 1.5],
    n_patients=20,
    null_hypothesis=0,
    alt_hypothesis=0.5,
    variance=1,
    return_table=False
):

    upper_bounds = np.asarray(upper_bounds, dtype=float)
    lower_bounds = np.asarray(lower_bounds, dtype=float)

    # cumulative sample sizes
    n_patients_analysis = n_patients * np.arange(1, n_analyses + 1)

    # compute the means
    scale = np.sqrt(n_patients_analysis / (2 * variance))

    # mean_0 is under null, mean_1 is under alternative
    mean_0 = null_hypothesis * scale
    mean_1 = alt_hypothesis * scale

    # n_i becomes a (x by 1) vector
    # n_j becomes a (1 by x) vector
    n_i = n_patients_analysis[:, None]
    n_j = n_patients_analysis[None, :]

    # np.minimum and np.maximum broadcasts the vectors into (x by x)
    # matrix where np.minimum contains the min across the rows and colums
    # comparing each dimension and np.maximum contains the max across
    # the rows and columns comparing each dimension
    full_sigma = np.sqrt(np.minimum(n_i, n_j) / np.maximum(n_i, n_j))

    # store results in arrays first, then add to DataFrame at the end
    futility_null = np.empty(n_analyses)
    efficacy_null = np.empty(n_analyses)
    futility_alt = np.empty(n_analyses)
    efficacy_alt = np.empty(n_analyses)

    for i in range(n_analyses):

        # the number of dimensions to subselect
        dim = i + 1

        # select a (dim by dim) section of the matrix
        cov = full_sigma[:dim, :dim]

        mvn_null = multivariate_normal(
            mean=mean_0[:dim],
            cov=cov
        )

        mvn_alt = multivariate_normal(
            mean=mean_1[:dim],
            cov=cov
        )

        # Futility bounds
        fut_l = np.concatenate([lower_bounds[:i], [-np.inf]])
        fut_u = np.concatenate([upper_bounds[:i], [lower_bounds[i]]])

        # Efficacy bounds
        eff_l = np.concatenate([lower_bounds[:i], [upper_bounds[i]]])
        eff_u = np.concatenate([upper_bounds[:i], [np.inf]])

        futility_null[i] = mvn_null.cdf(fut_u, lower_limit=fut_l)
        futility_alt[i] = mvn_alt.cdf(fut_u, lower_limit=fut_l)

        efficacy_null[i] = mvn_null.cdf(eff_u, lower_limit=eff_l)
        efficacy_alt[i] = mvn_alt.cdf(eff_u, lower_limit=eff_l)

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
