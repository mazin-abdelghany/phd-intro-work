import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

# simulate the trials to obtain alpha and beta
def group_sequential_designs(
        n_analyses=3,
        upper_bounds=[2.5, 2, 1.5],
        lower_bounds=[0, 0.75, 1.5],
        n_patients=20,
        null_hypothesis=0,
        alt_hypothesis=0.5,
        variance=1):

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

    # fill a DataFrame with the outputs
    probs_to_return = pd.DataFrame({
        "futility_null": futility_null,
        "efficacy_null": efficacy_null,
        "futility_alt": futility_alt,
        "efficacy_alt": efficacy_alt,
    }, index=[f"analysis_{i}" for i in range(1, n_analyses + 1)])

    # get alpha and power; alpha is the error of claiming efficacy
    # under the null. Power is the "correct" call of efficacy under
    # the alternative
    alpha = efficacy_null.sum()
    power = efficacy_alt.sum()

    # calculate the expected sample size by summing stopping for
    # futility or efficacy under the alternative
    summed_probs = futility_alt + efficacy_alt
    expected_sample_size = np.sum(summed_probs * n_patients_analysis)

    return probs_to_return, alpha, power, expected_sample_size
