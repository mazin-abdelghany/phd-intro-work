import numpy as np
import pandas as pd
from scipy import stats

# simulate the trials to obtain alpha and beta
def group_sequential_designs(
        n_analyses=3,
        upper_bounds=[2.5, 2, 1.5],
        lower_bounds=[0, 0.75, 1.5],
        n_patients=20,
        null_hypothesis=0,
        alt_hypothesis=0.5,
        variance=1):

    # convert list to np ndarray
    upper_bounds = np.array(upper_bounds)
    lower_bounds = np.array(lower_bounds)
    
    # assign values for null and alt hypotheses
    theta_0 = null_hypothesis
    delta = alt_hypothesis

    # empty list to fill mean vectors
    mean_0 = []
    mean_1 = []

    # number of patients in each analysis
    n_patients_analysis = np.array([x for x in range(1, n_analyses + 1, 1)]) * n_patients

    # need to parse the upper and lower boundaries of the design
    # for futility and efficacy, must put the bounds of integration correctly
    # for pmvnorm
    futility_l_bounds = [[]]
    futility_u_bounds = [[]]
    efficacy_l_bounds = [[]]
    efficacy_u_bounds = [[]]

    n_analyses = len(upper_bounds)

    # loop through number of analyses
    for i in range(n_analyses):

        # special case of i = 1
        if i == 0:
            futility_l_bounds[i].append(-np.inf)
            futility_u_bounds[i].append(lower_bounds[i])
            efficacy_l_bounds[i].append(upper_bounds[i])
            efficacy_u_bounds[i].append(np.inf)
            continue

        # all other cases
        futility_l_bounds.append(np.append(lower_bounds[0:i], -np.inf))
        futility_u_bounds.append(np.append(upper_bounds[0:i], lower_bounds[i]))
        efficacy_l_bounds.append(np.append(lower_bounds[0:i], upper_bounds[i]))
        efficacy_u_bounds.append(np.append(upper_bounds[0:i], np.inf))

    # empty dictionary of SIGMA matrices
    SIGMA_dict = dict()

    # generate the SIGMA matrices
    for i in range(n_analyses):

        if i == 0: SIGMA_dict.update({i : np.sqrt(variance)})

        # start with diagonal matrix for SIGMA
        SIGMA = np.eye(N = i+1)

        # n = 2, need to fill all but 11, 22
        # n = 3, need to fill all but 11, 22, 33
        # n = 4, need to fill all but 11, 22, 33, 44
        # etc.
        for j in range(i+1):
            for k in range(i+1):

                # leave the 1s on the diagonal, skip interation
                if j == k: continue

                # when j is less than k, the lower number of patients will be in numerator
                if j < k: SIGMA[j,k] = np.sqrt(n_patients_analysis[j] / n_patients_analysis[k])

                # when j is greater than j, the lower number of patients will be in numerator
                if j > k: SIGMA[j,k] = np.sqrt(n_patients_analysis[k] / n_patients_analysis[j])

        SIGMA_dict.update({i : SIGMA})

    # empty data frame to collect probabilities
    # dictionary to DataFrame, column is the key:value pair of the dictionary
    probs_to_return = {
        "futility_null" : np.empty(n_analyses),
        "efficacy_null" : np.empty(n_analyses),
        "futility_alt" : np.empty(n_analyses),
        "efficacy_alt" : np.empty(n_analyses)
    }

    # generate the row names
    row_names = ["analysis_" + str(row_names) for row_names in range(1, n_analyses + 1, 1)]

    # create the empty data frame with data and index
    probs_to_return = pd.DataFrame(data = probs_to_return, index = row_names)
    
    # start calculations for the analyses
    for i in range(n_analyses):
        
        # mean under null
        mean_0.append(theta_0 * np.sqrt(n_patients_analysis[i] / (2 * variance)))

        # mean under alternative
        mean_1.append(delta   * np.sqrt(n_patients_analysis[i] / (2 * variance)))
        
        # generate the null and alt multivariate normal
        mvn_null = stats.multivariate_normal(mean = mean_0, cov = SIGMA_dict[i])
        mvn_alt  = stats.multivariate_normal(mean = mean_1, cov = SIGMA_dict[i])
        
        # prob stop for futility under null
        futility_null = mvn_null.cdf(futility_u_bounds[i], lower_limit = futility_l_bounds[i])
        
        # prob stop for futility under alt
        futility_alt = mvn_alt.cdf(futility_u_bounds[i], lower_limit = futility_l_bounds[i])
        
        # prob stop for efficacy under null
        efficacy_null = mvn_null.cdf(efficacy_u_bounds[i], lower_limit = efficacy_l_bounds[i])

        # prob stop for efficacy under alt
        efficacy_alt = mvn_alt.cdf(efficacy_u_bounds[i], lower_limit = efficacy_l_bounds[i])

        # add them to the data frame to return
        probs_to_return.iloc[i] = [futility_null, efficacy_null, futility_alt, efficacy_alt]

    # get the type I error (alpha)
    alpha = probs_to_return.sum(axis = 0)["efficacy_null"]

    # get the power (1 - beta)
    power = probs_to_return.sum(axis = 0)["efficacy_alt"]

    # get expected sample size
    summed_probs = probs_to_return.iloc[:,2:4].sum(axis=1)
    expected_sample_size = np.sum(summed_probs * n_patients_analysis)
    
    return probs_to_return, alpha, power, expected_sample_size