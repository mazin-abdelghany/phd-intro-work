# imports for study design step (Step 1)
import numpy as np
import pandas as pd
from scipy import stats

# imports for GP regression (Step 3)
import gpflow

# imports for Bayes opt (Step 4-6)
import trieste
from trieste.space import Box
from trieste.models.gpflow.models import GaussianProcessRegression
import tensorflow as tf

# simulate the trials to obtain alpha and beta
def simulate_group_sequential_designs(
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
        

simulate_group_sequential_designs(
    n_analyses=3,
    upper_bounds=[2.5, 2, 1.5],
    lower_bounds=[0, 0.75, 1.5],
    n_patients=[20],
    null_hypothesis=0,
    alt_hypothesis=0.5,
    variance=1
)

# Pocock boundaries
def calculate_pocock_boundaries(
        n_analyses=3,
        alpha=0.05,
        n_patients=20,
        sided="one.sided"):
    
    # the precision of the estimate for alpha
    epsilon = 1e-8

    # starting first set of bounds
    ub1 = np.repeat(0, repeats = n_analyses)
    lb1 = -np.repeat(0, repeats = n_analyses)

    # starting second set of bounds
    ub2 = np.repeat(10, repeats = n_analyses)
    lb2 = -np.repeat(10, repeats = n_analyses)

    # first alpha calcuation
    _, sim_alpha, _, _ = simulate_group_sequential_designs(
        n_analyses = n_analyses,
        upper_bounds = ub1,
        lower_bounds = lb1,
        n_patients = n_patients
    )

    while abs(sim_alpha - alpha) > epsilon:

        # calculate the first midpoint
        mid_u = (ub1 + ub2) / 2
        mid_l = -mid_u

        # calculate the simulated alpha
        probs, sim_alpha, power, ess = simulate_group_sequential_designs(
            n_analyses = n_analyses,
            upper_bounds = mid_u,
            lower_bounds = mid_l,
            n_patients = n_patients
        )

        if sim_alpha > alpha:
            ub1 = mid_u
            lb1 = mid_l
        else:
            ub2 = mid_u
            lb2 = mid_l

    if sided == "one.sided":
        return [
            ub1,
            np.append(lb1[0:n_analyses-1], ub1[n_analyses-1]),
            probs,
            sim_alpha,
            power,
            ess
        ]
    else:
        return [
            ub1,
            lb1,
            probs,
            sim_alpha,
            power,
            ess
        ]

# O'Brien-Fleming boundaries
def calculate_of_boundaries(
        n_analyses=3,
        alpha=0.05,
        n_patients=20,
        sided="one.sided"):

    # the precision of the estimate for alpha
    epsilon = 1e-8

    # starting first set of bounds
    ub1 = np.repeat(0, repeats = n_analyses)
    lb1 = -np.repeat(0, repeats = n_analyses)

    # starting second set of bounds
    ub2 = np.repeat(10, repeats = n_analyses)
    lb2 = -np.repeat(10, repeats = n_analyses)

    # of bounds
    of_u1 = ub1 * (np.arange(1, n_analyses+1, 1)/n_analyses)**(-0.5)
    of_l1 = -of_u1
    
    # first alpha calcuation
    _, sim_alpha, _, _ = simulate_group_sequential_designs(
        n_analyses = n_analyses,
        upper_bounds = of_u1,
        lower_bounds = of_l1,
        n_patients = n_patients
    )

    while abs(sim_alpha - alpha) > epsilon:

        # calculate the first midpoint
        mid_u = (ub1 + ub2) / 2
        mid_l = -mid_u

        # convert to O'Brien-Fleming bounds
        mid_of_u = mid_u * (np.arange(1, n_analyses+1, 1)/n_analyses)**(-0.5)
        mid_of_l = -mid_of_u

        # calculate the simulated alpha
        probs, sim_alpha, power, ess = simulate_group_sequential_designs(
            n_analyses = n_analyses,
            upper_bounds = mid_of_u,
            lower_bounds = mid_of_l,
            n_patients = n_patients
        )

        if sim_alpha > alpha:
            ub1 = mid_u
            lb1 = mid_l
        else:
            ub2 = mid_u
            lb1 = mid_l

    if sided == "one.sided":
        return [
            mid_of_u,
            np.append(mid_of_l[0:n_analyses-1], mid_of_u[n_analyses-1]),
            probs,
            sim_alpha,
            power,
            ess
        ]
    else:
        return [
            mid_of_u,
            mid_of_l,
            probs,
            sim_alpha,
            power,
            ess
        ]

# Triangular boundaries
def calculate_triangular_boundaries(
        n_analyses = 3,
        alpha = 0.05,
        delta = 0.5):

    # maximum information calculation
    I_L_term1 = (4 * (0.583**2))/n_analyses
    I_L_term2 = 8 * np.log(1/(2*alpha))
    I_L_term3 = (2*0.583) / np.sqrt(n_analyses)

    I_L = (np.sqrt(I_L_term1 + I_L_term2) - I_L_term3)**2 * (1/delta)**2

    # boundary calculation
    bounds_term1 = (2/delta) * np.log(1/(2*alpha))
    bounds_term2 = 0.583 * np.sqrt(I_L/n_analyses)

    analysis_fracs = np.arange(1, n_analyses+1, 1)/n_analyses

    I_L_fracs = I_L * analysis_fracs

    e_l = (bounds_term1 - bounds_term2 + ((0.25*delta) * analysis_fracs * I_L ))/np.sqrt(I_L_fracs)

    f_l = (-bounds_term1 + bounds_term2 + ((0.75*delta) * analysis_fracs * I_L ))/np.sqrt(I_L_fracs)

    return [
        e_l,
        f_l, 
        I_L_fracs
    ]

# sample size per group for a difference in means
def sample_size_means(
        ratio=1,
        variance=1,
        power=0.8,
        alpha=0.05,
        delta=1):

    # ratio of smaller group to larger group
    r = (ratio+1)/ratio

    # z statistic for power
    z_power = stats.norm.ppf(power)
    
    # z statistic for alpha
    z_alpha = abs(stats.norm.ppf(alpha))

    # sample size
    n = r * ((variance**2 * (z_power+z_alpha)**2) / delta**2)

    return n

# obtain maximum expected sample size for an interval of interest
def max_ess(
        delta_start=0,
        n_analyses=3,
        upper_bounds=[2.5, 2, 1.5],
        lower_bounds=[0, 0.75, 1.5],
        n_patients=20,
        null_hypothesis=0,
        variance=1):

    # epsilon precision for max ESS calculated
    epsilon = 1e-4

    # delta_stop based on the variance (5x the variance)
    delta_stop = variance * 5
    
    # random starting values of ess
    ess_delta_start = 10
    ess_delta_stop = 0
    
    # while the error is greater than desired precision
    while abs(ess_delta_start - ess_delta_stop) > epsilon:

        # simulate the trial under delta_start
        probs, sim_alpha, power, ess_delta_start = simulate_group_sequential_designs(
            n_analyses = n_analyses,
            upper_bounds = upper_bounds,
            lower_bounds = lower_bounds,
            n_patients = n_patients,
            null_hypothesis = null_hypothesis,
            alt_hypothesis = delta_start,
            variance = variance
        )
    
        # simulate trial under delta_stop
        probs, sim_alpha, power, ess_delta_stop = simulate_group_sequential_designs(
            n_analyses = n_analyses,
            upper_bounds = upper_bounds,
            lower_bounds = lower_bounds,
            n_patients = n_patients,
            null_hypothesis = null_hypothesis,
            alt_hypothesis = delta_stop,
            variance = variance
        )
        
        if ess_delta_start >= ess_delta_stop:
            delta_stop = (delta_start + delta_stop)/2 
        else:
            delta_start = (delta_start + delta_stop)/2

    return ess_delta_start

deltas=np.linspace(start=0, stop=3, num=1000)
ess_list = np.empty(1000)

for delta in deltas:
    probs, sim_alpha, power, ess = simulate_group_sequential_designs(
        alt_hypothesis = delta
    )

    ess_list = np.append(ess_list, ess)
    
ess_list.max()

def find_sample_size(
        power_target=0.9,
        n_analyses=3,
        upper_bounds=[2.5, 2, 1.5],
        lower_bounds=[0, 0.75, 1.5],
        null_hypothesis=0,
        alt_hypothesis=0.5,
        variance=1):

    # precision to get to power
    epsilon = 1e-8

    # initial sample sizes
    n_patients_left=1
    n_patients_right=1e6

    # calcuate first power
    probs, sim_alpha, power, ess = simulate_group_sequential_designs(
        n_analyses = n_analyses,
        upper_bounds = upper_bounds,
        lower_bounds = lower_bounds,
        n_patients = n_patients_left,
        null_hypothesis = null_hypothesis,
        alt_hypothesis = alt_hypothesis,
        variance = variance
    )
    
    # interval bisection loop
    while abs(power - power_target) > epsilon:

        # generate the midpoint
        n_patients_mid = (n_patients_left + n_patients_right)/2

        # calcuate power
        probs, sim_alpha, power, ess = simulate_group_sequential_designs(
            n_analyses = n_analyses,
            upper_bounds = upper_bounds,
            lower_bounds = lower_bounds,
            n_patients = n_patients_mid,
            null_hypothesis = null_hypothesis,
            alt_hypothesis = alt_hypothesis,
            variance = variance
        )

        if power > power_target:
            n_patients_right = n_patients_mid
        else:
            n_patients_left = n_patients_mid

    return [n_patients_left, power]

# generate the penalty term
def feasibility_penalty(
        ratio=1,
        variance=1,
        power=0.8,
        alpha=0.05,
        delta=1,
        beta_prime=0.7,
        alpha_prime=0.1):

    # calculate the sample size for one-stage design
    mu = sample_size_means(
        ratio=ratio,
        variance=variance,
        power=power,
        alpha=alpha,
        delta=delta
    )

    # create the indicator functions
    def alpha_indicator(alpha_prime):
        if alpha_prime > alpha: return 1
        return 0

    def beta_indicator(beta_prime):
        if beta_prime > power: return 1
        return 0

    return mu * ( (((alpha_prime-alpha)/alpha)*alpha_indicator(alpha_prime)) + (((beta_prime-power)/power)*beta_indicator(beta_prime)) )

# function to minimize
def function_to_minimize(max_ess_val, penalty):

    return max_ess_val + penalty

# define a function that generates the input
# takes boundaries and n_patients as inputs and outputs a single array
# may need to add scaling if GPR fit is challenging
def generate_gpr_input(
        n_analyses,
        upper_bounds,
        lower_bounds,
        n_patients):

    # turn n_patients into an array
    n_patients = np.array([n_patients])

    # remove the redundant value from the lower bound
    lower_bounds_trunc = lower_bounds[0:n_analyses-1]

    # concatenate and return
    return np.concatenate((upper_bounds, lower_bounds_trunc, n_patients))

################
# GPR workflow #
################

# some set defaults
num_analyses = 3
target_alpha = 0.05
target_power = 0.9
important_diff_delta = 1
assumed_variance = 3

# to obtain mu (sample size at one stage)
group_ratio = 1

# simulate the trial design 
simulation = calculate_pocock_boundaries(
    n_analyses=num_analyses,
    alpha=target_alpha,
    n_patients=20
)

# pull inputs to the below functions from simulation
upper = simulation[0]
lower = simulation[1]
alpha_prime = simulation[3]

# 1. Find the number of patients that achieves 90% power (beta 0.1)
# here we get beta_prime and alpha prime
n_power09, beta_prime = find_sample_size(n_analyses=num_analyses,
                                         upper_bounds=upper,
                                         lower_bounds=lower,
                                         alt_hypothesis=important_diff_delta,
                                         variance=assumed_variance)

# 2. Generate the GPR input values
# note that the input includes the sample size at power 0.9
x1 = generate_gpr_input(n_analyses = num_analyses,
                        upper_bounds=upper,
                        lower_bounds=lower,
                        n_patients=n_power09)

# 3. Generate maximum expected sample size and feasibility penalty
max_ess_new = max_ess(n_analyses=num_analyses,
                      upper_bounds=upper,
                      lower_bounds=lower,
                      n_patients=n_power09)

penalty = feasibility_penalty(ratio=group_ratio,
                              variance=assumed_variance,
                              power=target_power,
                              alpha=target_alpha,
                              delta=important_diff_delta,
                              beta_prime=beta_prime,
                              alpha_prime=alpha_prime)


# 4. Calculate the function value (GPR output)
y1 = function_to_minimize(max_ess_val=max_ess_new, penalty=penalty)

# simulate the trial design 
simulation = calculate_of_boundaries(
    n_analyses=num_analyses,
    alpha=target_alpha,
    n_patients=20
)

# pull inputs to the below functions from simulation
upper = simulation[0]
lower = simulation[1]
alpha_prime = simulation[3]

# 1. Find the number of patients that achieves 90% power (beta 0.1)
# here we get beta_prime and alpha prime
n_power09, beta_prime = find_sample_size(n_analyses=num_analyses,
                                         upper_bounds=upper,
                                         lower_bounds=lower,
                                         alt_hypothesis=important_diff_delta,
                                         variance=assumed_variance)

# 2. Generate the GPR input values
# note that the input includes the sample size at power 0.9
x2 = generate_gpr_input(n_analyses = num_analyses,
                       upper_bounds=upper,
                       lower_bounds=lower,
                       n_patients=n_power09)

# 3. Generate maximum expected sample size and feasibility penalty
max_ess_new = max_ess(n_analyses=num_analyses,
                      upper_bounds=upper,
                      lower_bounds=lower,
                      n_patients=n_power09)

penalty = feasibility_penalty(ratio=group_ratio,
                              variance=assumed_variance,
                              power=target_power,
                              alpha=target_alpha,
                              delta=important_diff_delta,
                              beta_prime=beta_prime,
                              alpha_prime=alpha_prime)


# 4. Calculate the function value (GPR output)
y2 = function_to_minimize(max_ess_val=max_ess_new, penalty=penalty)

def build_model(X, Y, kernel_func=None):
    variance = tf.math.reduce_variance(X)
    
    if kernel_func is None:
        kernel = gpflow.kernels.Matern52(variance=variance)
    else:
        kernel = kernel_func(variance)
        
    gpr = gpflow.models.GPR(
        data = (X, Y),
        # setting the kernel variance to something other than 1 breaks the optimization
        # likely because things are not standardized/scaled?
        kernel = kernel_func(variance=1),
        likelihood = gpflow.likelihoods.Gaussian()
    )

    gpflow.utilities.print_summary(gpr, fmt="notebook")
    
    return GaussianProcessRegression(gpr)

# GP regression
model = build_model(X = X, Y = Y, kernel_func=gpflow.kernels.SquaredExponential)

# create a dataset that works well with trieste
initial_data = trieste.data.Dataset(query_points=X, observations=Y)

# create the search space using trieste Box function
search_space = Box([-20, -20, -20, -20, -20, 0], [20, 20, 20, 20, 20, 1000])

ask_tell = trieste.ask_tell_optimization.AskTellOptimizer(search_space = search_space,
                                                          datasets = initial_data, 
                                                          models = model)

results = ask_tell.ask()

def format_boundaries_after_ask(result):
    upper_bounds = np.array(result[0][0:3])
    lower_bounds = np.concatenate((result[0][3:5], [result[0][2]]))
    n_patients = np.array(result[0][5])

    return [
        upper_bounds,
        lower_bounds,
        n_patients
    ]

output = format_boundaries_after_ask(results)

sim_results, alpha_prime, beta_prime, ess = simulate_group_sequential_designs(
    n_analyses=num_analyses,
    upper_bounds=output[0],
    lower_bounds=output[1],
    n_patients=output[2],
    alt_hypothesis=important_diff_delta,
    variance=assumed_variance
)

n_power09, beta_prime = find_sample_size(n_analyses=num_analyses,
                                         upper_bounds=output[0],
                                         lower_bounds=output[1],
                                         alt_hypothesis=important_diff_delta,
                                         variance=assumed_variance)

# 1. Find the number of patients that achieves 90% power (beta 0.1)
# here we get beta_prime and alpha prime
n_power09, beta_prime = find_sample_size(n_analyses=num_analyses,
                                         upper_bounds=output[0],
                                         lower_bounds=output[1],
                                         alt_hypothesis=important_diff_delta,
                                         variance=assumed_variance)

# 2. Generate the GPR input values
# note that the input includes the sample size at power 0.9
x3 = generate_gpr_input(n_analyses = num_analyses,
                       upper_bounds=output[0],
                       lower_bounds=output[1],
                       n_patients=n_power09)

# 3. Generate maximum expected sample size and feasibility penalty
max_ess_new = max_ess(n_analyses=num_analyses,
                      upper_bounds=output[0],
                      lower_bounds=output[1],
                      n_patients=n_power09)

penalty = feasibility_penalty(ratio=group_ratio,
                              variance=assumed_variance,
                              power=target_power,
                              alpha=target_alpha,
                              delta=important_diff_delta,
                              beta_prime=beta_prime,
                              alpha_prime=alpha_prime)


# 4. Calculate the function value (GPR output)
y3 = function_to_minimize(max_ess_val=max_ess_new, penalty=penalty)