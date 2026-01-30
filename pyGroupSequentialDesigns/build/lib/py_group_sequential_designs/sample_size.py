from scipy import stats

from . import simulate as sim

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
        probs, sim_alpha, power, ess_delta_start = sim.group_sequential_designs(
            n_analyses = n_analyses,
            upper_bounds = upper_bounds,
            lower_bounds = lower_bounds,
            n_patients = n_patients,
            null_hypothesis = null_hypothesis,
            alt_hypothesis = delta_start,
            variance = variance
        )
    
        # simulate trial under delta_stop
        probs, sim_alpha, power, ess_delta_stop = sim.group_sequential_designs(
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
    probs, sim_alpha, power, ess = sim.group_sequential_designs(
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
        probs, sim_alpha, power, ess = sim.group_sequential_designs(
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