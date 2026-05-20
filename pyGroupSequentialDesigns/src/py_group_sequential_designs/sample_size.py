from scipy import stats
from scipy.optimize import minimize_scalar
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
    z_alpha = stats.norm.ppf(1-alpha)

    # sample size
    n = r * ((variance * (z_power+z_alpha)**2) / delta**2)

    return n

# obtain maximum expected sample size for an interval of interest
# using interval search on the derivative
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
    
    # helper function 
    def run_sim(delta):
        _, _, ess = sim.group_sequential_designs(
            n_analyses = n_analyses,
            upper_bounds = upper_bounds,
            lower_bounds = lower_bounds,
            n_patients = n_patients,
            null_hypothesis = null_hypothesis,
            alt_hypothesis = delta,
            variance = variance
        )
        return ess

    # calculate initial values outside the loop
    ess_delta_start = run_sim(delta_start)
    ess_delta_stop = run_sim(delta_stop)
    
    # while the error is greater than desired precision
    while abs(ess_delta_start - ess_delta_stop) > epsilon:
        # calculate the midpoint
        midpoint = (delta_start + delta_stop) / 2.0
        
        if ess_delta_start >= ess_delta_stop:
            # move the right bound to the midpoint
            delta_stop = midpoint
            # only update the ESS value that actually changed
            ess_delta_stop = run_sim(delta_stop)
        else:
            # move the left bound to the midpoint
            delta_start = midpoint
            # only update the ESS value that actually changed
            ess_delta_start = run_sim(delta_start)

    return ess_delta_start

def find_sample_size(
        power_target=0.9,
        n_analyses=3,
        upper_bounds=[2.5, 2, 1.5],
        lower_bounds=[0, 0.75, 1.5],
        null_hypothesis=0,
        alt_hypothesis=0.5,
        variance=1,
        epsilon=1e-2):

    # initial sample sizes
    n_patients_min = 2
    n_patients_max = 5000

    while (n_patients_max - n_patients_min) > epsilon:

        n_mid = (n_patients_min + n_patients_max) / 2

        _, power_mid, _ = sim.group_sequential_designs(
            n_analyses = n_analyses,
            upper_bounds = upper_bounds,
            lower_bounds = lower_bounds,
            n_patients = n_mid,
            null_hypothesis = null_hypothesis,
            alt_hypothesis = alt_hypothesis,
            variance = variance
        )

        if power_mid >= power_target:
            n_patients_max = n_mid
        else:
            n_patients_min = n_mid

    return [n_patients_min, power_mid]
