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
        variance=1,
        epsilon=1e-2):

    delta_stop = variance * 5.
    
    # step size to calculate the derivative (slope)
    h = 1e-5

    def get_ess(delta):
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

    # bisection loop
    while (delta_stop - delta_start) > epsilon:
        midpoint = (delta_start + delta_stop) / 2.0
        
        # sample slightly to the left and right of the midpoint to get the slope
        ess_left = get_ess(midpoint - h)
        ess_right = get_ess(midpoint + h)
        
        slope = ess_right - ess_left

        if slope > 0:
            # slope is positive: we are climbing up the left side of the hill.
            # the peak must be to the right.
            delta_start = midpoint
        else:
            # slope is negative: we are sliding down the right side of the hill.
            # The peak must be to the left.
            delta_stop = midpoint

    # Return the final optimized ESS at the peak midpoint
    return get_ess((delta_start + delta_stop) / 2.0)

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
