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
        # Removed the rounding here so the optimization handles 
        # continuous, smooth values.
        return ess

    # Pure interval bisection for optimization (Dichotomous Search)
    while (delta_stop - delta_start) > epsilon:
        # Pick two test points close to the center of the current interval
        midpoint = (delta_start + delta_stop) / 2.0
        x1 = midpoint - (epsilon / 4.0)
        x2 = midpoint + (epsilon / 4.0)
        
        ess_1 = get_ess(x1)
        ess_2 = get_ess(x2)

        if ess_1 < ess_2:
            # ESS is higher to the right, so the peak cannot be in the far left
            delta_start = x1
        else:
            # ESS is higher to the left (or equal), so the peak cannot be in the far right
            delta_stop = x2

    # Return the final optimized ESS at the true peak midpoint
    final_delta = (delta_start + delta_stop) / 2.0
    return get_ess(final_delta)

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
