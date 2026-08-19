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
    z_alpha = stats.norm.ppf(1-alpha)

    # sample size
    n = r * ((variance * (z_power+z_alpha)**2) / delta**2)

    return n

# obtain maximum expected sample size for an interval of interest
# using golden-ratio search 
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

    # delta_stop based on the variance (10x the standard deviation)
    delta_stop = variance**0.5 * 10 
    
    def run_sim(delta):
        _, _, ess = sim.group_sequential_designs(
            n_analyses=n_analyses,
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
            n_patients=n_patients,
            null_hypothesis=null_hypothesis,
            alt_hypothesis=delta,
            variance=variance
        )
        return ess

    # inverse of the golden ratio constant 1/phi 
    inv_phi = (5**0.5 - 1) / 2.0  # ~0.618033

    a = delta_start
    b = delta_stop

    # evaluate starting point to avoid plateau violation (see comment below)
    fa = run_sim(a)

    # Define two interior points
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)

    # find the function evaluation at c and d 
    fc = run_sim(c)
    fd = run_sim(d)

    # Search until search interval width is less than epsilon
    while (b - a) > epsilon:
        # at >= 5 times the standard deviation, there is a plateau, which
        # violates the assumption for golden-section search. to avoid this 
        # violation, we check the left-most point in the search (delta_start).
        # if we are in the plateau (i.e., fc == fd) and f(a) > f(c), then we must
        # discard the [d, end] interval
        if (fc > fd) or (fc == fd and fa > fc):
            # if f(c) > f(d), then the interval [d, end] can be excluded
            b = d # b was the endpoint, it becomes d
            d = c # the already evaluated interior point c becomes d
            fd = fc
            c = b - inv_phi * (b - a) # choose a new point in the new interval
            fc = run_sim(c)
        # in this else statement, there are two possibilities, either the peak
        # is further rightward, or initial c and d have been selected on either
        # side of the maximum. either way, a valid search can continue in this 
        # block
        else:
            # if f(d) > f(c), then the interval [start, c] can be excluded
            a = c
            c = d
            fc = fd
            d = a + inv_phi * (b - a)
            fd = run_sim(d)

    # Evaluate max ESS at final midpoint
    best_delta = (a + b) / 2.0
    return run_sim(best_delta)

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
