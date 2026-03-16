import numpy as np

def format_boundaries_after_ask(n_analyses, result):

    array_len = (n_analyses*2)-1
    upper_bound_len = n_analyses

    upper_bounds = np.array(result[0][0:upper_bound_len])
    lower_bounds = np.concatenate(
        (result[0][upper_bound_len:array_len], 
         [result[0][upper_bound_len-1]])
    )
    n_patients = np.array(result[0][array_len])

    return [
        upper_bounds,
        lower_bounds,
        n_patients
    ]

# create a function that checks the monotonicity of bounds
def check_monotonicity(n_analyses, bounds):
    
    # first format the boundaries from the ask_tell interface
    # this takes bounds such as [upper1, upper2, upper3, lower1, lower2, n]
    # and outputs a list of lists 
    # [
    #   [upper1, upper2, upper3],
    #   [lower1, lower2, lower3], 
    #   n
    # ]
    fmt_bounds = format_boundaries_after_ask(
        n_analyses=n_analyses, 
        result=np.array([bounds])
    )

    # take the first two indices from the list, these are the upper and lower 
    # bounds
    upper = fmt_bounds[0]
    lower = fmt_bounds[1]

    # loop through the bounds
    for _i in range(len(upper)-1):

        # if we are not at the last stage
        if (_i != len(upper)-1):
            # a design is invalid if the upper bounds are not monotonicly
            # decreasing
            if upper[_i] < upper[_i+1]: return False
            # a design is invalid if the lower bounds are not monotonicly 
            # increasing
            if lower[_i] > lower[_i+1]: return False
            # a design is invalid if the upper bound is not greater than the 
            # lower bound
            if upper[_i] <= lower[_i]: return False

        # at the last stage, design is invalid if the upper bound is to the 
        # lower bound as this is a one-sided statistical test
        else:
            if upper[_i] != lower[_i]: return False

    return True
