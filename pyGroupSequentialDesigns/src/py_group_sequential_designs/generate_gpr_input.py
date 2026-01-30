import numpy as np

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