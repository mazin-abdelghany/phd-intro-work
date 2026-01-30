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