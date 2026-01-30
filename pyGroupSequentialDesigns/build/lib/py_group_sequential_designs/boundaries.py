import numpy as np

from . import simulate as sim

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
    _, sim_alpha, _, _ = sim.group_sequential_designs(
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
        probs, sim_alpha, power, ess = sim.group_sequential_designs(
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
    _, sim_alpha, _, _ = sim.group_sequential_designs(
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
        probs, sim_alpha, power, ess = sim.group_sequential_designs(
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