# generate the penalty term
def feasibility_penalty(
        mu,
        power=0.8,
        alpha=0.05,
        beta_prime=0.7,
        alpha_prime=0.1):

    # calculate beta from power
    beta = 1-power

    # create the indicator functions
    def alpha_indicator(alpha_prime):
        if alpha_prime > alpha: return 1
        return 0

    def beta_indicator(beta_prime):
        if beta_prime > beta: return 1
        return 0

    return mu * (
        (((alpha_prime-alpha)/alpha)*alpha_indicator(alpha_prime)) + 
        (((beta_prime-beta)/power)*beta_indicator(beta_prime)) 
    )

# new feasibility penalty
def new_penalty(
        mu,
        power,
        alpha,
        alpha_prime,
        beta_prime):
    
    # calculate power from beta
    beta = 1-power
    
    if (alpha_prime > alpha) and (beta_prime > beta):
        return mu * ((alpha_prime - alpha)**2 + (beta_prime - beta)**2)
    return mu
    
# smooth feasibility penalty
def smooth_penalty(
        mu,
        power,
        alpha,
        alpha_prime,
        beta_prime):
    
    # calculate beta from power
    beta = 1-power

    return mu * ((alpha_prime - alpha)**2 + (beta_prime - beta)**2)

# penalty with lower repulsion
def low_repulsion(
        mu,
        power,
        alpha,
        alpha_prime,
        beta_prime):
    
    # calculate beta from power
    beta = 1-power

    return mu * ((alpha_prime - alpha)**4 + (beta_prime - beta)**4)

# penalty with cliff and high repulsion
def absolute_cliff(
        mu,
        power,
        alpha,
        alpha_prime,
        beta_prime,
        alpha_epsilon = 0.01,
        beta_epsilon = 0.05):
    
    # calculate beta from power
    beta = 1-power

    alpha_met = (-alpha_epsilon <= alpha_prime - alpha) & (alpha_prime - alpha <= 0)
    beta_met = (-beta_epsilon <= beta_prime - beta) & (beta_prime - beta <= 0)

    if (alpha_met and beta_met):
        return 0
    else:
        return mu * (abs(alpha_prime - alpha) + abs(beta_prime - beta))
