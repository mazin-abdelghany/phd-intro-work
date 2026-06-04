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
        beta_prime,
        scaled=False,
        scale_factor=20):
    
    # calculate beta from power
    beta = 1-power
    
    # find the maximum alpha and beta
    if alpha <= 0.5:
        max_alpha = (alpha - 1)**2
    else:
        max_alpha = alpha**2
    
    if beta <= 0.5:
        max_beta = (beta - 1)**2
    else:
        max_beta = beta**2

    # min-max scale if scaled=True; note min alpha and beta is 0
    if scaled:
        val_a = (alpha_prime - alpha)**2
        val_b = (beta_prime - beta)**2

        val_a_scaled = val_a / max_alpha
        val_b_scaled = val_b / max_beta

        return scale_factor * (val_a_scaled + val_b_scaled)
    else:
        return mu * ((alpha_prime - alpha)**2 + (beta_prime - beta)**2)

# penalty with lower repulsion
def low_repulsion(
        mu,
        power,
        alpha,
        alpha_prime,
        beta_prime,
        scaled=False,
        scale_factor=20):
    
    # calculate beta from power
    beta = 1-power
    
    # find the maximum alpha and beta
    if alpha <= 0.5:
        max_alpha = (alpha - 1)**4
    else:
        max_alpha = alpha**4
    
    if beta <= 0.5:
        max_beta = (beta - 1)**4
    else:
        max_beta = beta**4

    # min-max scale if scaled=True; note min alpha and beta is 0
    if scaled:
        val_a = (alpha_prime - alpha)**4
        val_b = (beta_prime - beta)**4

        val_a_scaled = val_a / max_alpha
        val_b_scaled = val_b / max_beta

        return scale_factor * (val_a_scaled + val_b_scaled)
    else:
        return mu * ((alpha_prime - alpha)**4 + (beta_prime - beta)**4)

# penalty with cliff and high repulsion
def absolute_cliff(
        mu,
        power,
        alpha,
        alpha_prime,
        beta_prime,
        alpha_epsilon = 0.01,
        beta_epsilon = 0.05,
        scaled = False,
        scale_factor = 20):
    
    # calculate beta from power
    beta = 1-power

    alpha_met = (-alpha_epsilon <= alpha_prime - alpha) & (alpha_prime - alpha <= 0)
    beta_met = (-beta_epsilon <= beta_prime - beta) & (beta_prime - beta <= 0)

    # find the maximum alpha and beta
    if alpha <= 0.5:
        max_alpha = abs(alpha - 1)
    else:
        max_alpha = abs(alpha)
    
    if beta <= 0.5:
        max_beta = abs(beta - 1)
    else:
        max_beta = abs(beta)

    if (alpha_met and beta_met):
        return 0
    elif scaled:
        val_a = abs(alpha_prime - alpha)
        val_b = abs(beta_prime - beta)

        val_a_scaled = val_a / max_alpha
        val_b_scaled = val_b / max_beta

        return scale_factor * (val_a_scaled + val_b_scaled)
    else:
        return mu * (abs(alpha_prime - alpha) + abs(beta_prime - beta))
