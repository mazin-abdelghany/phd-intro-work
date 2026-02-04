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
        alpha_prime,
        beta_prime,
        alpha,
        beta):
    if (alpha_prime > alpha) and (beta_prime > beta):
        return mu * ((alpha_prime - alpha)**2 + (beta_prime - beta)**2)
    return mu