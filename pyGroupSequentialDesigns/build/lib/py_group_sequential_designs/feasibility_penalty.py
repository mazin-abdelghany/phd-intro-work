from . import sample_size as ss

# generate the penalty term
def feasibility_penalty(
        ratio=1,
        variance=1,
        power=0.8,
        alpha=0.05,
        delta=1,
        beta_prime=0.7,
        alpha_prime=0.1):

    # calculate the sample size for one-stage design
    mu = ss.sample_size_means(
        ratio=ratio,
        variance=variance,
        power=power,
        alpha=alpha,
        delta=delta
    )

    # create the indicator functions
    def alpha_indicator(alpha_prime):
        if alpha_prime > alpha: return 1
        return 0

    def beta_indicator(beta_prime):
        if beta_prime > power: return 1
        return 0

    return mu * ( (((alpha_prime-alpha)/alpha)*alpha_indicator(alpha_prime)) + (((beta_prime-power)/power)*beta_indicator(beta_prime)) )