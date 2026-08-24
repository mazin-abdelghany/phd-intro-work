library(gsDesign)

obj_f <- function(
    mu,
    upper_bounds,
    lower_bounds,
    n_analyses,
    n_patients,
    target_power,
    target_alpha,
    null_hypothesis,
    alt_hypothesis,
    variance
)
{
  # standardize the input
  theta0 <- null_hypothesis / (sqrt(2 * variance))
  theta1  <- alt_hypothesis / (sqrt(2 * variance))
  
  n_info = n_patients * c(1:3)
  
  # simulate the trial
  trial_probs <- gsProbability(
    k = n_analyses,
    theta = c(theta0, theta1),
    n.I = n_info,
    a = lower_bounds,
    b = upper_bounds,
    r = 79 # higher accuracy, more r points as in Jennison
  )
  
  # alpha is probability of rejecting null when it is true
  # that is, efficacy under the null, exiting upper bounds under null
  alpha_prime = sum(trial_probs$upper$prob[,1])
  
  # efficacy under alternative, exiting upper bounds under alt
  power_prime = sum(trial_probs$upper$prob[,2])
  beta_prime = 1 - power_prime
  
  # resimulate the trial under many theta and get max
  get_max_ess <- gsProbability(
    k = n_analyses,
    theta = seq(-0.5 * theta1, 1.5 * theta1, length.out = 500),
    n.I = n_info,
    a = lower_bounds,
    b = upper_bounds,
    r = 79 # higher accuracy, more r points as in Jennison
  )
  
  mess <- max(get_max_ess$en)
  
  # calculate the penalty
  target_beta = 1 - target_power
  penalty_term1 = mu * ((alpha_prime - target_alpha)**2 + (beta_prime - target_beta)**2)
  penalty_term2 = mess/mu
  
  # total penalty
  penalty = penalty_term1 + penalty_term2
  
  return_vals <- c(alpha_prime, power_prime, mess, penalty)
  names(return_vals) <- c("alpha'", "power'", "mESS", "obj func")
  
  return(
    return_vals
  )
  
}


upper_bounds <- c(1.92954495508180470,1.92954495508180470,1.67438418298127730)
lower_bounds <- c(0.08765734460384889,0.95156077374888060,1.67438418298127730)
mu <- 154

obj_f(mu = mu,
      upper_bounds = upper_bounds,
      lower_bounds = lower_bounds,
      n_analyses = 3,
      n_patients = 57.724780532,
      target_power = 0.9,
      target_alpha = 0.05,
      null_hypothesis = 0,
      alt_hypothesis = 1,
      variance = 9)

gsProbability(
  k = 3,
  theta = seq(-0.5 * 0, 1.5 * 2, length.out = 500),
  n.I = n_info,
  a = lower_bounds,
  b = upper_bounds,
  r = 79 # higher accuracy, more r points as in Jennison
)
