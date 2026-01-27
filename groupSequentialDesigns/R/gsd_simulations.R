#' Simulate a group sequential design for a clinical trial
#'
#' `gsd_simulations` simulates a group sequential design for a clinical trial.
#' It uses given upper and lower bounds, an assumed null and alternative
#' hypotheses for a mean difference between groups, an assumed variance for
#' these mean differences, the number of interim analyses, and sample sizes
#' (per group) to calculate the probabilities of stopping for futility or
#' efficacy under the null and alternative hypotheses at each analysis. It also
#' calculates the expected sample size, the variance of the sample size, and the
#' type I error (\eqn{\alpha}) and power (\eqn{1 - \beta}) where \eqn{\beta} is
#' the type II error.
#'
#' @param n_analyses Integer indicating the number of interim analyses
#' that will be performed in the group sequential design.
#' @param upper_bounds Value of the upper bounds to use for group sequential
#' design simulation
#' @param lower_bounds Value of the lower bounds to use for group sequential
#' design simulation
#' @param n_patients Vector of integers indicating the sample size (per group)
#' at each of the interim analyses. Note that the length of this vector must
#' correspond to the number of analyses `n_analyses`.
#' @param null_hypothesis The mean difference between groups under the null
#' hypothesis (default is 0).
#' @param alt_hypothesis The mean difference between groups under the
#' alternative hypothesis (default is 0).
#' @param variance The assumed variance in the mean values of the null and
#' alternative mean differences.
#'
#' @details Below is a mathematical description of how each of the values are
#' calculated.
#'
#' ## Stopping probabilities
#'
#' Assuming a normally distributed end point, responses for patients on control
#' treatment is \eqn{Y_{0i} \sim \mathcal{N}(\mu_0, \sigma^2)} and for patients
#' on experimental treatment is \eqn{Y_{1i} \sim \mathcal{N}(\mu_1, \sigma^2)}.
#'
#' Assume \eqn{\sigma} is known and the parameter of interest is \eqn{\theta = \mu_1 = \mu_0}.
#'
#' At analysis \eqn{j}, MLE \eqn{\hat{\theta}_j} is normally distributed with
#' mean \eqn{\theta} and variance \eqn{\frac{2\sigma^2}{n_j}} and the Wald test
#' statistic is \eqn{Z_j = \bar{Y}^{(j)}_1 -\bar{Y}^{(j)}_0}. The joint
#' distribution of each test statistic is multivariate normal.
#'
#' The following equation is used to calculate the probability of stopping for
#' futility **under the null** at Analysis 2. It is the lower tail of the
#' multivariate normal distribution up to the second lower bound (the selected
#' critical value).
#'
#' \deqn{
#'
#'  \int_{l_1}^{u_1}\int_{-\infty}^{l_2} \phi_2 \left( \begin{bmatrix}
#'  y_1 & y_2
#'  \end{bmatrix}, \begin{bmatrix}
#'  \theta \sqrt{\frac{n_1}{2\sigma^2}} & \theta \sqrt{\frac{n_2}{2\sigma^2}}
#'  \end{bmatrix}, \begin{bmatrix} 1 & \sqrt{\frac{n_1}{n_2}} \\ \sqrt{\frac{n_1}{n_2}} & 1 \end{bmatrix} \right) dy_2\,dy_1
#'
#' }
#'
#' ## Expected sample size
#'
#' The expected sample size is calculated across a set of possible true treatment
#' effects \eqn{\boldsymbol{\delta} = \{\delta_1, \delta_2, \dots, \delta_n \}} as
#' a function of the design elements (1) number of analyses \eqn{k = \{1, 2, \dots, K\}},
#' (2) number of patients at each analysis \eqn{\mathbf{n} = \{n_1, n_2, \dots, n_K\}},
#' (3) the upper and lower bounds \eqn{\mathbf{u} = (u_1, u_2, \dots, u_K)}
#' and \eqn{\boldsymbol{\ell} = (\ell_1, \ell_2, \dots, \ell_K)}.
#'
#' The expected sample size is:
#' \deqn{
#' E[N \, | \, \delta]=\sum_{i=1}^K n_i P(\text{trial stops after analysis }i \, | \, \delta)
#' }
#'
#' The probability that the trial stops after analysis $i$ is the sum of the
#' probabilities that the trial stops for efficacy after analysis $i$ and the
#' probability that the trial stops for futility after analysis $i$.
#'
#' \deqn{
#'  E[N \, | \, \delta]=\sum_{i=1}^K n_i \Big[ P(\text{trial stops for efficacy after analysis }i \, | \, \delta) + \\
#'  P(\text{trial stops for futility after analysis }i \, | \, \delta) \Big]
#' }
#'
#' ## Variance of sample size
#'
#' The definition of the variance for a random variable \eqn{X},
#'
#' \deqn{
#'   Var(X) = E[X]^2 - E[X^2]
#' }
#'
#' is used to calculate the variance for the expected sample size.
#'
#' ## Type I error
#'
#' The type I error for the design (\eqn{\alpha}) is calculated by summing the
#' probabilities for stopping for efficacy under the null hypothesis for each
#' analysis.
#'
#' ## Power
#'
#' The power for the design is calculated by summing the probabilities for
#' stopping for futility under the alternative to calculate the type II error
#' (\eqn{\beta}) and the subtracting this from 1.
#'
#'
#' @returns
#' A list of length (\eqn{n} + 4) where \eqn{n} is the number of analyses
#' specified by `n_analyses`.
#'
#' * For each analysis from 1 to \eqn{n} (`analysis_1` to `analysis_n`) 4
#' stopping probabilities are returned, the probability of stopping:
#'      1. for futility under the null.
#'      2. for efficacy under the null.
#'      3. for futility under the alternative.
#'      4. for efficacy under the alternative.
#'* The expected sample size (ESS).
#'* The variance in sample size (VSS).
#'* The type I error for the design, (\eqn{alpha}).
#'* The power, \eqn{1 - \beta}, where \eqn{\beta} is
#' the type II error.
#'
#' @export
#'
#'
#' @examples
#' gsd_simulations <- function(n_analyses = 3,
#'                             upper_bounds = c(2.5, 2, 1.5),
#'                             lower_bounds = c(0, 0.75, 1.5),
#'                             n_patients = c(20, 40, 60),
#'                             null_hypothesis = 0,
#'                             alt_hypothesis = 0.5,
#'                             variance = 1)

gsd_simulations <- function(n_analyses = 3,
                            upper_bounds = c(2.5, 2, 1.5),
                            lower_bounds = c(0, 0.75, 1.5),
                            n_patients = c(20, 40, 60),
                            null_hypothesis = 0,
                            alt_hypothesis = 0.5,
                            variance = 1) {

  # sanity checks
  # sanity checks, function stops
  if(length(upper_bounds) != length(lower_bounds)) {
    stop("Warning: number of upper bounds must equal number of lower bounds")
  }

  if(length(n_patients) != length(upper_bounds)) {
    stop("Warning: number of patients vector must equal number of bounds")
  }

  if(n_analyses != length(upper_bounds)) {
    stop("Warning: number of analyses must equal number of bounds")
  }

  # assign values for null and alt hypotheses
  theta_0 <- null_hypothesis
  delta <- alt_hypothesis

  # empty mean vectors to fill
  mean_0 <- c()
  mean_1 <- c()

  # need to parse the upper and lower boundaries of the design
  # for futility and efficacy, must put the bounds of integration correctly
  # for pmvnorm
  futility_l_bounds <- list()
  futility_u_bounds <- list()
  efficacy_l_bounds <- list()
  efficacy_u_bounds <- list()

  n_analyses <- length(upper_bounds)

  for (i in 1:n_analyses) {

    # special case of i = 1
    if (i == 1) {
      futility_l_bounds[[i]] <- lower_bounds[i]
      futility_u_bounds[[i]] <- upper_bounds[i]
      efficacy_l_bounds[[i]] <- lower_bounds[i]
      efficacy_u_bounds[[i]] <- upper_bounds[i]
      next
    }

    # all other cases
    futility_l_bounds[[i]] <- c(lower_bounds[1:i-1], -Inf)
    futility_u_bounds[[i]] <- c(upper_bounds[1:i-1], lower_bounds[i])

    efficacy_l_bounds[[i]] <- c(lower_bounds[1:i-1], upper_bounds[i])
    efficacy_u_bounds[[i]] <- c(upper_bounds[1:i-1], Inf)
  }

  # list of probabilities to return
  probs_to_return <- list()

  # list of SIGMAs
  SIGMA_list <- list()

  for (i in 1:n_analyses) {
    if (i == 1) next

    # start with diagonal matrix for SIGMA
    SIGMA <- diag(nrow = i)

    # n = 2, need to fill all but 11, 22
    # n = 3, need to fill all but 11, 22, 33
    # n = 4, need to fill all but 11, 22, 33, 44
    # etc.
    for(j in 1:i) {
      for(k  in 1:i) {

        # leave the 1s on the diagonal, skip this iteration of for loop
        if(j == k) next

        # when i is less than j, the lower number of patients will be in numerator
        if(j < k) SIGMA[j,k] <- sqrt(n_patients[j] / n_patients[k])

        # when i is greater than j, the lower number of patients will be in numerator
        if(j > k) SIGMA[j,k] <- sqrt(n_patients[k] / n_patients[j])

      }
    }

    SIGMA_list[[i]] <- SIGMA
  }


  for (i in 1:n_analyses) {

    ##############
    # ANALYSIS 1 #
    ##############
    if(i == 1) {
      # mean under null
      mean_0[i] <- theta_0 * sqrt(n_patients[i]/(2*variance))

      # mean under alternative
      mean_1[i] <- delta * sqrt(n_patients[i]/(2*variance))

      # prob stop for futility, null
      futility_null <- pnorm(futility_l_bounds[[i]],
                             mean = mean_0,
                             sd = sqrt(variance))

      # prob stop for efficacy, null
      efficacy_null <- pnorm(efficacy_u_bounds[[i]],
                             mean = mean_0,
                             sd = sqrt(variance),
                             lower.tail = FALSE)

      # prob stop for futility, alt
      futility_alt <- pnorm(futility_l_bounds[[i]],
                            mean = mean_1,
                            sd = sqrt(variance))

      # prob stop for efficacy
      efficacy_alt <- pnorm(efficacy_u_bounds[[i]],
                            mean = mean_1,
                            sd = sqrt(variance),
                            lower.tail = FALSE)

      probs_to_return[[i]] <- c(futility_null, efficacy_null, futility_alt, efficacy_alt)
      names(probs_to_return[[i]]) <- c("futility_null", "efficacy_null", "futility_alt", "efficacy_alt")

      next
    }

    ######################
    # ALL OTHER ANALYSES #
    ######################

    # next mean under null
    mean_0[i] <- theta_0 * sqrt(n_patients[i] / (2 * variance))

    # next mean under alternative
    mean_1[i] <- delta * sqrt(n_patients[i]/ (2*variance))

    # bounds for these will be same
    # futility under null
    futility_null <- pmvnorm(lower = futility_l_bounds[[i]],
                             upper = futility_u_bounds[[i]],
                             mean = mean_0, corr = SIGMA_list[[i]])
    # futility under alt
    futility_alt <- pmvnorm(lower = futility_l_bounds[[i]],
                            upper = futility_u_bounds[[i]],
                            mean = mean_1, corr = SIGMA_list[[i]])

    # bounds for these will be same
    # futility under null
    efficacy_null <- pmvnorm(lower = efficacy_l_bounds[[i]],
                             upper = efficacy_u_bounds[[i]],
                             mean = mean_0, corr = SIGMA_list[[i]])
    # efficacy under alt
    efficacy_alt <- pmvnorm(lower = efficacy_l_bounds[[i]],
                            upper = efficacy_u_bounds[[i]],
                            mean = mean_1, corr = SIGMA_list[[i]])

    probs_to_return[[i]] <- c(futility_null, efficacy_null, futility_alt, efficacy_alt)
    names(probs_to_return[[i]]) <- c("futility_null", "efficacy_null", "futility_alt", "efficacy_alt")

  }

  # vector to collect the sum of futility and efficacy probabilities
  sum_probs <- c()

  # get alpha and power
  alpha <- 0
  power <- 0

  for (i in 1:n_analyses) {

    # pull the probabilities from the list
    tmp_probs <- probs_to_return[[i]]

    # gather them into a vector
    # 3:4 because we want to calculate under the alternative
    sum_probs <- c(sum_probs, sum(tmp_probs[3:4]))

    alpha <- tmp_probs[2] + alpha
    power <- tmp_probs[4] + power

  }

  # calculate the expected sample size
  ess <- sum(n_patients * sum_probs)
  vss <- sum(n_patients^2 * sum_probs) - ess^2

  # add the expected sample size to the list
  return_values <- append(probs_to_return, values = c(ess, vss, alpha, power))

  # name the list
  names_for_list <- as.vector(sapply("analysis_", paste0, 1:n_analyses))
  names_for_list <- c(names_for_list, "expected_sample_size", "var_sample_size",
                      "alpha", "power")
  names(return_values) <- names_for_list

  # return probabilities and ESS
  return_values
}
