import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bayesian optimization of group sequential designs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Motivating problem
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Introduction
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In Bayesian optimization, the goal is to optimize a blackbox function, $f$. In our case, this function takes inputs of

    $$
    D = \{n, u_1, \ell_1, u_2, \ell_2, \cdots, u_k, \ell_k \}
    $$

    where

    - $D$ is the design (an $n$-tuple set),
    - $n$ is the sample size at each analysis point (total for both groups),
    - $u_1$ and $\ell_1$ are the upper and lower bounds for the first stage, and
    - $k$ is the total number of stages

    and outputs several values of interest, $T$ (an $m$-tuple set), including (but not limited to)

    - $\alpha$, the type I error
    - $\beta$, the type II error
    - $\mathbb{E}[N \, | \, \boldsymbol{\delta}]$, the expected sample size (ESS) $N$ over a range of differences, $\delta$, between groups $\boldsymbol{\delta} = \{\delta_1, \delta_2, \dots, \delta_j \}$

    Above, $N = k*n$ and

    The expected sample size is calculated across a set of possible true treatment effects $\boldsymbol{\delta} = \{\delta_1, \delta_2, \dots, \delta_j \}$ as a function of the design elements $D$: (1) number of analyses $\{1, 2, \dots, k\}$, (2) number of patients at each analysis $\mathbf{n} = \{n_1, n_2, \dots, n_k\}$, and (3) the upper and lower bounds $\mathbf{u} = (u_1, u_2, \dots, u_k)$ and $\boldsymbol{\ell} = (\ell_1, \ell_2, \dots, \ell_k)$.

    It is calculated as

    $$
    \mathbb{E}[N \, | \, \boldsymbol{\delta}]=\sum_{i=1}^k n_i P(\text{trial stops after analysis }i \, | \, \boldsymbol{\delta})
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pseudocode for Bayesian optimization loop
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order to generated an optimized clinical trial design, the following steps must be followed:

    1. Generate several feasible trial designs, $\boldsymbol{D}=\{D_1, D_2, \cdots, D_p\}$.
    2. Generate the corresponding outputs, $\boldsymbol{T}=\{T_1, T_2, \cdots, T_p\}$.
    3. Fit a Gaussian process regression model to estimate the blackbox function $f: D \to T$.
    4. Perform a step of Bayesian optimization to find the next trial design of interest $D_i$.
    5. Obtain the corresponding outputs that correspond to this design $T_i$.
    6. Refit the Gaussian process regression model on the new data $n$-tuples $\{(\boldsymbol{D}, \boldsymbol{T}), (D_i, T_i)\}$

    Repeat until termination policy is reached.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Function to minimize
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As the above optimization problem applies to clinical trial designs, there are certain feasibility constraints that must be considered. Most importantly, the type I and type II error (or power)&mdash;$\alpha$, $\beta$ (or $1-\beta$),respectively&mdash;must be near the set nominal levels. In other words, if the design requires that a one-sided $\alpha = 0.025$, then feasible designs must have values of $\alpha$ near this value. In Wason et al. (Statist. Med. 2012, 31 301–312), feasible designs are defined as "design[s] for which the significance level and power meet the required constraints."

    Though Bayesian optimization can be constrained in such a manner, for simplicity, a penalty term will be included within the objective function $f$, which will take these constraints into account. Again, borrowing from Wason et al., the penalty term is:

    $$
    \mathcal{L} = \mu \cdot \left( \mathbb{I}_{\{\alpha' > \alpha\}}\cdot\frac{\alpha' - \alpha}{\alpha} + \mathbb{I}_{\{\beta' > \beta\}}\cdot\frac{\beta' - \beta}{\beta}  \right)
    $$

    where $\mu$ is the sample size for a one-stage design, $\alpha$ and $\beta$ are the set nominal values for type I and II error, respectively, $\alpha'$ and $\beta'$ are the type I and II errors for the new design, and $\mathbb{I}$ is the indicator function.

    The function that we aim to minimise, $f$, is the sum of the maximum expected sample size of the design and a penalty function that penalizes designs that are not considered feasible:

    $$
    f = \max\left\{\mathbb{E}[N \, | \, \boldsymbol{\delta}]\right\} + \mathcal{L}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementation of Bayesian optimization loop
    """)
    return


@app.cell
def _():
    # higher resolution graphs
    # magic command not supported in marimo; please file an issue to add support
    # %config InlineBackend.figure_format='retina'
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 1: Generate study designs
    """)
    return


@app.cell
def _():
    # imports for study design step (Step 1)
    import numpy as np
    import pandas as pd
    from scipy import stats

    # imports for GP regression (Step 3)
    import gpflow

    # imports for Bayes opt (Step 4-6)
    import trieste
    from trieste.space import Box
    from trieste.models.gpflow.models import GaussianProcessRegression
    import tensorflow as tf
    return Box, GaussianProcessRegression, gpflow, np, pd, stats, tf, trieste


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Test speed of multivariate integration vs. single variable integration
    """)
    return


@app.cell
def _(stats):
    mvn = stats.multivariate_normal(mean = [1, 2], cov = [[1, 0.5], [0.5, 1]])
    return (mvn,)


@app.cell
def _(mvn, np):
    mvn.cdf([2, np.inf], lower_limit=[-np.inf, -np.inf])
    return


@app.cell
def _(stats):
    stats.norm.cdf(2, 1, 1)
    return


@app.cell
def _(stats):
    # magic command not supported in marimo; please file an issue to add support
    # %%timeit
    stats.norm.cdf(2, 1, 1)
    return


@app.cell
def _(mvn, np):
    # magic command not supported in marimo; please file an issue to add support
    # %%timeit
    mvn.cdf([2, np.inf], lower_limit=[-np.inf, -np.inf])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Simulate group sequential designs function
    """)
    return


@app.cell
def _(np, pd, stats):
    # simulate the trials to obtain alpha and beta
    def simulate_group_sequential_designs(
            n_analyses=3,
            upper_bounds=[2.5, 2, 1.5],
            lower_bounds=[0, 0.75, 1.5],
            n_patients=20,
            null_hypothesis=0,
            alt_hypothesis=0.5,
            variance=1):

        # convert list to np ndarray
        upper_bounds = np.array(upper_bounds)
        lower_bounds = np.array(lower_bounds)

        # assign values for null and alt hypotheses
        theta_0 = null_hypothesis
        delta = alt_hypothesis

        # empty list to fill mean vectors
        mean_0 = []
        mean_1 = []

        # number of patients in each analysis
        n_patients_analysis = np.array([x for x in range(1, n_analyses + 1, 1)]) * n_patients

        # need to parse the upper and lower boundaries of the design
        # for futility and efficacy, must put the bounds of integration correctly
        # for pmvnorm
        futility_l_bounds = [[]]
        futility_u_bounds = [[]]
        efficacy_l_bounds = [[]]
        efficacy_u_bounds = [[]]

        n_analyses = len(upper_bounds)

        # loop through number of analyses
        for i in range(n_analyses):

            # special case of i = 1
            if i == 0:
                futility_l_bounds[i].append(-np.inf)
                futility_u_bounds[i].append(lower_bounds[i])
                efficacy_l_bounds[i].append(upper_bounds[i])
                efficacy_u_bounds[i].append(np.inf)
                continue

            # all other cases
            futility_l_bounds.append(np.append(lower_bounds[0:i], -np.inf))
            futility_u_bounds.append(np.append(upper_bounds[0:i], lower_bounds[i]))
            efficacy_l_bounds.append(np.append(lower_bounds[0:i], upper_bounds[i]))
            efficacy_u_bounds.append(np.append(upper_bounds[0:i], np.inf))

        # empty dictionary of SIGMA matrices
        SIGMA_dict = dict()

        # generate the SIGMA matrices
        for i in range(n_analyses):

            if i == 0: SIGMA_dict.update({i : np.sqrt(variance)})

            # start with diagonal matrix for SIGMA
            SIGMA = np.eye(N = i+1)

            # n = 2, need to fill all but 11, 22
            # n = 3, need to fill all but 11, 22, 33
            # n = 4, need to fill all but 11, 22, 33, 44
            # etc.
            for j in range(i+1):
                for k in range(i+1):

                    # leave the 1s on the diagonal, skip interation
                    if j == k: continue

                    # when j is less than k, the lower number of patients will be in numerator
                    if j < k: SIGMA[j,k] = np.sqrt(n_patients_analysis[j] / n_patients_analysis[k])

                    # when j is greater than j, the lower number of patients will be in numerator
                    if j > k: SIGMA[j,k] = np.sqrt(n_patients_analysis[k] / n_patients_analysis[j])

            SIGMA_dict.update({i : SIGMA})

        # empty data frame to collect probabilities
        # dictionary to DataFrame, column is the key:value pair of the dictionary
        probs_to_return = {
            "futility_null" : np.empty(n_analyses),
            "efficacy_null" : np.empty(n_analyses),
            "futility_alt" : np.empty(n_analyses),
            "efficacy_alt" : np.empty(n_analyses)
        }

        # generate the row names
        row_names = ["analysis_" + str(row_names) for row_names in range(1, n_analyses + 1, 1)]

        # create the empty data frame with data and index
        probs_to_return = pd.DataFrame(data = probs_to_return, index = row_names)

        # start calculations for the analyses
        for i in range(n_analyses):

            # mean under null
            mean_0.append(theta_0 * np.sqrt(n_patients_analysis[i] / (2 * variance)))

            # mean under alternative
            mean_1.append(delta   * np.sqrt(n_patients_analysis[i] / (2 * variance)))

            # generate the null and alt multivariate normal
            mvn_null = stats.multivariate_normal(mean = mean_0, cov = SIGMA_dict[i])
            mvn_alt  = stats.multivariate_normal(mean = mean_1, cov = SIGMA_dict[i])

            # prob stop for futility under null
            futility_null = mvn_null.cdf(futility_u_bounds[i], lower_limit = futility_l_bounds[i])

            # prob stop for futility under alt
            futility_alt = mvn_alt.cdf(futility_u_bounds[i], lower_limit = futility_l_bounds[i])

            # prob stop for efficacy under null
            efficacy_null = mvn_null.cdf(efficacy_u_bounds[i], lower_limit = efficacy_l_bounds[i])

            # prob stop for efficacy under alt
            efficacy_alt = mvn_alt.cdf(efficacy_u_bounds[i], lower_limit = efficacy_l_bounds[i])

            # add them to the data frame to return
            probs_to_return.iloc[i] = [futility_null, efficacy_null, futility_alt, efficacy_alt]

        # get the type I error (alpha)
        alpha = probs_to_return.sum(axis = 0)["efficacy_null"]

        # get the power (1 - beta)
        power = probs_to_return.sum(axis = 0)["efficacy_alt"]

        # get expected sample size
        summed_probs = probs_to_return.iloc[:,2:4].sum(axis=1)
        expected_sample_size = np.sum(summed_probs * n_patients_analysis)

        return probs_to_return, alpha, power, expected_sample_size
    return (simulate_group_sequential_designs,)


@app.cell
def _(simulate_group_sequential_designs):
    simulate_group_sequential_designs(
        n_analyses=3,
        upper_bounds=[2.5, 2, 1.5],
        lower_bounds=[0, 0.75, 1.5],
        n_patients=[20],
        null_hypothesis=0,
        alt_hypothesis=0.5,
        variance=1
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Functions for 3 different study designs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Pocock boundaries function
    """)
    return


@app.cell
def _(np, simulate_group_sequential_designs):
    # Pocock boundaries
    def calculate_pocock_boundaries(n_analyses=3, alpha=0.05, n_patients=20, sided='one.sided'):
        epsilon = 1e-08
        ub1 = np.repeat(0, repeats=n_analyses)
        lb1 = -np.repeat(0, repeats=n_analyses)
        ub2 = np.repeat(10, repeats=n_analyses)
        lb2 = -np.repeat(10, repeats=n_analyses)
        _, sim_alpha, _, _ = simulate_group_sequential_designs(n_analyses=n_analyses, upper_bounds=ub1, lower_bounds=lb1, n_patients=n_patients)  # the precision of the estimate for alpha
        while abs(sim_alpha - alpha) > epsilon:
            mid_u = (ub1 + ub2) / 2
            mid_l = -mid_u  # starting first set of bounds
            probs, sim_alpha, power, _ess = simulate_group_sequential_designs(n_analyses=n_analyses, upper_bounds=mid_u, lower_bounds=mid_l, n_patients=n_patients)
            if sim_alpha > alpha:
                ub1 = mid_u
                lb1 = mid_l  # starting second set of bounds
            else:
                ub2 = mid_u
                lb2 = mid_l
        if sided == 'one.sided':  # first alpha calcuation
            return [ub1, np.append(lb1[0:n_analyses - 1], ub1[n_analyses - 1]), probs, sim_alpha, power, _ess]
        else:
            return [ub1, lb1, probs, sim_alpha, power, _ess]  # calculate the first midpoint  # calculate the simulated alpha
    return (calculate_pocock_boundaries,)


@app.cell
def _(calculate_pocock_boundaries):
    calculate_pocock_boundaries()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### O'Brien-Fleming boundaries function
    """)
    return


@app.cell
def _(np, simulate_group_sequential_designs):
    # O'Brien-Fleming boundaries
    def calculate_of_boundaries(n_analyses=3, alpha=0.05, n_patients=20, sided='one.sided'):
        epsilon = 1e-08
        ub1 = np.repeat(0, repeats=n_analyses)
        lb1 = -np.repeat(0, repeats=n_analyses)
        ub2 = np.repeat(10, repeats=n_analyses)
        lb2 = -np.repeat(10, repeats=n_analyses)
        of_u1 = ub1 * (np.arange(1, n_analyses + 1, 1) / n_analyses) ** (-0.5)  # the precision of the estimate for alpha
        of_l1 = -of_u1
        _, sim_alpha, _, _ = simulate_group_sequential_designs(n_analyses=n_analyses, upper_bounds=of_u1, lower_bounds=of_l1, n_patients=n_patients)
        while abs(sim_alpha - alpha) > epsilon:  # starting first set of bounds
            mid_u = (ub1 + ub2) / 2
            mid_l = -mid_u
            mid_of_u = mid_u * (np.arange(1, n_analyses + 1, 1) / n_analyses) ** (-0.5)
            mid_of_l = -mid_of_u  # starting second set of bounds
            probs, sim_alpha, power, _ess = simulate_group_sequential_designs(n_analyses=n_analyses, upper_bounds=mid_of_u, lower_bounds=mid_of_l, n_patients=n_patients)
            if sim_alpha > alpha:
                ub1 = mid_u
                lb1 = mid_l  # of bounds
            else:
                ub2 = mid_u
                lb1 = mid_l
        if sided == 'one.sided':  # first alpha calcuation
            return [mid_of_u, np.append(mid_of_l[0:n_analyses - 1], mid_of_u[n_analyses - 1]), probs, sim_alpha, power, _ess]
        else:
            return [mid_of_u, mid_of_l, probs, sim_alpha, power, _ess]  # calculate the first midpoint  # convert to O'Brien-Fleming bounds  # calculate the simulated alpha
    return (calculate_of_boundaries,)


@app.cell
def _(calculate_of_boundaries):
    calculate_of_boundaries()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Triangular boundaries function
    """)
    return


@app.cell
def _(np):
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
    return (calculate_triangular_boundaries,)


@app.cell
def _(calculate_triangular_boundaries):
    calculate_triangular_boundaries()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 2: Simulate the trials to obtain $\alpha$, $\beta$, maximum ESS
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are two sets of variables important in determining maximum expected sample size: (1) the difference, $\delta$, of interest between the groups of the trial (e.g., null hypothesis $\delta_0=0$ and alternative hypothesis $\delta_1=0.5$ implies a difference of interest of 0.5 between the two groups) and (2) the design of the trial (i.e., the selected type I error rate, $\alpha$, type II error rate, $\beta$, and boundaries). Though the boundaries determine the type I error rate, $\alpha$, there is an infinite set of boundaries that meet this condition.

    There are further dependencies that determine the type II error rate, $\beta$, namely the sample size for the trial. Typically, the larger the sample size, the smaller the type II error rate (or conversely, the higher the power, $1-\beta$). The variance of the selected measure also affects the type II error rate. As the variance increases, there will be a smaller standardized effect size ($\frac{\delta}{\sigma^2}$, and therefore a larger sample size will be required to obtain the same $\beta$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="/tf/2026-01-t09/maximum-expected-sample-size-dependencies.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Because of these dependencies, Wason et al. 2011 fix the following:
    - Type I error rate: $\alpha=0.05$
    - Type II error rate: $\beta=0.1$
    - Null difference: $\delta_0=0$
    - Alternative difference: $\delta_1=1$
        - Implying difference of interest: $\delta=1$
    - Standard deviation: $\sigma=3$ (i.e., variance, $\sigma^2=9$)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Sample size calculator function
    """)
    return


@app.cell
def _(stats):
    # sample size per group for a difference in means
    def sample_size_means(
            ratio=1,
            variance=1,
            power=0.8,
            alpha=0.05,
            delta=1):

        # ratio of smaller group to larger group
        r = (ratio+1)/ratio

        # z statistic for power
        z_power = stats.norm.ppf(power)

        # z statistic for alpha
        z_alpha = abs(stats.norm.ppf(alpha))

        # sample size
        n = r * ((variance**2 * (z_power+z_alpha)**2) / delta**2)

        return n
    return (sample_size_means,)


@app.cell
def _(sample_size_means):
    sample_size_means(variance=3, power=0.9, alpha=0.05)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Maximum expected sample size function
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Create a function that calculates the maximum expected sample size using interval bisection.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How do I make sure that the search space (`delta_start` and `delta_stop`) include the maximum sample size?
    """)
    return


@app.cell
def _(simulate_group_sequential_designs):
    # obtain maximum expected sample size for an interval of interest
    def max_ess(
            delta_start=0,
            n_analyses=3,
            upper_bounds=[2.5, 2, 1.5],
            lower_bounds=[0, 0.75, 1.5],
            n_patients=20,
            null_hypothesis=0,
            variance=1):

        # epsilon precision for max ESS calculated
        epsilon = 1e-4

        # delta_stop based on the variance (5x the variance)
        delta_stop = variance * 5

        # random starting values of ess
        ess_delta_start = 10
        ess_delta_stop = 0

        # while the error is greater than desired precision
        while abs(ess_delta_start - ess_delta_stop) > epsilon:

            # simulate the trial under delta_start
            probs, sim_alpha, power, ess_delta_start = simulate_group_sequential_designs(
                n_analyses = n_analyses,
                upper_bounds = upper_bounds,
                lower_bounds = lower_bounds,
                n_patients = n_patients,
                null_hypothesis = null_hypothesis,
                alt_hypothesis = delta_start,
                variance = variance
            )

            # simulate trial under delta_stop
            probs, sim_alpha, power, ess_delta_stop = simulate_group_sequential_designs(
                n_analyses = n_analyses,
                upper_bounds = upper_bounds,
                lower_bounds = lower_bounds,
                n_patients = n_patients,
                null_hypothesis = null_hypothesis,
                alt_hypothesis = delta_stop,
                variance = variance
            )

            if ess_delta_start >= ess_delta_stop:
                delta_stop = (delta_start + delta_stop)/2 
            else:
                delta_start = (delta_start + delta_stop)/2

        return ess_delta_start
    return (max_ess,)


@app.cell
def _(max_ess):
    max_ess()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check to make sure this is the correct answer based on a grid search.
    """)
    return


@app.cell
def _(np, simulate_group_sequential_designs):
    deltas = np.linspace(start=0, stop=3, num=1000)
    ess_list = np.empty(1000)
    for i, delta in enumerate(deltas):
        probs, sim_alpha, power, ess = simulate_group_sequential_designs(
            alt_hypothesis=delta
        )
        ess_list[i] = ess
    ess_list.max()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Sample size finder function
    """)
    return


@app.cell
def _(simulate_group_sequential_designs):
    def find_sample_size(power_target=0.9, n_analyses=3, upper_bounds=[2.5, 2, 1.5], lower_bounds=[0, 0.75, 1.5], null_hypothesis=0, alt_hypothesis=0.5, variance=1):
        epsilon = 1e-08
        n_patients_left = 1
        n_patients_right = 1000000.0
        probs, sim_alpha, power, _ess = simulate_group_sequential_designs(n_analyses=n_analyses, upper_bounds=upper_bounds, lower_bounds=lower_bounds, n_patients=n_patients_left, null_hypothesis=null_hypothesis, alt_hypothesis=alt_hypothesis, variance=variance)
        while abs(power - power_target) > epsilon:
            n_patients_mid = (n_patients_left + n_patients_right) / 2
            probs, sim_alpha, power, _ess = simulate_group_sequential_designs(n_analyses=n_analyses, upper_bounds=upper_bounds, lower_bounds=lower_bounds, n_patients=n_patients_mid, null_hypothesis=null_hypothesis, alt_hypothesis=alt_hypothesis, variance=variance)
            if power > power_target:
                n_patients_right = n_patients_mid  # precision to get to power
            else:
                n_patients_left = n_patients_mid
        return [n_patients_left, power]  # initial sample sizes  # calcuate first power  # interval bisection loop  # generate the midpoint  # calcuate power
    return (find_sample_size,)


@app.cell
def _(find_sample_size):
    find_sample_size(power_target=0.7)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Feasibility penalty function
    """)
    return


@app.cell
def _(sample_size_means):
    # generate the penalty term
    def feasibility_penalty(
            ratio=1, 
            variance=1, 
            power=0.8, 
            alpha=0.05, 
            delta=1, 
            beta_prime=0.7, 
            alpha_prime=0.1):
    
        mu = sample_size_means(
            ratio=ratio, 
            variance=variance, 
            power=power, 
            alpha=alpha, 
            delta=delta
        )

        def alpha_indicator(alpha_prime):
            if alpha_prime > alpha:
                return 1
            return 0

        def beta_indicator(beta_prime):
            if beta_prime > power:  
                return 1
            return 0
        
        return mu * ((alpha_prime - alpha) / alpha * alpha_indicator(alpha_prime) + (beta_prime - power) / power * beta_indicator(beta_prime))  # create the indicator functions
    return (feasibility_penalty,)


@app.cell
def _(feasibility_penalty):
    feasibility_penalty()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 3: Fit a Gaussian process regression model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, we need to create a function that will output the single value to minimize.
    """)
    return


@app.function
# function to minimize
def function_to_minimize(max_ess_val, penalty):
    return max_ess_val + penalty


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### What are the inputs and outputs for GP regression and Bayes opt?

    **Fixed inputs:**
    - Type I error rate: $\alpha=0.05$
    - Type II error rate: $\beta=0.1$
    - Null difference: $\delta_0=0$
    - Alternative difference: $\delta_1=1$
        - Implying difference of interest: $\delta=1$
    - Standard deviation: $\sigma=3$ (i.e., variance, $\sigma^2=9$)

    **Variable inputs:**
    - Maximum sample size
    - Feasibility penalty
    - Boundary values
    - Number of analyses, $k$

    **Outputs**:
    - New boundary values (from Bayes opt)
    - New $\alpha := \alpha'$
    - New $\beta := \beta'$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Pseudocode

    1. Start with feasible trial design boundaries.
    2. Using the fixed inputs, generate the maximum expected sample size and feasibility penalty.
    3. Calculate the function value, $f$, using the maximum expected sample size and feasibility penalty.
    4. Into the Gaussian process (GP) regression model, the design boundaries are inputs and the function values are outputs.
    5. Using the GP regression model, perform one iteration of Bayesian optimization to find the next input values for the Gaussian process.
    6. Using the result of Bayesian optimization (new boundary values), calculate:
        - $\alpha'$
        - $\beta'$
        - maximum expected sample size
        - feasibility penalty
        - new function value $f$
    7. Feed the input (boundary values) and output (function value) into the GP regression model.
    8. Obtain new design boundaries.
    9. Repeat until termination policy reached.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Input dimensions

    In order to fit the multidimensional Gaussian process regression, it is important to know how many input dimensions exist for the problem at hand. For a one-sided statistical test, there will be $2k-1$ unique boundaries (as the last boundary value is equal) and a sample size scaled by each stage (e.g., $n=20$ with $k=3$ stages implies $n_1=20, n_2=40, n_3=60$) for a total of $2k$ inputs.
    """)
    return


@app.cell
def _(np):
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
    return (generate_gpr_input,)


@app.cell
def _(
    calculate_pocock_boundaries,
    feasibility_penalty,
    find_sample_size,
    generate_gpr_input,
    max_ess,
):
    ################
    # GPR workflow #
    ################

    num_analyses = 3
    target_alpha = 0.05

    # some set defaults
    target_power = 0.9
    important_diff_delta = 1
    assumed_variance = 3
    group_ratio = 1

    simulation = calculate_pocock_boundaries(
        n_analyses=num_analyses, 
        alpha=target_alpha, 
        n_patients=20
    )

    upper = simulation[0]
    lower = simulation[1]
    alpha_prime = simulation[3]

    # simulate the trial design 
    n_power09, beta_prime = find_sample_size(
        n_analyses=num_analyses, 
        upper_bounds=upper, 
        lower_bounds=lower, 
        alt_hypothesis=important_diff_delta, 
        variance=assumed_variance
    )

    x1 = generate_gpr_input(
        n_analyses=num_analyses, 
        upper_bounds=upper, 
        lower_bounds=lower, 
        n_patients=n_power09
    )

    max_ess_new = max_ess(
        n_analyses=num_analyses, 
        upper_bounds=upper, 
        lower_bounds=lower, 
        n_patients=n_power09
    )

    penalty = feasibility_penalty(
        ratio=group_ratio, 
        variance=assumed_variance, 
        power=target_power, 
        alpha=target_alpha, 
        delta=important_diff_delta, 
        beta_prime=beta_prime, 
        alpha_prime=alpha_prime
    )

    # pull inputs to the below functions from simulation
    # 1. Find the number of patients that achieves 90% power (beta 0.1)
    # here we get beta_prime and alpha prime
    # 2. Generate the GPR input values
    # note that the input includes the sample size at power 0.9
    # 3. Generate maximum expected sample size and feasibility penalty
    # 4. Calculate the function value (GPR output)
    y1 = function_to_minimize(max_ess_val=max_ess_new, penalty=penalty)
    return (
        assumed_variance,
        beta_prime,
        group_ratio,
        important_diff_delta,
        num_analyses,
        target_alpha,
        target_power,
        x1,
        y1,
    )


@app.cell
def _(
    assumed_variance,
    beta_prime,
    calculate_of_boundaries,
    feasibility_penalty,
    find_sample_size,
    generate_gpr_input,
    group_ratio,
    important_diff_delta,
    max_ess,
    num_analyses,
    target_alpha,
    target_power,
):
    simulation2 = calculate_of_boundaries(
        n_analyses=num_analyses, 
        alpha=target_alpha, 
        n_patients=20
    )

    upper2 = simulation2[0]
    lower2 = simulation2[1]
    alpha_prime_1 = simulation2[3]

    n_power09_2, beta_prime2 = find_sample_size(
        n_analyses=num_analyses, 
        upper_bounds=upper2, 
        lower_bounds=lower2, 
        alt_hypothesis=important_diff_delta, 
        variance=assumed_variance
    )

    x2 = generate_gpr_input(
        n_analyses=num_analyses, 
        upper_bounds=upper2, 
        lower_bounds=lower2, 
        n_patients=n_power09_2
    )

    max_ess_new2 = max_ess(
        n_analyses=num_analyses,
        upper_bounds=upper2,
        lower_bounds=lower2,
        n_patients=n_power09_2
    )

    penalty2 = feasibility_penalty(
        ratio=group_ratio, 
        variance=assumed_variance, 
        power=target_power, 
        alpha=target_alpha, 
        delta=important_diff_delta, 
        beta_prime=beta_prime, 
        alpha_prime=alpha_prime_1
    )

    y2 = function_to_minimize(max_ess_val=max_ess_new2, penalty=penalty2)
    return x2, y2


@app.cell
def _(x1):
    x1
    return


@app.cell
def _(x2):
    x2
    return


@app.cell
def _(y1):
    y1
    return


@app.cell
def _(y2):
    y2
    return


@app.cell
def _(np, x1, x2):
    X = np.concatenate((x1,x2)).reshape(2,6)
    return (X,)


@app.cell
def _(X):
    X
    return


@app.cell
def _(np, y1, y2):
    Y = np.concatenate(([y1], [y2])).reshape(2,1)
    return (Y,)


@app.cell
def _(Y):
    Y
    return


@app.cell
def _(GaussianProcessRegression, gpflow, tf):
    def build_model(X, Y, kernel_func=None):
        variance = tf.math.reduce_variance(X)

        if kernel_func is None:
            kernel = gpflow.kernels.Matern52(variance=variance)
        else:
            kernel = kernel_func(variance)

        gpr = gpflow.models.GPR(
            data = (X, Y),
            # setting the kernel variance to something other than 1 breaks the optimization
            # likely because things are not standardized/scaled?
            kernel = kernel_func(variance=1),
            likelihood = gpflow.likelihoods.Gaussian()
        )

        gpflow.utilities.print_summary(gpr, fmt="notebook")

        return GaussianProcessRegression(gpr)
    return (build_model,)


@app.cell
def _(X, Y, build_model, gpflow):
    # GP regression
    model = build_model(X = X, Y = Y, kernel_func=gpflow.kernels.SquaredExponential)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Steps 4-6: Bayesian optimization loop
    """)
    return


@app.cell
def _(X, Y, trieste):
    # create a dataset that works well with trieste
    initial_data = trieste.data.Dataset(query_points=X, observations=Y)
    return (initial_data,)


@app.cell
def _(initial_data):
    initial_data
    return


@app.cell
def _(Box):
    # create the search space using trieste Box function
    search_space = Box([-20, -20, -20, -20, -20, 0], [20, 20, 20, 20, 20, 1000])
    return (search_space,)


@app.cell
def _(search_space):
    search_space
    return


@app.cell
def _(initial_data, model, search_space, trieste):
    ask_tell = trieste.ask_tell_optimization.AskTellOptimizer(search_space = search_space,
                                                              datasets = initial_data, 
                                                              models = model)
    return (ask_tell,)


@app.cell
def _(ask_tell):
    results = ask_tell.ask()
    return (results,)


@app.cell
def _(results):
    results
    return


@app.cell
def _(np):
    def format_boundaries_after_ask(result):
        upper_bounds = np.array(result[0][0:3])
        lower_bounds = np.concatenate((result[0][3:5], [result[0][2]]))
        n_patients = np.array(result[0][5])

        return [
            upper_bounds,
            lower_bounds,
            n_patients
        ]
    return (format_boundaries_after_ask,)


@app.cell
def _(format_boundaries_after_ask, results):
    output = format_boundaries_after_ask(results)
    return (output,)


@app.cell
def _(output):
    output
    return


@app.cell
def _(
    assumed_variance,
    important_diff_delta,
    num_analyses,
    output,
    simulate_group_sequential_designs,
):
    (sim_results, alpha_prime_2, 
     sim_beta_prime, sim_ess) = simulate_group_sequential_designs(
        n_analyses=num_analyses, 
        upper_bounds=output[0], 
        lower_bounds=output[1], 
        n_patients=output[2], 
        alt_hypothesis=important_diff_delta, 
        variance=assumed_variance
     )
    return (alpha_prime_2,)


@app.cell
def _(
    alpha_prime_2,
    assumed_variance,
    feasibility_penalty,
    find_sample_size,
    generate_gpr_input,
    group_ratio,
    important_diff_delta,
    max_ess,
    num_analyses,
    output,
    target_alpha,
    target_power,
):
    # 1. Find the number of patients that achieves 90% power (beta 0.1)
    # here we get beta_prime and alpha prime
    sim_n_power09, sim_beta_prime2 = find_sample_size(
        n_analyses=num_analyses, 
        upper_bounds=output[0], 
        lower_bounds=output[1], 
        alt_hypothesis=important_diff_delta, 
        variance=assumed_variance
    )

    # 2. Generate the GPR input values
    # note that the input includes the sample size at power 0.9
    x3 = generate_gpr_input(
        n_analyses=num_analyses, 
        upper_bounds=output[0], 
        lower_bounds=output[1], 
        n_patients=sim_n_power09
    )

    # 3. Generate maximum expected sample size and feasibility penalty
    x3_max_ess_new = max_ess(
        n_analyses=num_analyses, 
        upper_bounds=output[0], 
        lower_bounds=output[1], 
        n_patients=sim_n_power09
    )

    x3_penalty = feasibility_penalty(
        ratio=group_ratio, 
        variance=assumed_variance, 
        power=target_power, 
        alpha=target_alpha, 
        delta=important_diff_delta, 
        beta_prime=sim_beta_prime2, 
        alpha_prime=alpha_prime_2
    )

    y3 = function_to_minimize(max_ess_val=x3_max_ess_new, penalty=x3_penalty)
    return x3, y3


@app.cell
def _(x3):
    x3
    return


@app.cell
def _(y3):
    y3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Questions answered

    ### Are the diagonals of the covariance matrix supposed to be the variance? Or are they always 1?

    The diagonals of the covariance matrix are the correlations between the sample size of the particular stage and itself, which is $\sqrt{\frac{n_i}{n_i}}$, which will always be 1. The assumed variance for the trial design affects $n_i$, which will affect the correlation between stages, but not between each stage and itself.

    ### Why does the single stage design sample size vary from expected calcuation?

    I was calculating the two-sided test ($\frac{\alpha}{2}$), but we are comparing a one-sided test ($\alpha$).

    ### How do I make sure that the search space (`delta_start` and `delta_stop`) include the maximum sample size?

    Using the variance, set the `delta_stop` as 4-5 times the variance. This should ensure that the search space will include the maximum expected sample size.

    ### The maximum expected sample size is at a different delta than the alternative for finding the correct beta?

    The authors were optimizing for "worst-case scenario".

    ### The above loop finds the boundaries then adjusts the sample size to obtain the desired power. In the paper, the boundaries are optimized to obtain the target power without modifying sample size?

    This is a question open for debate!
    """)
    return


if __name__ == "__main__":
    app.run()
