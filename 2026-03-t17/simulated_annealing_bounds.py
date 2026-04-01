import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pseudocode for simulated annealing

    Loop $n_\texttt{generate}$ times:
    1. Start with a an initial design $D$ and set first evaluation of objective function $f$ to $f_\texttt{min}$
    2. Generate a candidate design, $D'$.
    3. Calculate the objective function value for this design, $f'$.
    4. Increment the counter for the number of candidate designs generated.
    5. Perform simulated annealing:
       - Generate a uniform (0,1) random variable $x$ where $X\sim U(0,1)$.
       - <span style="color:red"><b>IF</span></b> $\exp\left\{-\frac{(f' - f)}{T}\right\}\ge x$:
            - Set $f'$ to $f$
            - Reduce $T$ by $\rho_{\texttt{cost}}$, calculated as $T\cdot\rho_{\texttt{cost}}$
            - Set the generated candidate design $D'$ to the current design $D$
            - <span style="color:red"><b>IF</span></b> $f' < f_{\texttt{min}}$:
                - Save the current design $D$ as the "best" design $D_{\texttt{min}}$
                - Set $f_{\texttt{min}}$ to $f'$
                - Reset a counter of number of loops since objective function reduction to 0
            - <span style="color:red"><b>ELSE</span></b> increment number since objective function reduction
        - <span style="color:red"><b>ELSE</span></b> increment number since objective function reduction

    After $n_{\texttt{generate}}$ times:
    1. Set $D$ to $D_{\texttt{min}}$
    2. Set $f$ to $f_{\texttt{min}}$
    3. Reset $T$ to its starting value
    5. Reset $n_{\texttt{generate}}$ to 0
    6. Increment number of restarts
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - $D=\{\texttt{upper\_1}, \dots, \texttt{upper\_k},\texttt{lower\_1}\dots, \texttt{lower\_{(k-1)}},n\}$
    - $f(\cdot)$ is

    $$
    \mathcal{L}(D,\mu) =\mu (\alpha' - \alpha)^2 + \mu(\beta' - \beta)^2 + \frac{\max\big(\mathbb{E}[N \, | \, \boldsymbol{\delta}]\big)}{\mu}
    $$
    - $T=100$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


@app.cell
def _():
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt

    return np, plt, stats


@app.cell
def _():
    # group sequential design assessment imports
    from py_group_sequential_designs import generate_boundaries as bd
    from py_group_sequential_designs import feasibility_penalty as fp
    from py_group_sequential_designs import boundary_manipulations as fmt_bd
    from py_group_sequential_designs import function_to_minimize as fn_min
    from py_group_sequential_designs import generate_gpr_input as gen_input
    from py_group_sequential_designs import simulate as sim
    from py_group_sequential_designs import sample_size as ss

    return bd, fmt_bd, fn_min, fp, sim, ss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Boundary generator function
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Boundary generator 1
    """)
    return


@app.cell
def _(fmt_bd, np, rng, stats):
    def boundary_generator1(
            n_analyses,
            sample_size_samples
    ):
        lower_bounds = np.zeros(n_analyses)
        upper_bounds = np.zeros(n_analyses)

        n = rng.choice(sample_size_samples)

        while not (
            # monotonic
            fmt_bd.check_monotonicity(
                bounds = np.concatenate((upper_bounds, lower_bounds)),
                n_analyses = n_analyses) and

            # first bounds between -8 and 8
            (lower_bounds[0] > -9) and
            (upper_bounds[0] < 9)
        ):

            # first set of bounds
            # lower_bounds[0] = stats.gennorm.rvs(size=1, beta=0.388, loc=-2.37, scale=0.084)[0]
            # lower_bounds[0] = -1*stats.lognorm.rvs(size=1, s=0.5175, loc=-1.765, scale=3.98)[0]
            # lower_bounds[0] = -1*stats.lognorm.rvs(size=1, s=0.6189, loc=-1.185, scale=3.04)[0]
            lower_bounds[0] = -1*stats.truncnorm.rvs(size=1, a=-0.20909, b=2.544, loc=0.063, scale=3.364)[0]
            upper_bounds[0] = stats.lognorm.rvs(size=1, s=1.12, loc=1.6, scale=1.31)[0]

            for _i in range(n_analyses-1):
                lower_bounds[_i+1] = lower_bounds[_i] + stats.expon.rvs(size=1, loc=0.0488, scale=0.693)[0]
                upper_bounds[_i+1] = upper_bounds[_i] - stats.expon.rvs(size=1, loc=0.000502, scale=0.27)[0]

            lower_bounds = np.concatenate((lower_bounds[0:2], np.array(upper_bounds[2], ndmin=1)))

        return (
            [upper_bounds, lower_bounds],
            n
        )

    return (boundary_generator1,)


@app.cell
def _(boundary_generator1, rounded_sample_size):
    boundary_generator1(3, rounded_sample_size)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Boundary generator 2
    """)
    return


@app.cell
def _(stats):
    # generate the boundary distribution samples
    upper_dist = stats.lognorm(s=0.7034, loc=1.64997, scale=0.6621)
    lower_dist1 = stats.weibull_max(c=1.4223, loc=2.421, scale=2.5)
    lower_dist2 = stats.lognorm(s=0.84899, loc=1.8153, scale=0.59599)
    return lower_dist1, lower_dist2, upper_dist


@app.cell
def _():
    # number of samples to pull from the distribution
    num_samples = 10000000
    return (num_samples,)


@app.cell
def _(num_samples, upper_dist):
    # generate the upper bound samples
    upper_bound_samples = upper_dist.rvs(size = num_samples)
    return (upper_bound_samples,)


@app.cell
def _(np):
    # random number generator
    rng = np.random.default_rng()
    return (rng,)


@app.cell
def _(lower_dist1, lower_dist2, np, num_samples, rng):
    # generate the lower bound samples
    lower_bound_weights = [0.6, 0.4]
    choices = rng.choice(a = [0, 1], size = num_samples, p = lower_bound_weights)

    lower_bound_samples = np.select(
        [choices == 0, choices == 1],
        [lower_dist1.rvs(size = num_samples), lower_dist2.rvs(size = num_samples) * -1]
    )
    return (lower_bound_samples,)


@app.cell
def _(lower_bound_samples, upper_bound_samples):
    upper_bounds_allowed = upper_bound_samples[(upper_bound_samples > 1.5) & (upper_bound_samples < 9)]
    lower_bounds_allowed = lower_bound_samples[(lower_bound_samples > -9)  & (lower_bound_samples < 2.5)]
    return lower_bounds_allowed, upper_bounds_allowed


@app.cell
def _(lower_bounds_allowed, np, plt, upper_bounds_allowed):
    # make sure the distribution looks about right
    plt.hist(np.concatenate((lower_bounds_allowed, upper_bounds_allowed)), bins = 100, density = True)
    return


@app.cell
def _(lower_bounds_allowed, plt):
    plt.hist(lower_bounds_allowed, bins = 100, density = True)
    return


@app.cell
def _(plt, upper_bounds_allowed):
    plt.hist(upper_bounds_allowed, bins = 100, density = True)
    return


@app.cell
def _(np, plt, stats):
    loc = 20
    scale = 10
    a = (9 - loc) / scale
    b = (50 - loc) / scale

    xx = np.linspace(start = 9, stop = 50, num = 1000)
    plt.plot(xx, stats.truncnorm.pdf(x = xx, a = a, b = b, loc = loc, scale = scale))
    return a, b, loc, scale


@app.cell
def _(a, b, loc, num_samples, scale, stats):
    sample_size_samples = stats.truncnorm.rvs(size = num_samples, a = a, b = b, loc = loc, scale = scale)
    return (sample_size_samples,)


@app.cell
def _(np, sample_size_samples):
    rounded_sample_size = np.round(sample_size_samples, decimals = 0)
    return (rounded_sample_size,)


@app.cell
def _(plt, rounded_sample_size):
    plt.hist(rounded_sample_size, bins = 100, density = True)
    return


@app.cell
def _(fmt_bd, np):
    def boundary_generator2(
            n_analyses, 
            upper_bound_samples, 
            lower_bound_samples,
            sample_size_samples,
            rng = np.random.default_rng()
    ):
        lower_bounds = np.zeros(n_analyses)
        upper_bounds = np.zeros(n_analyses)

        n = rng.choice(sample_size_samples)

        while not fmt_bd.check_monotonicity(
            bounds=np.concatenate((upper_bounds, lower_bounds)),
            n_analyses=n_analyses
        ):
            # upper bounds generation
            for _i in range(n_analyses):
                if _i == 0: 
                    upper_bounds[_i] = rng.choice(upper_bound_samples)
                else:
                    filter = upper_bound_samples <= upper_bounds[_i-1]
                    upper_bounds[_i] = rng.choice(upper_bound_samples[filter])

            # lower bounds generation
            for _j in range(n_analyses):
                if _j == 0: 
                    lower_bounds[_j] = rng.choice(lower_bound_samples)
                else:
                    filter = lower_bound_samples >= lower_bounds[_j-1]
                    lower_bounds[_j] = rng.choice(lower_bound_samples[filter])

            lower_bounds = np.concatenate((lower_bounds[0:2], np.array(upper_bounds[2], ndmin=1)))

        return (
            [upper_bounds, lower_bounds],
            n
        )

    return (boundary_generator2,)


@app.cell
def _(
    boundary_generator2,
    lower_bounds_allowed,
    rounded_sample_size,
    upper_bounds_allowed,
):
    test1, test2 = boundary_generator2(
        n_analyses=3,
        upper_bound_samples=upper_bounds_allowed,
        lower_bound_samples=lower_bounds_allowed,
        sample_size_samples=rounded_sample_size
    )
    return test1, test2


@app.cell
def _(test1):
    test1
    return


@app.cell
def _(test2):
    test2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simulated annealing function
    """)
    return


@app.cell
def _(bd, boundary_generator1, fn_min, fp, np, rng, sim, ss):
    def simulated_annealing_bounds1(
            sample_size_dist,
            num_analyses = 3,
            target_alpha = 0.05,
            target_power = 0.9,
            null_hypothesis = 0,
            important_diff_delta = 1,
            assumed_variance = 3,
            n_generate = 10000,
            temperature_start = 100
    ):
        # sample size at one stage
        mu = ss.sample_size_means(
            ratio = 1,
            variance = assumed_variance,
            power = target_power,
            alpha = target_alpha,
            delta = important_diff_delta
        )

        initial_design = bd.calculate_triangular_boundaries(
            n_analyses = num_analyses,
            alpha = target_alpha,
            delta = important_diff_delta,
            n_patients = 20
        )

        n_power, calced_power = ss.find_sample_size(
            power_target = target_power,
            n_analyses = num_analyses,
            upper_bounds = initial_design[0],
            lower_bounds = initial_design[1],
            null_hypothesis = null_hypothesis,
            alt_hypothesis = important_diff_delta,
            variance = assumed_variance
        )

        max_ess_initial = ss.max_ess(
            n_analyses = num_analyses,
            upper_bounds = initial_design[0],
            lower_bounds = initial_design[1],
            n_patients = n_power,
            null_hypothesis = null_hypothesis,
            variance = assumed_variance
        )

        penalty = fp.smooth_penalty(
            mu = mu,
            power = target_power,
            alpha = target_alpha,
            beta_prime = 1-calced_power,
            alpha_prime = initial_design[3]
        )

        f_value = fn_min.function_to_minimize(
            max_ess_val = max_ess_initial/mu, 
            penalty = penalty
        )

        f_min = f_value
        best_design = ([initial_design[0], initial_design[1]], n_power)
        current_design = best_design

        for i in range(n_generate):

            candidate_design, candidate_n = boundary_generator1(
                n_analyses = num_analyses,
                sample_size_samples=sample_size_dist
            )

            new_sim_trial = sim.group_sequential_designs(
                n_analyses = num_analyses,
                upper_bounds = candidate_design[0],
                lower_bounds = candidate_design[1],
                n_patients = candidate_n,
                null_hypothesis = null_hypothesis,
                alt_hypothesis = important_diff_delta,
                variance = assumed_variance
            )

            candidate_max_ess = ss.max_ess(
                n_analyses = num_analyses,
                upper_bounds = candidate_design[0],
                lower_bounds = candidate_design[1],
                n_patients = candidate_n,
                null_hypothesis = null_hypothesis,
                variance = assumed_variance
            )

            penalty = fp.smooth_penalty(
                mu = mu,
                power = target_power,
                alpha = target_alpha,
                beta_prime = 1-new_sim_trial[2],
                alpha_prime = new_sim_trial[1]
            )

            f_new = fn_min.function_to_minimize(
                max_ess_val = candidate_max_ess/mu, 
                penalty = penalty
            )

            temperature = temperature_start * (1 - (i/n_generate))
            uniform_selector = rng.uniform(size = 1)
            if np.exp( -1*(f_new - f_value)/temperature ) >= uniform_selector:
                f_value = f_new
                current_design = (candidate_design, candidate_n)

                if f_new < f_min:
                    best_design = current_design
                    f_min = f_new

            if i % 500 == 0:
                print(f"Completed loop {i}.")
                print(f"Temperature was {temperature}")
                print(f"Candidate design was {candidate_design, candidate_n}.")
                print(f"f_new = {f_new}.")
                print(f"f_min = {f_min}.")
                print(f"Best design is {best_design}.")
                print("\n")

        print(f"Final best design is {best_design}")


    return (simulated_annealing_bounds1,)


@app.cell
def _(rounded_sample_size, simulated_annealing_bounds1):
    simulated_annealing_bounds1(
        n_generate=10000,
        temperature_start=100,
        sample_size_dist=rounded_sample_size
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Best design is ([array([2.17003935, 1.96589949, 1.62327857]), array([0.04915954, 0.8690456 , 1.62327857])], 19.0).

    Final best design is ([array([2.0062693 , 1.98506952, 1.77954201]), array([-0.02909431,  0.89330079,  1.77954201])], 20.0)
    """)
    return


@app.cell
def _(bd, boundary_generator2, fn_min, fp, np, rng, sim, ss):
    def simulated_annealing_bounds2(
            upper_bound_dist,
            lower_bound_dist,
            sample_size_dist,
            num_analyses = 3,
            target_alpha = 0.05,
            target_power = 0.9,
            null_hypothesis = 0,
            important_diff_delta = 1,
            assumed_variance = 3,
            n_generate = 10000,
            temperature_start = 100
    ):
        # sample size at one stage
        mu = ss.sample_size_means(
            ratio = 1,
            variance = assumed_variance,
            power = target_power,
            alpha = target_alpha,
            delta = important_diff_delta
        )

        initial_design = bd.calculate_triangular_boundaries(
            n_analyses = num_analyses,
            alpha = target_alpha,
            delta = important_diff_delta,
            n_patients = 20
        )

        n_power, calced_power = ss.find_sample_size(
            power_target = target_power,
            n_analyses = num_analyses,
            upper_bounds = initial_design[0],
            lower_bounds = initial_design[1],
            null_hypothesis = null_hypothesis,
            alt_hypothesis = important_diff_delta,
            variance = assumed_variance
        )

        max_ess_initial = ss.max_ess(
            n_analyses = num_analyses,
            upper_bounds = initial_design[0],
            lower_bounds = initial_design[1],
            n_patients = n_power,
            null_hypothesis = null_hypothesis,
            variance = assumed_variance
        )

        penalty = fp.smooth_penalty(
            mu = mu,
            power = target_power,
            alpha = target_alpha,
            beta_prime = 1-calced_power,
            alpha_prime = initial_design[3]
        )

        f_value = fn_min.function_to_minimize(
            max_ess_val = max_ess_initial/mu, 
            penalty = penalty
        )

        f_min = f_value
        best_design = ([initial_design[0], initial_design[1]], n_power)
        current_design = best_design

        for i in range(n_generate):

            candidate_design, candidate_n = boundary_generator2(
                n_analyses = num_analyses,
                upper_bound_samples = upper_bound_dist,
                lower_bound_samples = lower_bound_dist,
                sample_size_samples=sample_size_dist
            )

            new_sim_trial = sim.group_sequential_designs(
                n_analyses = num_analyses,
                upper_bounds = candidate_design[0],
                lower_bounds = candidate_design[1],
                n_patients = candidate_n,
                null_hypothesis = null_hypothesis,
                alt_hypothesis = important_diff_delta,
                variance = assumed_variance
            )

            candidate_max_ess = ss.max_ess(
                n_analyses = num_analyses,
                upper_bounds = candidate_design[0],
                lower_bounds = candidate_design[1],
                n_patients = candidate_n,
                null_hypothesis = null_hypothesis,
                variance = assumed_variance
            )

            penalty = fp.smooth_penalty(
                mu = mu,
                power = target_power,
                alpha = target_alpha,
                beta_prime = 1-new_sim_trial[2],
                alpha_prime = new_sim_trial[1]
            )

            f_new = fn_min.function_to_minimize(
                max_ess_val = candidate_max_ess/mu, 
                penalty = penalty
            )

            temperature = temperature_start * (1 - (i/n_generate))
            uniform_selector = rng.uniform(size = 1)
            if np.exp( -1*(f_new - f_value)/temperature ) >= uniform_selector:
                f_value = f_new
                current_design = (candidate_design, candidate_n)

                if f_new < f_min:
                    best_design = current_design
                    f_min = f_new

            if i % 500 == 0:
                print(f"Completed loop {i}.")
                print(f"Temperature was {temperature}")
                print(f"Candidate design was {candidate_design, candidate_n}.")
                print(f"f_new = {f_new}.")
                print(f"f_min = {f_min}.")
                print(f"Best design is {best_design}.")
                print("\n")

        print(f"Final best design is {best_design}")


    return (simulated_annealing_bounds2,)


@app.cell
def _(
    lower_bounds_allowed,
    rounded_sample_size,
    simulated_annealing_bounds2,
    upper_bounds_allowed,
):
    simulated_annealing_bounds2(
        upper_bound_dist=upper_bounds_allowed,
        lower_bound_dist=lower_bounds_allowed, 
        sample_size_dist=rounded_sample_size,
        n_generate=10000,
        temperature_start=500
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Best design is ([array([1.94943652, 1.91900699, 1.88601071]), array([0.17162129, 1.10561495, 1.88601071])], 22.0).

    Final best design is ([array([1.87341895, 1.86175462, 1.77313281]), array([0.31780135, 1.31478985, 1.77313281])], 23.0)

    Final best design is ([array([2.11329269, 1.91919625, 1.7352423 ]), array([-0.15211942,  0.90897635,  1.7352423 ])], 19.0)

    Final best design is ([array([1.91843665, 1.83516768, 1.80215854]), array([0.47155417, 1.22180347, 1.80215854])], 24.0)

    Final best design is ([array([2.0055345 , 1.90415117, 1.79344621]), array([0.05456375, 0.91071865, 1.79344621])], 20.0)

    Final best design is ([array([2.10517982, 1.87240929, 1.77612033]), array([0.20623976, 1.04914026, 1.77612033])], 21.0)

    Final best design is ([array([2.00497444, 1.84259133, 1.81906221]), array([0.02176436, 1.13295626, 1.81906221])], 21.0)

    Final best design is ([array([1.93388083, 1.92289168, 1.84658093]), array([0.21927796, 1.11880345, 1.84658093])], 22.0)
    """)
    return


@app.cell
def _(lower_bound_samples, upper_bound_samples):
    upper_bounds_allowed_smaller = upper_bound_samples[(upper_bound_samples > 1.7) & (upper_bound_samples < 2.3)]
    lower_bounds_allowed_smaller = lower_bound_samples[(lower_bound_samples > -0.2)  & (lower_bound_samples < 2)]
    return lower_bounds_allowed_smaller, upper_bounds_allowed_smaller


@app.cell
def _(lower_bounds_allowed_smaller, np, plt, upper_bounds_allowed_smaller):
    # make sure the distribution looks about right
    plt.hist(np.concatenate((lower_bounds_allowed_smaller, upper_bounds_allowed_smaller)), bins = 100, density = True)
    return


@app.cell
def _(lower_bounds_allowed_smaller, plt):
    plt.hist(lower_bounds_allowed_smaller, bins = 100, density = True)
    return


@app.cell
def _(plt, upper_bounds_allowed_smaller):
    plt.hist(upper_bounds_allowed_smaller, bins = 100, density = True)
    return


@app.cell
def _(
    lower_bounds_allowed_smaller,
    rounded_sample_size,
    simulated_annealing_bounds2,
    upper_bounds_allowed_smaller,
):
    simulated_annealing_bounds2(
        upper_bound_dist=upper_bounds_allowed_smaller,
        lower_bound_dist=lower_bounds_allowed_smaller, 
        sample_size_dist=rounded_sample_size,
        n_generate=10000,
        temperature_start=500
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Final best design is ([array([1.98692188, 1.84852525, 1.78486305]), array([0.35658645, 0.97906147, 1.78486305])], 22.0)

    Final best design is ([array([2.08163471, 1.7700902 , 1.76215497]), array([0.25645982, 0.96502152, 1.76215497])], 21.0)

    Final best design is ([array([2.02999178, 1.83689778, 1.83592052]), array([0.22870415, 1.11982109, 1.83592052])], 22.0)
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
