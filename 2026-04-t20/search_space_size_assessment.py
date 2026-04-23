import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    return np, plt


@app.cell
def _():
    from py_group_sequential_designs import generate_boundaries as bd
    from py_group_sequential_designs import boundary_manipulations as fmt_bd
    from py_group_sequential_designs import sample_size as ss
    from py_group_sequential_designs import simulate as sim
    from py_group_sequential_designs import feasibility_penalty as fp
    from py_group_sequential_designs import function_to_minimize as fn_min

    return bd, fmt_bd, fn_min, fp, sim, ss


@app.cell
def _(np):
    def reverse_to_boundaries(params, K):
        params = np.asarray(params).flatten()
        c = params[0]

        delta_u = params[1::2][::-1]
        delta_l = params[2::2][::-1]

        upper_bounds = np.array([c + np.sum(delta_u[k:]) for k in range(K)])
        lower_bounds = np.array([c - np.sum(delta_l[k:]) for k in range(K)])

        return upper_bounds, lower_bounds

    def boundaries_to_reverse(upper_bounds, lower_bounds):
        upper_bounds = np.asarray(upper_bounds)
        lower_bounds = np.asarray(lower_bounds)

        K = len(upper_bounds)
        c = upper_bounds[-1]

        delta_u = np.diff(upper_bounds[::-1])
        delta_l = np.diff(lower_bounds)[::-1]

        increments = np.empty(2 * (K - 1))
        increments[0::2] = delta_u
        increments[1::2] = delta_l

        return np.concatenate([[c], increments])

    return boundaries_to_reverse, reverse_to_boundaries


@app.cell
def _(delta0, delta1, fn_min, fp, num_analyses, sigma2, sim, ss):
    # this function contains a penalty for non-monotonicity
    def obj_f(
            mu,
            upper_bounds,
            lower_bounds,
            n_analyses,
            target_power,
            target_alpha):

        n_power09, calc_power = ss.find_sample_size(
            power_target = target_power,
            n_analyses = n_analyses,
            upper_bounds = upper_bounds,
            lower_bounds = lower_bounds,
            null_hypothesis = delta0,
            alt_hypothesis = delta1,
            variance = sigma2
        )

        beta_prime = 1-calc_power

        alpha_prime = sim.group_sequential_designs(
            n_analyses = num_analyses,
            upper_bounds = upper_bounds,
            lower_bounds = lower_bounds,
            n_patients = n_power09, 
            null_hypothesis = delta0,
            alt_hypothesis = delta1,
            variance = sigma2
        )[1]

        max_ess = ss.max_ess(
            n_analyses = n_analyses,
            upper_bounds = upper_bounds,
            lower_bounds = lower_bounds,
            n_patients = n_power09
        )

        penalty = fp.smooth_penalty(
            mu = mu,
            power = target_power,
            alpha = target_alpha,
            beta_prime = beta_prime,
            alpha_prime = alpha_prime
        )

        f_val = fn_min.function_to_minimize(max_ess_val=max_ess/mu, penalty=penalty)

        return (
            alpha_prime,
            calc_power,
            n_power09,
            f_val
        )

    return (obj_f,)


@app.cell
def _(ss):
    num_analyses = 3
    target_alpha = 0.05
    target_power = 0.9
    delta0 = 0
    delta1 = 1.0
    sigma2 = 3.0

    mu = ss.sample_size_means(
        ratio=1,
        variance=sigma2,
        power=target_power,
        alpha=target_alpha,
        delta=delta1
    )
    print(f"Single-stage sample size mu = {mu:.2f}")
    return delta0, delta1, mu, num_analyses, sigma2, target_alpha, target_power


@app.cell
def _(
    bd,
    boundaries_to_reverse,
    delta1,
    mu,
    np,
    num_analyses,
    obj_f,
    target_alpha,
    target_power,
):
    tri = bd.calculate_triangular_boundaries(
        n_analyses=num_analyses,
        alpha=target_alpha,
        delta=delta1,
        n_patients=20
    )

    _,_,_,tri_obj = obj_f(
        mu = mu,
        upper_bounds = tri[0],
        lower_bounds = tri[1],
        n_analyses = num_analyses,
        target_power = target_power,
        target_alpha = target_alpha
    )

    tri_params = boundaries_to_reverse(tri[0], tri[1])
    c0 = tri_params[0]

    print(f"Triangular benchmark objective: {tri_obj:.4f}")
    print(f"Original trriangular params: {np.round(np.concatenate((tri[0], tri[1])), 4)}")
    print(f"Reparameterized triangular params: {np.round(tri_params, 4)}")
    print(f"Meeting point c0 = {c0:.4f}")
    return c0, tri, tri_params


@app.cell
def _(bd, num_analyses, target_alpha):
    po = bd.calculate_pocock_boundaries(
        n_analyses=num_analyses,
        alpha=target_alpha,
        n_patients=20
    )
    return (po,)


@app.cell
def _(bd, num_analyses, target_alpha):
    of = bd.calculate_of_boundaries(
        n_analyses=num_analyses,
        alpha=target_alpha,
        n_patients=20
    )
    return (of,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Original search space
    """)
    return


@app.cell
def _(c0, np):
    # c0 is meeting point
    # next element in array is delta_u{k-1}
    # next element in array is delta_l{k-1}
    b_lower = np.array([c0 - 3.0, 0.0, 0.0, 0.0, 0.0])
    b_upper = np.array([c0 + 3.0, 4.0, 4.0, 4.0, 4.0])
    return b_lower, b_upper


@app.cell
def _(np):
    rng_baseline = np.random.default_rng(seed=123)
    return (rng_baseline,)


@app.cell
def _(b_lower, b_upper, rng_baseline):
    n_baseline = 1000
    rb_bounds_to_test = rng_baseline.uniform(b_lower, b_upper, size=(n_baseline, len(b_upper)))
    return n_baseline, rb_bounds_to_test


@app.cell
def _(n_baseline, num_analyses, rb_bounds_to_test, reverse_to_boundaries):
    b_bounds_to_test = []

    for _i in range(n_baseline):
        b_bounds_to_test.append(reverse_to_boundaries(rb_bounds_to_test[_i], K = num_analyses))
    return (b_bounds_to_test,)


@app.cell
def _(b_bounds_to_test, n_baseline, of, plt, po, tri):
    _fig, _ax = plt.subplots()

    for _i in range(n_baseline):
        _ax.plot([1,2,3], b_bounds_to_test[_i][0], color = "purple", alpha = 0.1)
        _ax.plot([1,2,3], b_bounds_to_test[_i][1], color = "purple", alpha = 0.1)

    _ax.plot([1,2,3], tri[0], color = "red", lw = 2)
    _ax.plot([1,2], tri[1][0:2], color = "red", lw = 2)

    _ax.plot([1,2,3], po[0], color = "blue", lw = 2)
    _ax.plot([1,2], po[1][0:2], color = "blue", lw = 2)

    _ax.plot([1,2,3], of[0], color = "green", lw = 2)
    _ax.plot([1,2], of[1][0:2], color = "green", lw = 2)

    _fig
    return


@app.cell
def _(b_bounds_to_test, n_baseline, np, plt, tri):
    _fig, _ax = plt.subplots()

    bnum = 0
    for _i in range(n_baseline):
        up_bound = np.asarray(b_bounds_to_test[_i][0])
        lo_bound = np.asarray(b_bounds_to_test[_i][1])
    
        if ((up_bound[0] <= 2.5) and (lo_bound[0] >= -0.5)):
            _ax.plot([1,2,3], b_bounds_to_test[_i][0], color = "purple", alpha = 0.1)
            _ax.plot([1,2,3], b_bounds_to_test[_i][1], color = "purple", alpha = 0.1)

            bnum = bnum+1

    _ax.plot([1,2,3], tri[0], color = "red", lw = 2)
    _ax.plot([1,2], tri[1][0:2], color = "red", lw = 2)

    _fig
    return (bnum,)


@app.cell
def _(bnum):
    bnum
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Large search space
    """)
    return


@app.cell
def _(c0, np):
    # c0 is meeting point
    # next element in array is delta_u{k-1}
    # next element in array is delta_l{k-1}
    lower = np.array([c0 - 1, 0.0, 0.0, 0.0, 0.0])
    upper = np.array([c0 + 1, 1.0, 4.0, 1.0, 1.0])
    return lower, upper


@app.cell
def _(lower, n_baseline, rng_baseline, upper):
    reversed_bounds_to_test = rng_baseline.uniform(lower, upper, size=(n_baseline, len(lower)))
    return (reversed_bounds_to_test,)


@app.cell
def _(
    n_baseline,
    num_analyses,
    reverse_to_boundaries,
    reversed_bounds_to_test,
):
    bounds_to_test = []

    for _i in range(n_baseline):
        bounds_to_test.append(reverse_to_boundaries(reversed_bounds_to_test[_i], K = num_analyses))
    return (bounds_to_test,)


@app.cell
def _(bounds_to_test, n_baseline, of, plt, po, tri):
    _fig, _ax = plt.subplots()

    for _i in range(n_baseline):
        _ax.plot([1,2,3], bounds_to_test[_i][0], color = "purple", alpha = 0.1)
        _ax.plot([1,2,3], bounds_to_test[_i][1], color = "purple", alpha = 0.1)

    _ax.plot([1,2,3], tri[0], color = "red", lw = 2)
    _ax.plot([1,2], tri[1][0:2], color = "red", lw = 2)

    _ax.plot([1,2,3], po[0], color = "blue", lw = 2)
    _ax.plot([1,2], po[1][0:2], color = "blue", lw = 2)

    _ax.plot([1,2,3], of[0], color = "green", lw = 2)
    _ax.plot([1,2], of[1][0:2], color = "green", lw = 2)

    _fig
    return


@app.cell
def _(bounds_to_test, n_baseline, np, plt, tri):
    _fig, _ax = plt.subplots()

    num = 0
    for _i in range(n_baseline):
        upper_bound = np.asarray(bounds_to_test[_i][0])
        lower_bound = np.asarray(bounds_to_test[_i][1])
    
        if ((upper_bound[0] <= 2.5) and (lower_bound[0] >= -0.5)):
            _ax.plot([1,2,3], bounds_to_test[_i][0], color = "purple", alpha = 0.1)
            _ax.plot([1,2,3], bounds_to_test[_i][1], color = "purple", alpha = 0.1)

            num = num+1

    _ax.plot([1,2,3], tri[0], color = "red", lw = 2)
    _ax.plot([1,2], tri[1][0:2], color = "red", lw = 2)

    _fig
    return (num,)


@app.cell
def _(num):
    num
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Small search space
    """)
    return


@app.cell
def _(tri_params):
    tri_params
    return


@app.cell
def _(np, tri_params):
    small_lower = []
    small_upper = []

    for param in tri_params:
        small_lower.append(max(0, param - 0.4))
        small_upper.append(param + 0.4)

    print(f"Lower: {np.round(small_lower, 3)}")
    print(f"Upper: {np.round(small_upper, 3)}")
    return small_lower, small_upper


@app.cell
def _(n_baseline, rng_baseline, small_lower, small_upper):
    small_reversed_bounds_to_test = rng_baseline.uniform(
        small_lower, small_upper, 
        size=(n_baseline, len(small_lower))
    )
    return (small_reversed_bounds_to_test,)


@app.cell
def _(
    n_baseline,
    num_analyses,
    reverse_to_boundaries,
    small_reversed_bounds_to_test,
):
    small_bounds_to_test = []

    for _i in range(n_baseline):
        small_bounds_to_test.append(reverse_to_boundaries(small_reversed_bounds_to_test[_i], K = num_analyses))
    return (small_bounds_to_test,)


@app.cell
def _(fmt_bd, n_baseline, np, small_bounds_to_test):
    n_monotonic = 0
    for _i in range(n_baseline):
    
        u = small_bounds_to_test[_i][0]
        l = small_bounds_to_test[_i][1]
    
        u = np.concatenate((u, l[0:2]))
        u = np.concatenate((u, [1]))
    
        mono = fmt_bd.check_monotonicity(
            n_analyses = 3,
            bounds = u
        )

        if mono: n_monotonic = n_monotonic + 1

    n_monotonic
    return


@app.cell
def _(n_baseline, of, plt, po, small_bounds_to_test, tri):
    _fig, _ax = plt.subplots()

    for _i in range(n_baseline):
        _ax.plot([1,2,3], small_bounds_to_test[_i][0], color = "purple", alpha = 0.1)
        _ax.plot([1,2,3], small_bounds_to_test[_i][1], color = "purple", alpha = 0.1)

    _ax.plot([1,2,3], tri[0], color = "red", lw = 2)
    _ax.plot([1,2], tri[1][0:2], color = "red", lw = 2)

    _ax.plot([1,2,3], po[0], color = "blue", lw = 2)
    _ax.plot([1,2], po[1][0:2], color = "blue", lw = 2)

    _ax.plot([1,2,3], of[0], color = "green", lw = 2)
    _ax.plot([1,2], of[1][0:2], color = "green", lw = 2)

    _fig
    return


@app.cell
def _(n_baseline, np, plt, small_bounds_to_test, tri):
    _fig, _ax = plt.subplots()

    small_num = 0
    for _i in range(n_baseline):
        u_bound = np.asarray(small_bounds_to_test[_i][0])
        l_bound = np.asarray(small_bounds_to_test[_i][1])
    
        if ((u_bound[0] <= 2.5) and (l_bound[0] >= -0.5)):
            _ax.plot([1,2,3], small_bounds_to_test[_i][0], color = "purple", alpha = 0.1)
            _ax.plot([1,2,3], small_bounds_to_test[_i][1], color = "purple", alpha = 0.1)

            small_num = small_num+1

    _ax.plot([1,2,3], tri[0], color = "red", lw = 2)
    _ax.plot([1,2], tri[1][0:2], color = "red", lw = 2)

    _fig
    return (small_num,)


@app.cell
def _(small_num):
    small_num
    return


if __name__ == "__main__":
    app.run()
