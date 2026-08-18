import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    return np, pd, plt


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


@app.cell
def _(ss):
    target_alpha = 0.05
    target_power = 0.9
    delta0 = 0.
    delta1 = 1.
    sigma2 = 9.

    mu = ss.sample_size_means(
        ratio=1,
        variance=sigma2,
        power=target_power,
        alpha=target_alpha,
        delta=delta1
    )
    return delta0, delta1, mu, sigma2, target_alpha, target_power


@app.cell
def _(fn_min, fp, sim, ss):
    def obj_f(mu,
              upper_bounds,
              lower_bounds,
              n_patients,
              n_analyses,
              target_power,
              target_alpha,
              null_hypothesis,
              alternative_hypothesis,
              variance):

        trial_sim = sim.group_sequential_designs(
            n_analyses = n_analyses,
            upper_bounds = upper_bounds,
            lower_bounds = lower_bounds,
            n_patients = n_patients, 
            null_hypothesis = null_hypothesis,
            alt_hypothesis = alternative_hypothesis,
            variance = variance
        )

        alpha_prime = trial_sim[0]
        beta_prime = 1-trial_sim[1]

        max_ess = ss.max_ess(
            n_analyses = n_analyses,
            upper_bounds = upper_bounds,
            lower_bounds = lower_bounds,
            n_patients = n_patients,
            null_hypothesis = null_hypothesis,
            variance = variance
        )

        penalty = fp.smooth_penalty(
            mu = mu,
            power = target_power,
            alpha = target_alpha,
            beta_prime = beta_prime,
            alpha_prime = alpha_prime
        )

        f_val = fn_min.function_to_minimize(max_ess_val = max_ess/mu, penalty = penalty)

        return (
            alpha_prime,
            1-beta_prime,
            max_ess,
            f_val
        )

    return (obj_f,)


@app.cell
def _():
    num_analyses = 3
    return (num_analyses,)


@app.cell
def _(
    bd,
    delta0,
    delta1,
    fmt_bd,
    mu,
    np,
    num_analyses,
    obj_f,
    sigma2,
    ss,
    target_alpha,
    target_power,
):
    tri = bd.calculate_triangular_boundaries(
        n_analyses = num_analyses,
        alpha = target_alpha,
        delta = delta1
    )

    tri_n_patients = ss.find_sample_size(
        power_target = target_power,
        n_analyses = num_analyses,
        upper_bounds = tri[0],
        lower_bounds = tri[1],
        null_hypothesis = delta0,
        alt_hypothesis = delta1,
        variance = sigma2
    )[0]

    tri_alpha, tri_power, tri_max_ess, tri_obj = obj_f(
        mu = mu,
        upper_bounds = tri[0],
        lower_bounds = tri[1],
        n_analyses = num_analyses,
        n_patients = tri_n_patients,
        target_power = target_power,
        target_alpha = target_alpha,
        null_hypothesis = delta0,
        alternative_hypothesis = delta1,
        variance = sigma2
    )

    tri_params = fmt_bd.boundaries_to_reverse(
        upper_bounds = tri[0],
        lower_bounds = tri[1]
    )

    c0 = tri_params[0]

    print(f"Original trriangular params: {np.round(np.concatenate((tri[0], tri[1])), 4)}")
    print(f"Reparameterized triangular params: {np.round(tri_params, 4)}")
    print(f"Meeting point c0 = {c0:.4f}\n")
    print(f"Triangular benchmark objective: {tri_obj:.4f}")
    print(f"Triangular alpha: {tri_alpha:.4f}")
    print(f"Triangular delta alpha: {abs(0.05-tri_alpha):.4f}")
    print(f"Triangular power: {tri_power:.4f}")
    print(f"Triangular delta beta: {abs(0.9-tri_power):.4f}")
    print(f"Triangular sample size: {tri_n_patients:.1f}")
    print(f"Triangular max ESS: {tri_max_ess:.1f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Read in data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5-stages
    """)
    return


@app.cell
def _(pd):
    scaled_trieste = pd.read_csv(filepath_or_buffer="/tf/experiments_rand_simann_bo/bayes_opt_experiments/bo_smooth_50x500_x_min_max_y_z_scaled_0.001_500haltons.csv")
    return (scaled_trieste,)


@app.cell
def _():
    upper_keys = ["upper" + str(i+1) for i in range(5)]
    lower_keys = ["lower" + str(i+1) for i in range(4)]
    lower_keys = lower_keys + ["upper5"]
    return lower_keys, upper_keys


@app.cell
def _(np):
    reversed_bounds = np.empty((25000,9))
    return (reversed_bounds,)


@app.cell
def _(fmt_bd, lower_keys, reversed_bounds, scaled_trieste, upper_keys):
    for i in range(len(scaled_trieste)):
        reversed_bounds[i] = fmt_bd.boundaries_to_reverse(
            upper_bounds=scaled_trieste[upper_keys].to_numpy()[i],
            lower_bounds=scaled_trieste[lower_keys].to_numpy()[i]
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3-stages
    """)
    return


@app.cell
def _(pd):
    three_stage = pd.read_csv(filepath_or_buffer="/tf/experiments_rand_simann_bo/bayes_opt_experiments/3-stage-designs/large_box_bo_smooth_50x500.csv")
    return (three_stage,)


@app.cell
def _(np):
    three_stage_reversed_bounds = np.empty((25000,5))
    return (three_stage_reversed_bounds,)


@app.cell
def _(lower_keys, upper_keys):
    upper_keys3 = upper_keys[0:3]
    lower_keys3 = lower_keys[0:2] + ["upper3"]
    return lower_keys3, upper_keys3


@app.cell
def _(
    fmt_bd,
    lower_keys3,
    three_stage,
    three_stage_reversed_bounds,
    upper_keys3,
):
    for j in range(len(three_stage)):
        three_stage_reversed_bounds[j] = fmt_bd.boundaries_to_reverse(
            upper_bounds=three_stage[upper_keys3].to_numpy()[j],
            lower_bounds=three_stage[lower_keys3].to_numpy()[j]
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # What percent of bounds include boundaries?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5-stage designs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### At upper limit
    """)
    return


@app.cell
def _(np, reversed_bounds):
    np.round(sum(reversed_bounds == 4.901) / 25000 * 100, 1)
    return


@app.cell
def _(np, reversed_bounds):
    np.round(sum(reversed_bounds == 4) / 25000 * 100, 1)
    return


@app.cell
def _(np, reversed_bounds):
    np.round(sum(reversed_bounds == 160) / 25000 * 100, 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### At lower limit
    """)
    return


@app.cell
def _(np, reversed_bounds):
    np.round(sum(reversed_bounds == -1.099) / 25000 * 100, 1)
    return


@app.cell
def _(np, reversed_bounds):
    np.round(sum(reversed_bounds == 0) / 25000 * 100, 1)
    return


@app.cell
def _(np, reversed_bounds):
    np.round(sum(reversed_bounds == 20) / 25000 * 100, 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3-stage designs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### At 4
    """)
    return


@app.cell
def _(np, three_stage_reversed_bounds):
    np.round(sum(three_stage_reversed_bounds == 4) / 25000 * 100, 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### At 0
    """)
    return


@app.cell
def _(np, three_stage_reversed_bounds):
    np.round(sum(three_stage_reversed_bounds == 0) / 25000 * 100, 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploring boundary seeking with graphs
    """)
    return


@app.cell
def _(three_stage, three_stage_reversed_bounds):
    best_reversed = three_stage_reversed_bounds[three_stage["obj_func"].idxmin()]
    return (best_reversed,)


@app.cell
def _(best_reversed):
    best_reversed
    return


@app.cell
def _(three_stage):
    three_stage.loc[three_stage["obj_func"].idxmin()]
    return


@app.cell
def _(three_stage):
    best_upper = three_stage.loc[three_stage["obj_func"].idxmin()][2:5].to_list()
    return (best_upper,)


@app.cell
def _(three_stage):
    best_lower = three_stage.loc[three_stage["obj_func"].idxmin()][5:7].to_list()
    return (best_lower,)


@app.cell
def _(best_upper):
    best_upper
    return


@app.cell
def _(best_lower, best_upper):
    best_lower + [best_upper[2]]
    return


@app.cell
def _(best_reversed, fmt_bd, num_analyses):
    fmt_bd.reverse_to_boundaries(
        params = best_reversed,
        K=num_analyses
    )
    return


@app.cell
def _(
    best_lower,
    best_upper,
    delta0,
    delta1,
    mu,
    num_analyses,
    obj_f,
    sigma2,
    target_alpha,
    target_power,
):
    obj_f(
        mu = mu,
        upper_bounds = best_upper,
        lower_bounds = best_lower + [best_upper[2]],
        n_analyses = num_analyses,
        n_patients = 47.35,
        target_power = target_power,
        target_alpha = target_alpha,
        null_hypothesis = delta0,
        alternative_hypothesis = delta1,
        variance = sigma2
    )
    return


@app.cell
def _(mo):
    slider = mo.ui.slider(start=0, stop=6, step=0.001)
    return (slider,)


@app.cell
def _(best_reversed):
    best_reversed
    return


@app.cell
def _(best_reversed, slider):
    new_bounds = best_reversed[0:3].tolist() + [slider.value] + [best_reversed[4].tolist()]
    return (new_bounds,)


@app.cell
def _(fmt_bd, new_bounds):
    fmt_bd.reverse_to_boundaries(
        params=new_bounds,
        K=3
    )
    return


@app.cell
def _(
    best_reversed,
    delta0,
    delta1,
    fmt_bd,
    mu,
    np,
    num_analyses,
    obj_f,
    sigma2,
    target_alpha,
    target_power,
):
    plot_bounds = np.linspace(start=0, stop=6, num=2000)
    obj_funct = np.empty(2000)

    for je, point in enumerate(plot_bounds):
        _new_bounds = best_reversed[0:3].tolist() + [point] + [best_reversed[4].tolist()]

        the_bounds = fmt_bd.reverse_to_boundaries(
            params=_new_bounds,
            K=3
        )

        _, _, _, obj_funct[je] = obj_f(
            mu = mu,
            upper_bounds = the_bounds[0],
            lower_bounds = the_bounds[1],
            n_analyses = num_analyses,
            n_patients = 47.35,
            target_power = target_power,
            target_alpha = target_alpha,
            null_hypothesis = delta0,
            alternative_hypothesis = delta1,
            variance = sigma2
        )
    return obj_funct, plot_bounds


@app.cell
def _(fmt_bd, new_bounds):
    point_bounds = fmt_bd.reverse_to_boundaries(
        params=new_bounds,
        K=3
    )
    return (point_bounds,)


@app.cell
def _(
    delta0,
    delta1,
    mu,
    num_analyses,
    obj_f,
    point_bounds,
    sigma2,
    target_alpha,
    target_power,
):
    _, _, mess, of = obj_f(
        mu = mu,
        upper_bounds = point_bounds[0],
        lower_bounds = point_bounds[1],
        n_analyses = num_analyses,
        n_patients = 47.35,
        target_power = target_power,
        target_alpha = target_alpha,
        null_hypothesis = delta0,
        alternative_hypothesis = delta1,
        variance = sigma2
    )
    return mess, of


@app.cell
def _(mo, slider):
    mo.vstack([
        slider,
        mo.md(f"Has value = {slider.value}")
    ])
    return


@app.cell
def _(point_bounds):
    point_bounds
    return


@app.cell
def _(mess, np, obj_funct, of, plot_bounds, plt, slider):
    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(plot_bounds, obj_funct)
    ax.scatter(slider.value, of)
    ax.text(2, 0.76, np.round(of, 8))
    ax.text(2, 0.78, mess)
    ax.axvline(4, color = "red")

    ax.set_ylabel("Objective function")
    ax.set_xlabel("Reverse bound value")
    return


@app.cell
def _(three_stage):
    three_stage.loc[three_stage["obj_func"].nsmallest(50).index]
    return


@app.cell
def _(pd, three_stage, three_stage_reversed_bounds):
    pd.DataFrame(three_stage_reversed_bounds[three_stage["obj_func"].nsmallest(50).index])
    return


@app.cell
def _(three_stage_reversed_bounds):
    test_case = three_stage_reversed_bounds[8447]
    return (test_case,)


@app.cell
def _():
    test_case_sample_size = 49.44
    return (test_case_sample_size,)


@app.cell
def _(three_stage_reversed_bounds):
    three_stage_reversed_bounds[8447]
    return


@app.cell
def _(
    delta0,
    delta1,
    fmt_bd,
    mu,
    np,
    num_analyses,
    obj_f,
    plot_bounds,
    sigma2,
    target_alpha,
    target_power,
    test_case,
    test_case_sample_size,
):
    change_last = np.empty(2000)

    for _i, _point in enumerate(plot_bounds):
        _new_bounds = test_case[0:4].tolist() + [_point] 

        _the_bounds = fmt_bd.reverse_to_boundaries(
            params=_new_bounds,
            K=3
        )

        _, _, _, change_last[_i] = obj_f(
            mu = mu,
            upper_bounds = _the_bounds[0],
            lower_bounds = _the_bounds[1],
            n_analyses = num_analyses,
            n_patients = test_case_sample_size,
            target_power = target_power,
            target_alpha = target_alpha,
            null_hypothesis = delta0,
            alternative_hypothesis = delta1,
            variance = sigma2
        )
    return (change_last,)


@app.cell
def _(change_last, plot_bounds, plt):
    _fig, _ax = plt.subplots(figsize=(12,6))

    _ax.plot(plot_bounds, change_last)
    _ax.axvline(0, color = "red")

    _ax.set_ylabel("Objective function")
    _ax.set_xlabel("Reverse bound value")
    return


@app.cell
def _(
    delta0,
    delta1,
    fmt_bd,
    mu,
    np,
    num_analyses,
    obj_f,
    plot_bounds,
    sigma2,
    target_alpha,
    target_power,
    test_case,
    test_case_sample_size,
):
    change_secondToLast = np.empty(2000)

    for _i, _point in enumerate(plot_bounds):
        _new_bounds = test_case[0:3].tolist() + [_point] + [test_case[4].tolist()]

        _the_bounds = fmt_bd.reverse_to_boundaries(
            params=_new_bounds,
            K=3
        )

        _, _, _, change_secondToLast[_i] = obj_f(
            mu = mu,
            upper_bounds = _the_bounds[0],
            lower_bounds = _the_bounds[1],
            n_analyses = num_analyses,
            n_patients = test_case_sample_size,
            target_power = target_power,
            target_alpha = target_alpha,
            null_hypothesis = delta0,
            alternative_hypothesis = delta1,
            variance = sigma2
        )
    return (change_secondToLast,)


@app.cell
def _(change_secondToLast, plot_bounds, plt):
    _fig, _ax = plt.subplots(figsize=(12,6))

    _ax.plot(plot_bounds, change_secondToLast)
    _ax.axvline(4, color = "red")

    _ax.set_ylabel("Objective function")
    _ax.set_xlabel("Reverse bound value")
    return


@app.cell
def _(
    delta0,
    delta1,
    fmt_bd,
    mu,
    np,
    num_analyses,
    obj_f,
    plot_bounds,
    sigma2,
    target_alpha,
    target_power,
    test_case,
    test_case_sample_size,
):
    change_thirdToLast = np.empty(2000)

    for _i, _point in enumerate(plot_bounds):
        _new_bounds = test_case[0:2].tolist() + [_point] + test_case[3:5].tolist()

        _the_bounds = fmt_bd.reverse_to_boundaries(
            params=_new_bounds,
            K=3
        )

        _, _, _, change_thirdToLast[_i] = obj_f(
            mu = mu,
            upper_bounds = _the_bounds[0],
            lower_bounds = _the_bounds[1],
            n_analyses = num_analyses,
            n_patients = test_case_sample_size,
            target_power = target_power,
            target_alpha = target_alpha,
            null_hypothesis = delta0,
            alternative_hypothesis = delta1,
            variance = sigma2
        )
    return (change_thirdToLast,)


@app.cell
def _(mo):
    slider2 = mo.ui.slider(start=0, stop=2000, step=1)
    return (slider2,)


@app.cell
def _(mo, slider2):
    mo.vstack([
        slider2,
        mo.md(f"Has value = {slider2.value}")
        ])
    return


@app.cell
def _(change_thirdToLast, plot_bounds, plt, slider2):
    _fig, _ax = plt.subplots(figsize=(12,6))

    _ax.plot(plot_bounds, change_thirdToLast)
    _ax.axvline(1.57184977, color = "red")
    _ax.scatter(plot_bounds[slider2.value], change_thirdToLast[slider2.value])
    _ax.text(2, 10, change_thirdToLast[slider2.value])
    _ax.set_ylabel("Objective function")
    _ax.set_xlabel("Reverse bound value")
    return


@app.cell
def _(plot_bounds):
    plot_bounds[540]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sanity checks
    """)
    return


@app.cell
def _(sim):
    from scipy.optimize import minimize_scalar

    def max_ess(
        delta_start=0,
        n_analyses=3,
        upper_bounds=None,
        lower_bounds=None,
        n_patients=20,
        null_hypothesis=0,
        variance=1
    ):
        if upper_bounds is None:
            upper_bounds = [2.5, 2.0, 1.5]
        if lower_bounds is None:
            lower_bounds = [0.0, 0.75, 1.5]

        delta_stop = variance * 5

        # Objective function to MINIMIZE (-1 * ESS to find the MAXIMUM)
        def objective(delta):
            _, _, ess = sim.group_sequential_designs(
                n_analyses=n_analyses,
                upper_bounds=upper_bounds,
                lower_bounds=lower_bounds,
                n_patients=n_patients,
                null_hypothesis=null_hypothesis,
                alt_hypothesis=delta,
                variance=variance
            )
            return -ess  # Negate because minimize_scalar finds the minimum

        # Maximize over bounds [delta_start, delta_stop]
        res = minimize_scalar(
            objective, 
            bounds=(delta_start, delta_stop), 
            method='bounded',
            options={'xatol': 1e-4}
        )

        max_delta = res.x
        max_ess_value = -res.fun

        return max_ess_value

    return (max_ess,)


@app.cell
def _(max_ess):
    max_ess()
    return


@app.cell
def _(ss):
    ss.max_ess()
    return


@app.cell
def _(sim):
    sim.group_sequential_designs(return_table=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reproducing results with published package
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In R

    ```{r}
    library(gsDesign)

    b_bounds <- c(2.5, 2.0, 1.5)  # upper bounds
    a_bounds <- c(0, 0.75, 1.5)   # lower bounds
    n_info   <- c(20, 40, 60)     # sample size per look

    res <- gsProbability(
      k = 3,
      theta = seq(0, 0.4, 0.01),
      n.I = n_info,
      a = a_bounds,
      b = b_bounds,
      r = 79
    )

    res
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    outputs

    ```
     Lower bounds   Upper bounds
      Analysis N   Z   Nominal p Z   Nominal p
             1 20 0.00    0.5000 2.5    0.0062
             2 40 0.75    0.7734 2.0    0.0228
             3 60 1.50    0.9332 1.5    0.0668

    Boundary crossing probabilities and expected sample size assume
    any cross stops the trial

    Upper boundary
              Analysis
      Theta      1      2      3  Total E{N}
       0.00 0.0062 0.0194 0.0383 0.0639 33.4
       0.01 0.0070 0.0225 0.0442 0.0737 34.0
       0.02 0.0080 0.0259 0.0506 0.0845 34.6
       0.03 0.0090 0.0298 0.0577 0.0965 35.3
       0.04 0.0101 0.0341 0.0655 0.1097 35.9
       0.05 0.0114 0.0389 0.0739 0.1242 36.5
       0.06 0.0128 0.0442 0.0829 0.1400 37.1
       0.07 0.0144 0.0501 0.0925 0.1570 37.7
       0.08 0.0161 0.0566 0.1027 0.1754 38.3
       0.09 0.0180 0.0636 0.1135 0.1951 38.9
       0.10 0.0200 0.0713 0.1247 0.2160 39.5
       0.11 0.0223 0.0797 0.1362 0.2382 40.0
       0.12 0.0248 0.0887 0.1481 0.2616 40.5
       0.13 0.0275 0.0984 0.1602 0.2861 41.0
       0.14 0.0305 0.1087 0.1724 0.3116 41.4
       0.15 0.0337 0.1198 0.1845 0.3380 41.9
       0.16 0.0372 0.1315 0.1965 0.3652 42.3
       0.17 0.0410 0.1438 0.2082 0.3930 42.6
       0.18 0.0450 0.1568 0.2195 0.4214 42.9
       0.19 0.0494 0.1704 0.2303 0.4502 43.2
       0.20 0.0542 0.1846 0.2404 0.4791 43.4
       0.21 0.0593 0.1993 0.2496 0.5082 43.6
       0.22 0.0647 0.2144 0.2580 0.5371 43.7
       0.23 0.0706 0.2299 0.2653 0.5658 43.8
       0.24 0.0768 0.2458 0.2715 0.5941 43.8
       0.25 0.0835 0.2618 0.2765 0.6218 43.8
       0.26 0.0906 0.2781 0.2802 0.6489 43.8
       0.27 0.0981 0.2944 0.2826 0.6751 43.7
       0.28 0.1061 0.3106 0.2837 0.7004 43.5
       0.29 0.1145 0.3268 0.2835 0.7247 43.3
       0.30 0.1234 0.3427 0.2819 0.7479 43.1
       0.31 0.1327 0.3583 0.2790 0.7700 42.9
       0.32 0.1426 0.3734 0.2749 0.7909 42.6
       0.33 0.1529 0.3880 0.2696 0.8105 42.2
       0.34 0.1637 0.4020 0.2632 0.8288 41.9
       0.35 0.1750 0.4152 0.2558 0.8460 41.5
       0.36 0.1867 0.4276 0.2475 0.8618 41.0
       0.37 0.1990 0.4391 0.2384 0.8764 40.6
       0.38 0.2117 0.4496 0.2286 0.8899 40.1
       0.39 0.2249 0.4590 0.2183 0.9021 39.7
       0.40 0.2385 0.4673 0.2075 0.9133 39.2
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And we can confirm futility probabilities under the null by

    ```{r}
    res$lower
    ```

    gives

    ```
             [,1]
    [1,] 0.5000000
    [2,] 0.2987760
    [3,] 0.1372852
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And efficacy probabilities under the null by:

    ```{r}
    res$upper
    ```

    gives

    ```
                [,1]
    [1,] 0.006209665
    [2,] 0.019414842
    [3,] 0.038314311
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, max ESS can be confirmed with the output if theta seen above with these row:

    ```
    Upper boundary
              Analysis
      Theta      1      2      3  Total E{N}
       0.21 0.0593 0.1993 0.2496 0.5082 43.6
       0.22 0.0647 0.2144 0.2580 0.5371 43.7
       0.23 0.0706 0.2299 0.2653 0.5658 43.8
       0.24 0.0768 0.2458 0.2715 0.5941 43.8
       0.25 0.0835 0.2618 0.2765 0.6218 43.8
       0.26 0.0906 0.2781 0.2802 0.6489 43.8
       0.27 0.0981 0.2944 0.2826 0.6751 43.7
       0.28 0.1061 0.3106 0.2837 0.7004 43.5
       0.29 0.1145 0.3268 0.2835 0.7247 43.3
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can work on the standarized scale in order to assess the bounds above. Our current experiment uses:
    - Null: $\delta_0 = 0$
    - Alternative: $\delta_1 = 1$
    - Variance: $\sigma^2 = 9$

    Calling the standardized parameter $\theta$,
    $$
    \theta = \frac{\delta_1 - \delta_0}{\sqrt{2 \sigma^2}} = \frac{1 - 0}{3\sqrt{2}} = 0.2357
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can use this to generate the plot above in R
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```{r}
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


    upper_bounds <- c(6.79534931, 2.08334931, 1.5176033)
    lower_bounds <- c(-0.24554805, 0.25558817, 1.5176033)
    mu <- 154

    obj_f(mu = mu,
          upper_bounds = upper_bounds,
          lower_bounds = lower_bounds,
          n_analyses = 3,
          n_patients = 47.35,
          target_power = 0.9,
          target_alpha = 0.05,
          null_hypothesis = 0,
          alt_hypothesis = 1,
          variance = 9)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    output:

    ```
          alpha'       power'         mESS     obj func
      0.06470101   0.88720125 116.98510419   0.81815237
    ```
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
