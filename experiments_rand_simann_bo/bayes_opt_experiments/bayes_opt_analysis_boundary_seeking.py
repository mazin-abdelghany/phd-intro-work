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


if __name__ == "__main__":
    app.run()
