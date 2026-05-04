import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


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
    from py_group_sequential_designs import simulate as sim
    from py_group_sequential_designs import sample_size as ss
    from py_group_sequential_designs import feasibility_penalty as fp
    from py_group_sequential_designs import function_to_minimize as fn_min

    return bd, fn_min, fp, sim, ss


@app.cell
def _(ss):
    # design settings
    num_analyses = 3
    target_alpha = 0.05
    target_power = 0.9
    delta0 = 0.
    delta1 = 1.
    sigma2 = 3.

    mu = ss.sample_size_means(
        ratio=1,
        variance=sigma2,
        power=target_power,
        alpha=target_alpha,
        delta=delta1
    )
    return delta0, delta1, mu, num_analyses, sigma2, target_alpha, target_power


@app.cell
def _(fn_min, fp, sim, ss):
    # this function contains a penalty for non-monotonicity
    def obj_f(
            mu,
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

        alpha_prime = trial_sim[1]
        beta_prime = 1-trial_sim[2]

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
def _(
    bd,
    delta0,
    delta1,
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
        delta = delta1,
        n_patients = 20
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


    print(f"Original trriangular params: {np.round(np.concatenate((tri[0], tri[1])), 4)}")
    print(f"Triangular benchmark objective: {tri_obj:.4f}")
    print(f"Triangular alpha: {tri_alpha:.4f}")
    print(f"Triangular delta alpha: {abs(0.05-tri_alpha):.4f}")
    print(f"Triangular power: {tri_power:.4f}")
    print(f"Triangular delta power: {abs(0.9-tri_power):.4f}")
    print(f"Triangular sample size: {tri_n_patients:.1f}")
    print(f"Triangular max ESS: {tri_max_ess:.1f}")
    return tri, tri_alpha, tri_max_ess, tri_obj


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Quantities to follow

    Our design goals are $\alpha=$ {target_alpha} and $\beta=$ {1-target_power}. We will assess:

    - The boundary values and corresponding $\alpha'$, $\beta'$, $n$, and maximum expected sample size for the optimal design.
    - Feasibility: proportion of designs where
    \[
        \alpha' \le \alpha + \epsilon_1 \qquad (1-\beta') \ge (1-\beta) - \epsilon_1
    \]
    - Strict feasibility: proportion of designs where
    \[
        \alpha-\epsilon_2 \le \alpha' \le \alpha + \epsilon_2 \qquad (1-\beta)-\epsilon_2 \le (1-\beta') \le (1-\beta) + \epsilon_2
    \]
    - Does the best overall design $D^\star$ have a maximum expected sample size that is smaller than the triangular design?
    - Does the best overall design $D^\star$ have $\alpha'$ that is closer to the target than the triangular design?
    - The best overall $\mathcal{L}(\cdot)$ value obtained and its associated index (i.e., loop number at which the optimum was reached).
    - Is the best overall $\mathcal{L}(\cdot)$ better than the triangular design?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Large box results
    """)
    return


@app.cell
def _():
    n_loops = 500
    return (n_loops,)


@app.cell
def _(pd):
    small_box = pd.read_csv(
        filepath_or_buffer="/tf/2026-04-t21/simulated_annealing_experiments/triagular_box_t100_neigh2_results.csv"
    )

    # remove the first column as it is not needed
    small_box = small_box.iloc[:, 1:]
    return (small_box,)


@app.cell
def _(n_loops, np, small_box):
    for k in range(50):
        start_index = k * n_loops
        stop_index = start_index + n_loops

        analysis_set = small_box.iloc[start_index:stop_index, 6:11]
        execute = small_box.iloc[start_index:stop_index, 11]
        print("######")
        print(f"Run {k+1}:")
        print("######")
        print(f"Execute time: {np.round(execute[start_index]/60, decimals = 2)}")
        print(analysis_set.describe().drop('count'))

        print("\n")
    return


@app.cell
def _(n_loops, np, small_box):
    best_indeces = []
    for ell in range(50):
        _start_index = ell * n_loops
        _stop_index = _start_index + n_loops

        _analysis_set = small_box.iloc[_start_index:_stop_index, 6:11]
        best_index = np.argmin(_analysis_set['obj_func'])
        best_indeces.append(best_index)

        print("######")
        print(f"Run {ell+1}:")
        print("######")
        print(f"Best index: {best_index}")
        print(_analysis_set.iloc[best_index, :])
        print("\n")

    print(f"Minimum best index: {np.min(best_indeces)}")
    print(f"Maximum best index: {np.max(best_indeces)}")
    print(f"Average best index: {np.mean(best_indeces)}")
    print(f"Median best index: {np.median(best_indeces)}")
    return (best_indeces,)


@app.cell
def _(n_loops, np, small_box, target_alpha, target_power):
    epsilon1 = 0.01
    epsilon2 = 0.01

    for m in range(50):
        _start_index = m * n_loops
        _stop_index = _start_index + n_loops

        _analysis_set = small_box.iloc[_start_index:_stop_index, 6:11]

        within_e1 = (_analysis_set["alpha"] <= target_alpha + epsilon1) & (_analysis_set["power"] >= target_power - epsilon1)

        within_e2_alpha = (_analysis_set["alpha"]>=target_alpha-epsilon2) & (_analysis_set["alpha"]<=target_alpha+epsilon2)
        within_e2_power = (_analysis_set["power"]>=target_power-epsilon2) & (_analysis_set["power"]<=target_power+epsilon2)

        within_e2 = within_e2_alpha & within_e2_power

        print("######")
        print(f"Run {m+1}:")
        print("######")
        print(f"Feasibility: {np.round(np.mean(within_e1)*100, decimals = 1)}")
        print(f"Strict feasibility: {np.round(np.mean(within_e2)*100, decimals = 1)}")
        print("\n")
    return epsilon1, epsilon2


@app.cell
def _(epsilon1, epsilon2, np, small_box, target_alpha, target_power):
    all_within_e1 = (small_box["alpha"] <= target_alpha + epsilon1) & (small_box["power"] >= target_power - epsilon1)

    all_within_e2_alpha = (small_box["alpha"]>=target_alpha-epsilon2) & (small_box["alpha"]<=target_alpha+epsilon2)
    all_within_e2_power = (small_box["power"]>=target_power-epsilon2) & (small_box["power"]<=target_power+epsilon2)

    all_within_e2 = all_within_e2_alpha & all_within_e2_power

    print(f"Overall feasibility: {np.round(np.mean(all_within_e1)*100, decimals = 1)}")
    print(f"Overall strict feasibility: {np.round(np.mean(all_within_e2)*100, decimals = 1)}")
    return


@app.cell
def _(best_indeces, n_loops, small_box, tri_alpha, tri_max_ess, tri_obj):
    tri_diff = abs(0.05-tri_alpha)

    diffs = 0
    ess = 0
    objf = 0

    for _i in range(50):
        _start_index = _i * n_loops
        _stop_index = _start_index + n_loops

        _analysis_set = small_box.iloc[_start_index:_stop_index, 6:11]

        _alpha = _analysis_set.iloc[best_indeces[_i], :]['alpha']
        _max_ess = _analysis_set.iloc[best_indeces[_i], :]['max_ess']
        _obj_func = _analysis_set.iloc[best_indeces[_i], :]['obj_func']

        _alpha_diff = abs(0.05 - _alpha)

        diff_better_tri = _alpha_diff < tri_diff
        if diff_better_tri: diffs += 1

        ess_better_tri = _max_ess < tri_max_ess
        if ess_better_tri: ess += 1

        objf_better_tri = _obj_func < tri_obj
        if objf_better_tri: objf += 1

        print("######")
        print(f"Run {_i+1}:")
        print("######")
        print(f"Alpha closer than triangular? {diff_better_tri}")
        print(f"Max ESS lower than triangular? {ess_better_tri}")
        print(f"Objective f lower than triangular? {objf_better_tri}")
        print("\n")

    print(f"Total alpha closer: {diffs}")
    print(f"Total ESS lower:  {ess}")
    print(f"Total obj_f lower: {objf}")
    return


@app.cell
def _(best_indeces, n_loops, np, num_analyses, plt, small_box, tri):
    _fig, _ax = plt.subplots(figsize=(12,6))

    analysis_labels = [i+1 for i in range(num_analyses)]

    _ax.plot(analysis_labels, tri[0], color = "red", lw = 2)
    _ax.plot(analysis_labels, tri[1], color = "red", lw = 2)

    for _i, idx in enumerate(best_indeces):
        add_to_index = _i * n_loops
        _ax.plot(analysis_labels, small_box.iloc[idx+add_to_index, 1:4], color = "purple", alpha = 0.1)
        _ax.plot(analysis_labels, 
                 np.concatenate((small_box.iloc[idx+add_to_index, 4:6], [small_box.iloc[idx+add_to_index, 3]])),
                 color = "purple", alpha = 0.1)

    _ax.set_title("50 best boundaries: Large box")
    _fig
    return (analysis_labels,)


@app.cell
def _(mo):
    slider = mo.ui.slider(start=0, stop=49, label="Slider")
    return (slider,)


@app.cell
def _(mo, slider):
    mo.vstack([slider, mo.md(f"Has value: {slider.value}")])
    return


@app.cell
def _(
    analysis_labels,
    best_indeces,
    n_loops,
    np,
    num_analyses,
    plt,
    slider,
    small_box,
    tri,
):
    _fig, _ax = plt.subplots(figsize=(12,6))

    # get stage labels for plotting
    _analysis_labels = [i+1 for i in range(num_analyses)]

    # plot the triangular bounds
    _ax.plot(analysis_labels, tri[0], color = "darkorange", lw = 2)
    _ax.plot(analysis_labels, tri[1], color = "darkorange", lw = 2)

    # get the index of the best bounds
    _add_to_index = slider.value * n_loops
    _idx = best_indeces[slider.value]

    # plot the best bounds
    _ax.plot(_analysis_labels, small_box.iloc[_idx+_add_to_index, 1:4], color = "purple")
    _ax.plot(_analysis_labels, 
             np.concatenate((small_box.iloc[_idx+_add_to_index, 4:6], [small_box.iloc[_idx+_add_to_index, 3]])),
             color = "purple")

    _ax.text(2.75, -4, np.round(small_box.iloc[_idx+_add_to_index]['obj_func'], decimals = 4))

    _ax.set_title("50 best boundaries: Small box")
    _ax.set_ylim(-6,9)
    _fig
    return


@app.cell
def _(np, small_box):
    absolute_best = np.argmin(small_box['obj_func'])
    return (absolute_best,)


@app.cell
def _(absolute_best, small_box):
    small_box.iloc[absolute_best, 6:11]
    return


@app.cell
def _(absolute_best, analysis_labels, np, num_analyses, plt, small_box, tri):
    _fig, _ax = plt.subplots(figsize=(12,6))

    # get stage labels for plotting
    _analysis_labels = [i+1 for i in range(num_analyses)]

    # plot the triangular bounds
    _ax.plot(analysis_labels, tri[0], color = "darkorange", lw = 2)
    _ax.plot(analysis_labels, tri[1], color = "darkorange", lw = 2)

    # plot the best bounds
    _ax.plot(_analysis_labels, small_box.iloc[absolute_best, 1:4], color = "purple")
    _ax.plot(_analysis_labels, 
             np.concatenate((small_box.iloc[absolute_best, 4:6], [small_box.iloc[absolute_best, 3]])),
             color = "purple")

    _ax.text(2.75, -4, np.round(small_box.iloc[absolute_best]['obj_func'], decimals = 4))

    _ax.set_title("Absolute best boundaries: Small box")
    _ax.set_ylim(-6,9)
    _fig
    return


@app.cell
def _(small_box):
    small_box[small_box["index"] == 1001]["lower2"].item()
    return


@app.cell
def _(small_box):
    small_box[small_box["index"] == 1002]["lower2"].item()
    return


@app.cell
def _(n_loops, small_box):
    total_unique = 0
    column_labels = ["upper1", "upper2", "upper3", "lower1", "lower2", "sample_size"]

    for _i in range(50):
    
        index_counter = (_i+1) * 1000
    
        for _j in range(n_loops):
            n_check = 0
            row_to_check = index_counter + (_j + 1)
        
            print(row_to_check)
        
            above = small_box[small_box["index"] == row_to_check]
            below = small_box[small_box["index"] == row_to_check+1]
        
            for column in column_labels:
                print(above[column].item(), below[column].item())
                if above[column].item() == below[column].item():
                    n_check += 1
                
            if n_check == 0:
                total_unique += 1
    return


@app.cell
def _(small_box):
    small_box
    return


if __name__ == "__main__":
    app.run()
