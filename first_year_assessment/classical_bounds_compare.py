import marimo

__generated_with = "0.23.8"
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

    return np, plt


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
    num_analyses = 3
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

    print(f"""We are running an experiment with a trial design with {num_analyses} stages, with:

    a target alpha of              {target_alpha},
    a target power of              {target_power},
    a null hypothesis of           {delta0},
    an alternative hypothesis of   {delta1},
    and an assumed variance of     {sigma2}\n""")
      
    print(f"Single-stage sample size mu = {mu:.2f}")
    return delta0, delta1, num_analyses, sigma2, target_alpha, target_power


@app.cell
def _(fn_min, fp, sim, ss):
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pocock boundaries
    """)
    return


@app.cell
def _(
    bd,
    delta0,
    delta1,
    fmt_bd,
    np,
    num_analyses,
    obj_f,
    sigma2,
    ss,
    target_alpha,
    target_power,
):
    poc = bd.calculate_pocock_boundaries(
        n_analyses = num_analyses,
        alpha = target_alpha
    )

    poc_n_patients = ss.find_sample_size(
        power_target = target_power,
        n_analyses = num_analyses,
        upper_bounds = poc[0],
        lower_bounds = poc[1],
        null_hypothesis = delta0,
        alt_hypothesis = delta1,
        variance = sigma2
    )[0]

    poc_alpha, poc_power, poc_max_ess, poc_obj = obj_f(
        mu = 154,
        upper_bounds = poc[0],
        lower_bounds = poc[1],
        n_analyses = num_analyses,
        n_patients = poc_n_patients,
        target_power = target_power,
        target_alpha = target_alpha,
        null_hypothesis = delta0,
        alternative_hypothesis = delta1,
        variance = sigma2
    )

    poc_params = fmt_bd.boundaries_to_reverse(
        upper_bounds = poc[0],
        lower_bounds = poc[1]
    )

    c0_poc = poc_params[0]

    print(f"Original Pocock params:         {np.round(np.concatenate((poc[0], poc[1])), 4)}")
    print(f"Reparameterized Pocock params:  {np.round(poc_params, 4)}")
    print(f"Meeting point c0 =              {c0_poc:.4f}\n")
    print(f"Pocock benchmark objective:     {poc_obj:.4f}")
    print(f"Pocock alpha:                   {poc_alpha:.4f}")
    print(f"Pocock delta alpha:             {abs(0.05-poc_alpha):.4f}")
    print(f"Pocock power:                   {poc_power:.4f}")
    print(f"Pocock delta beta:              {abs(0.9-poc_power):.4f}")
    print(f"Pocock sample size:             {poc_n_patients:.1f}")
    print(f"Pocock max ESS:                 {poc_max_ess:.1f}")
    return poc, poc_max_ess, poc_obj


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # O'Brien-Fleming boundaries
    """)
    return


@app.cell
def _(
    bd,
    delta0,
    delta1,
    fmt_bd,
    np,
    num_analyses,
    obj_f,
    sigma2,
    ss,
    target_alpha,
    target_power,
):
    obf = bd.calculate_of_boundaries(
        n_analyses = num_analyses,
        alpha = target_alpha
    )

    obf_n_patients = ss.find_sample_size(
        power_target = target_power,
        n_analyses = num_analyses,
        upper_bounds = obf[0],
        lower_bounds = obf[1],
        null_hypothesis = delta0,
        alt_hypothesis = delta1,
        variance = sigma2
    )[0]

    obf_alpha, obf_power, obf_max_ess, obf_obj = obj_f(
        mu = 154,
        upper_bounds = obf[0],
        lower_bounds = obf[1],
        n_analyses = num_analyses,
        n_patients = obf_n_patients,
        target_power = target_power,
        target_alpha = target_alpha,
        null_hypothesis = delta0,
        alternative_hypothesis = delta1,
        variance = sigma2
    )

    obf_params = fmt_bd.boundaries_to_reverse(
        upper_bounds = obf[0],
        lower_bounds = obf[1]
    )

    c0_obf = obf_params[0]

    print(f"Original OBF params:         {np.round(np.concatenate((obf[0], obf[1])), 4)}")
    print(f"Reparameterized OBF params:  {np.round(obf_params, 4)}")
    print(f"Meeting point c0 =           {c0_obf:.4f}\n")
    print(f"OBF benchmark objective:     {obf_obj:.4f}")
    print(f"OBF alpha:                   {obf_alpha:.4f}")
    print(f"OBF delta alpha:             {abs(0.05-obf_alpha):.4f}")
    print(f"OBF power:                   {obf_power:.4f}")
    print(f"OBF delta beta:              {abs(0.9-obf_power):.4f}")
    print(f"OBF sample size:             {obf_n_patients:.1f}")
    print(f"OBF max ESS:                 {obf_max_ess:.1f}")
    return obf, obf_max_ess, obf_obj


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Triangular boundaries
    """)
    return


@app.cell
def _(
    bd,
    delta0,
    delta1,
    fmt_bd,
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
        mu = 154,
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

    print(f"Original trriangular params:        {np.round(np.concatenate((tri[0], tri[1])), 4)}")
    print(f"Reparameterized triangular params:  {np.round(tri_params, 4)}")
    print(f"Meeting point c0 =                  {c0:.4f}\n")
    print(f"Triangular benchmark objective:     {tri_obj:.4f}")
    print(f"Triangular alpha:                   {tri_alpha:.4f}")
    print(f"Triangular delta alpha:             {abs(0.05-tri_alpha):.4f}")
    print(f"Triangular power:                   {tri_power:.4f}")
    print(f"Triangular delta beta:              {abs(0.9-tri_power):.4f}")
    print(f"Triangular sample size:             {tri_n_patients:.1f}")
    print(f"Triangular max ESS:                 {tri_max_ess:.1f}")
    return tri, tri_max_ess, tri_obj


@app.cell
def _(
    num_analyses,
    obf,
    obf_max_ess,
    obf_obj,
    plt,
    poc,
    poc_max_ess,
    poc_obj,
    tri,
    tri_max_ess,
    tri_obj,
):
    _fig, _ax = plt.subplots(nrows=1, ncols=3, figsize=(14,4), sharey=True)

    stages = [i+1 for i in range(num_analyses)]

    _ax[0].plot(stages, poc[0], color = "purple")
    _ax[0].plot(stages, poc[1], color = "purple")
    _ax[0].scatter(stages, poc[0], color = "black", s=25, zorder=2)
    _ax[0].scatter(stages, poc[1], color = "black", s=25, zorder=2)

    _ax[1].plot(stages, obf[0], color = "purple")
    _ax[1].plot(stages, obf[1], color = "purple")
    _ax[1].scatter(stages, obf[0], color = "black", s=25, zorder=2)
    _ax[1].scatter(stages, obf[1], color = "black", s=25, zorder=2)

    _ax[2].plot(stages, tri[0], color = "purple")
    _ax[2].plot(stages, tri[1], color = "purple")
    _ax[2].scatter(stages, tri[0], color = "black", s=25, zorder=2)
    _ax[2].scatter(stages, tri[1], color = "black", s=25, zorder=2)

    _ax[0].set_ylabel("Standardised $Z_k$")
    _ax[0].set_ylim(-4, 4)
    _ax[0].set_xticks(stages)
    _ax[0].set_title("Pocock boundaries")
    _ax[0].text(2.1, -2.8, "$\mathcal{L}$ = " + f"{poc_obj:.4f}")
    _ax[0].text(2.1, -3.3, f"max ESS = {poc_max_ess:.1f}")

    _ax[1].set_xticks(stages)
    _ax[1].set_title("O'Brien-Fleming boundaries")
    _ax[1].text(2.1, -2.8, "$\mathcal{L}$ = " + f"{obf_obj:.4f}")
    _ax[1].text(2.1, -3.3, f"max ESS = {obf_max_ess:.1f}")

    _ax[2].set_xticks(stages)
    _ax[2].set_title("Triangular boundaries")
    _ax[2].text(2.1, -2.8, "$\mathcal{L}$ = " + f"{tri_obj:.4f}")
    _ax[2].text(2.1, -3.3, f"max ESS = {tri_max_ess:.1f}")

    for _i in range(num_analyses):
        _ax[_i].set_xlabel("Analysis stage, $k$")

    _fig.savefig("/tf/first_year_assessment/figures/classical_bounds.png", dpi=300, bbox_inches="tight")
    plt.gca()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
