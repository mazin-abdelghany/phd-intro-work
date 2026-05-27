import marimo

__generated_with = "0.23.6"
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

    return np, pd


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
    ##############
    # Setup cell #
    ##############

    # design settings
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
    return delta0, delta1, mu, num_analyses, sigma2, target_alpha, target_power


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


@app.cell
def _(
    bd,
    delta0,
    delta1,
    mu,
    num_analyses,
    obj_f,
    sigma2,
    ss,
    target_alpha,
    target_power,
):
    tri = bd.calculate_triangular_boundaries(
        n_analyses = num_analyses,
        alpha = target_alpha
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
    return


@app.cell
def _(mo):
    n_experiments = mo.ui.number(label="No. experiments = ")
    return (n_experiments,)


@app.cell
def _(mo):
    n_loops = mo.ui.number(label="Number of loops = ")
    return (n_loops,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Experimental setup
    """)
    return


@app.cell
def _(mo, n_experiments):
    mo.vstack(
        [n_experiments]
    )
    return


@app.cell
def _(mo, n_loops):
    mo.vstack(
        [n_loops]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data files to compare
    """)
    return


@app.cell
def _(mo):
    file_a = mo.ui.text(placeholder="/path/to/file.csv", label="File 1: ")
    return (file_a,)


@app.cell
def _(mo):
    file_b = mo.ui.text(placeholder="/path/to/file.csv", label="File 2: ")
    return (file_b,)


@app.cell
def _(mo):
    label_a = mo.ui.text(placeholder="data_label", label="Method 1: ")
    return (label_a,)


@app.cell
def _(mo):
    label_b = mo.ui.text(placeholder="data_label", label="Method 2: ")
    return (label_b,)


@app.cell
def _(file_a, file_b, label_a, label_b, mo):
    mo.hstack(
        [mo.vstack([label_a, label_b]),
         mo.vstack([file_a, file_b])]
    )
    return


@app.cell
def _(file_a, file_b, pd):
    # import the data
    data_a = pd.read_csv(file_a.value)
    data_b = pd.read_csv(file_b.value)

    # remove the first column as it is not needed
    data_a = data_a.iloc[:, 1:]
    data_b = data_b.iloc[:, 1:]
    return data_a, data_b


@app.cell
def _(n_experiments, n_loops, np):
    runs = np.concatenate([np.repeat(f"Run_{i+1}", n_loops.value) for i in range(n_experiments.value)])
    return (runs,)


@app.cell
def _(data_a, data_b, runs):
    data_a["runs"] = runs
    data_b["runs"] = runs
    return


@app.cell
def _(mo):
    seed_for_runs = mo.ui.number(label="RNG seed: ")
    return (seed_for_runs,)


@app.cell
def _(mo, seed_for_runs):
    mo.vstack(
        [seed_for_runs]
    )
    return


@app.cell
def _(np, seed_for_runs):
    rng = np.random.default_rng(seed=seed_for_runs.value)
    return (rng,)


@app.cell
def _(rng):
    runs_to_compare = rng.integers(low=1, high=51, size=6)
    return (runs_to_compare,)


@app.cell
def _(mo, runs_to_compare):
    mo.Html(f"We will be summarizing runs: {runs_to_compare}")
    return


@app.cell
def _(runs_to_compare):
    column_runs_compare = ["Run_"+str(run) for run in runs_to_compare]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Distribution of characteristics
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Boundary comparisons
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
