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

    return (plt,)


@app.cell
def _():
    from py_group_sequential_designs import simulate as sim

    return (sim,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Trial properties
    """)
    return


@app.cell
def _():
    num_analyses = 3
    target_alpha = 0.05
    target_power = 0.9
    delta0 = 0
    delta1 = 1.0
    sigma2 = 3.0
    return delta0, delta1, num_analyses, sigma2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Generate example bounds with sliders
    """)
    return


@app.cell
def _(mo):
    upper_bound1 = mo.ui.slider(start=0, stop=3, step=0.001, 
                                label="Upper bound 1", orientation="horizontal",
                                show_value=True)
    upper_bound1
    return (upper_bound1,)


@app.cell
def _(mo):
    upper_bound2 = mo.ui.slider(start=0, stop=3, step=0.001, 
                                label="Upper bound 2", orientation="horizontal",
                                show_value=True)
    upper_bound2
    return (upper_bound2,)


@app.cell
def _(mo):
    upper_bound3 = mo.ui.slider(start=0, stop=3, step=0.001, 
                                label="Upper bound 3", orientation="horizontal",
                                show_value=True)
    upper_bound3
    return (upper_bound3,)


@app.cell
def _(mo):
    lower_bound1 = mo.ui.slider(start=-3, stop=2, step=0.001, 
                                label="Lower bound 1", orientation="horizontal",
                                show_value=True)
    lower_bound1
    return (lower_bound1,)


@app.cell
def _(mo):
    lower_bound2 = mo.ui.slider(start=-3, stop=2, step=0.001, 
                                label="Lower bound 2", orientation="horizontal",
                                show_value=True)
    lower_bound2
    return (lower_bound2,)


@app.cell
def _(mo):
    sample_size = mo.ui.slider(start=1, stop=50, step=1, 
                                label="sample size", orientation="horizontal",
                                show_value=True)
    sample_size
    return (sample_size,)


@app.cell
def _(
    delta0,
    delta1,
    lower_bound1,
    lower_bound2,
    num_analyses,
    sample_size,
    sigma2,
    sim,
    upper_bound1,
    upper_bound2,
    upper_bound3,
):
    sim.group_sequential_designs(
        n_analyses = num_analyses,
        upper_bounds = [upper_bound1.value, upper_bound2.value, upper_bound3.value],
        lower_bounds = [lower_bound1.value, lower_bound2.value, upper_bound3.value],
        n_patients = sample_size.value,
        null_hypothesis = delta0,
        alt_hypothesis = delta1,
        variance = sigma2
    )
    return


@app.cell
def _(
    lower_bound1,
    lower_bound2,
    plt,
    upper_bound1,
    upper_bound2,
    upper_bound3,
):
    _fig, _ax = plt.subplots()

    upper_bounds = [upper_bound1.value, upper_bound2.value, upper_bound3.value]
    lower_bounds = [lower_bound1.value, lower_bound2.value, upper_bound3.value]

    _ax.plot([1,2,3], upper_bounds, color = "red", lw = 2)
    _ax.plot([1,2,3], lower_bounds, color = "red", lw = 2)

    _ax.set_ylim(-4,3)

    _fig
    return


if __name__ == "__main__":
    app.run()
