import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from py_group_sequential_designs import boundaries as bd
    from py_group_sequential_designs import feasibility_penalty as fp
    from py_group_sequential_designs import format_boundaries_after_ask as fmt_bd
    from py_group_sequential_designs import function_to_minimize as fn_min
    from py_group_sequential_designs import generate_gpr_input as gen_input
    from py_group_sequential_designs import simulate as sim
    from py_group_sequential_designs import sample_size as ss

    return bd, sim


@app.cell
def _(sim):
    sim.group_sequential_designs()
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell
def _(pd):
    trial_designs = pd.read_csv(
        filepath_or_buffer="/tf/cpp_workspace/wason_delta_minimax/minimax_tests/runs_modified_code.txt",
        index_col=False,
        sep=" "
    )
    return (trial_designs,)


@app.cell
def _(trial_designs):
    trial_designs
    return


@app.cell
def _(trial_designs):
    trial_designs["power"].describe()
    return


@app.cell
def _(trial_designs):
    trial_designs["typeIerror"].describe()
    return


@app.cell
def _(trial_designs):
    trial_designs["upper1"].describe()
    return


@app.cell
def _(trial_designs):
    trial_designs["lower1"].describe()
    return


@app.cell
def _(trial_designs):
    trial_designs["lower1"].hist(bins=50)
    return


@app.cell
def _(trial_designs):
    trial_designs["upper1"].hist(bins=50)
    return


@app.cell
def _(bd):
    bd.calculate_triangular_boundaries()
    return


@app.cell
def _(plt, trial_designs):
    fig, ax = plt.subplots()

    for i in range(5):
        ax.plot([1,2,3], trial_designs.loc[i, ["upper1", "upper2", "lower3"]])
        ax.plot([1,2,3], trial_designs.loc[i, ["lower1", "lower2","lower3"]])

    ax.plot([1,2,3], [2.11957748, 1.87345951, 1.83560794], lw=3, color="red")
    ax.plot([1,2,3], [6.28553399e-16, 1.12407571e+00, 1.83560794e+00], lw=3, color="red")

    fig
    return


@app.cell
def _(mo):
    slider1 = mo.ui.slider(start=0, stop=112, label="Slider")
    return (slider1,)


@app.cell
def _(slider1):
    slider1
    return


@app.cell
def _(plt, slider1, trial_designs):
    _fig, _ax = plt.subplots()

    _ax.plot([1,2,3], [2.11957748, 1.87345951, 1.83560794], lw=3, color="orange")
    _ax.plot([1,2,3], [6.28553399e-16, 1.12407571e+00, 1.83560794e+00], lw=3, color="orange")

    _ax.plot([1,2,3], trial_designs.loc[slider1.value, ["upper1", "upper2", "lower3"]],
             color = "purple", lw=2)
    _ax.plot([1,2,3], trial_designs.loc[slider1.value, ["lower1", "lower2","lower3"]],
             color = "purple", lw=2)

    _ax.set_ylim([-0.3,2.5])

    _fig
    return


@app.cell
def _(pd):
    old_trial_designs = pd.read_csv(
        filepath_or_buffer="/tf/cpp_workspace/wason_delta_minimax/minimax_tests/old_code_runs.txt",
        index_col=False,
        sep=" "
    )
    return (old_trial_designs,)


@app.cell
def _(old_trial_designs):
    old_trial_designs
    return


@app.cell
def _(old_trial_designs):
    old_trial_designs["power"].describe()
    return


@app.cell
def _(old_trial_designs):
    old_trial_designs["typeIerror"].describe()
    return


@app.cell
def _(old_trial_designs):
    old_trial_designs["upper1"].describe()
    return


@app.cell
def _(old_trial_designs):
    old_trial_designs["lower1"].describe()
    return


@app.cell
def _(old_trial_designs):
    old_trial_designs["lower1"].hist(bins=50)
    return


@app.cell
def _(old_trial_designs):
    old_trial_designs["upper1"].hist(bins=50)
    return


@app.cell
def _(old_trial_designs, plt):
    _fig, _ax = plt.subplots()

    for _i in range(10):
        _ax.plot([1,2,3], old_trial_designs.loc[_i, ["upper1", "upper2", "upper3"]])
        _ax.plot([1,2,3], old_trial_designs.loc[_i, ["lower1", "lower2","lower3"]])

    _ax.plot([1,2,3], [2.11957748, 1.87345951, 1.83560794], lw=3, color="red")
    _ax.plot([1,2,3], [6.28553399e-16, 1.12407571e+00, 1.83560794e+00], lw=3, color="red")

    _fig
    return


@app.cell
def _(mo):
    slider2 = mo.ui.slider(start=0, stop=99, label="Slider")
    return (slider2,)


@app.cell
def _(slider2):
    slider2
    return


@app.cell
def _(old_trial_designs, plt, slider2):
    _fig, _ax = plt.subplots()

    _ax.plot([1,2,3], [2.11957748, 1.87345951, 1.83560794], lw=3, color="orange")
    _ax.plot([1,2,3], [6.28553399e-16, 1.12407571e+00, 1.83560794e+00], lw=3, color="orange")

    _ax.plot([1,2,3], old_trial_designs.loc[slider2.value, ["upper1", "upper2", "lower3"]],
             color = "purple", lw=2)
    _ax.plot([1,2,3], old_trial_designs.loc[slider2.value, ["lower1", "lower2","lower3"]],
             color = "purple", lw=2)

    _ax.set_ylim([-0.3,2.5])

    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
