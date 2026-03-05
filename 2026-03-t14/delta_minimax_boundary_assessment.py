import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return np, pd, plt


@app.cell
def _(pd):
    trial_designs = pd.read_csv(
        filepath_or_buffer="/tf/2026-03-t14/candidate_designs.txt",
        index_col=False,
        sep=" "
    )
    return (trial_designs,)


@app.cell
def _(trial_designs):
    trial_designs
    return


@app.cell
def _(mo):
    slider1 = mo.ui.slider(start=0, stop=99, label="Slider")
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
    large_sample = pd.read_csv(
        filepath_or_buffer="/tf/cpp_workspace/wason_delta_minimax/minimax_tests/1000_runs.txt",
        index_col=False,
        sep=" "
    )
    return (large_sample,)


@app.cell
def _(large_sample):
    large_sample
    return


@app.cell
def _(large_sample):
    large_sample["typeIerror"].describe()
    return


@app.cell
def _(large_sample):
    large_sample["power"].describe()
    return


@app.cell
def _(large_sample):
    large_sample["upper3"].describe()
    return


@app.cell
def _():
    #                    0          1        2         3         4         5
    columns_to_grab = ['lower1', 'upper1', 'lower2', 'upper2', 'lower3', 'upper3']
    axes_rows = [1, 0, 1, 0, 1, 0]
    axes_cols = [0, 0, 1, 1, 2, 2]
    return axes_cols, axes_rows, columns_to_grab


@app.cell
def _(axes_cols, axes_rows, columns_to_grab, large_sample, plt):
    _fig, _ax = plt.subplots(nrows=2, ncols=3, sharey=True)

    #   0 (0, 0)  2 (0, 1)   4 (0, 2)
    #   1 (1, 0)  3 (1, 1)   5 (1, 2)

    for _i, _column in enumerate(columns_to_grab):
        _ax[axes_rows[_i], axes_cols[_i]].hist(large_sample[_column], bins=50)
        _ax[axes_rows[_i], axes_cols[_i]].set_title([_column])

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(large_sample, plt):
    _fig, _ax = plt.subplots()

    for i in range(1000):
        _ax.plot([1,2,3], large_sample.loc[i, ["upper1", "upper2", "lower3"]],
                 color = "purple", alpha=0.05)
        _ax.plot([1,2,3], large_sample.loc[i, ["lower1", "lower2","lower3"]],
                 color = "purple", alpha=0.05)

    _ax.plot([1,2,3], [2.11957748, 1.87345951, 1.83560794], color="orange")
    _ax.plot([1,2,3], [6.28553399e-16, 1.12407571e+00, 1.83560794e+00], color="orange")

    _fig
    return


@app.cell
def _(large_sample):
    large_sample.columns
    return


@app.cell
def _(columns_to_grab, large_sample, np, pd):
    # get quartiles for plotting for all of the bounds
    quartiles_plotting = pd.DataFrame()

    for _column in columns_to_grab:
        quartiles_plotting[_column] = np.percentile(large_sample[_column], [2.5, 50, 97.5])
    return (quartiles_plotting,)


@app.cell
def _(quartiles_plotting):
    quartiles_plotting.iloc[:, 0]
    return


@app.cell
def _(columns_to_grab, large_sample, plt, quartiles_plotting):
    _fig, _ax = plt.subplots()

    positions = [1, 2, 3]

    for position in positions:
        _ax.violinplot(large_sample[columns_to_grab[position-1]], positions=[position],
                       showmeans=False, 
                       showmedians=False,
                       showextrema=False)
        _ax.scatter(position, quartiles_plotting.iloc[1, position-1], marker='o', color='white', s=30, zorder=3)
        _ax.vlines(position, quartiles_plotting.iloc[0, position-1], 
                   quartiles_plotting.iloc[2, position-1], color='black', linestyle='-', lw=5)

        _ax.violinplot(large_sample[columns_to_grab[position]], positions=[position],
                       showmeans=False, 
                       showmedians=False,
                       showextrema=False)
        _ax.scatter(position, quartiles_plotting.iloc[1, position], marker='o', color='white', s=30, zorder=3)
        _ax.vlines(position, quartiles_plotting.iloc[0, position], 
                   quartiles_plotting.iloc[2, position], color='black', linestyle='-', lw=5)

        _ax.scatter(position, large_sample[columns_to_grab[position-1]].max(), marker="_", color='black', s=30, zorder=3)
        _ax.scatter(position, large_sample[columns_to_grab[position]].max(), marker="_", color='black', s=30, zorder=3)

        _ax.scatter(position, large_sample[columns_to_grab[position-1]].min(), marker="_", color='black', s=30, zorder=3)
        _ax.scatter(position, large_sample[columns_to_grab[position]].min(), marker="_", color='black', s=30, zorder=3)

    _fig
    return


if __name__ == "__main__":
    app.run()
