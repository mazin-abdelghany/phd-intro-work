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

    return (plt,)


@app.cell
def _():
    import pandas as pd

    return (pd,)


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


if __name__ == "__main__":
    app.run()
