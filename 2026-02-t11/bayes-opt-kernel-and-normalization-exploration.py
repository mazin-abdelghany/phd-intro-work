import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To do:
    - Trial penalty function without step
    - Run Bayes opt without normalization
    - Try different kernels (Matern with and without step)
    - Check GPR model if all values are trainable (in full Bayes opt loop)
    - Try GPR model with low likelihood variance (low but not as close to zero as 1e-5)
    - Can consider training only first boundary values and force monotonicity with a functional form (similar to O’Brien-Fleming)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Penalty function without step
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bayes opt without normalization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Kernel exploration
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GPR model assessment
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GPR model with low likelihood variance
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Single boundaries with enforced monotonicity
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
