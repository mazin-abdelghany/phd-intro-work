import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bayes opt loop design with failure regions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Our optimization problem has an issue: designs that are not monotonic are not statistically possible. However, while exploring the search space, there is no restriction on monotonicity. In initial testing, penalizing monotonicity does improve model fit, but >90% of designs are still discarded because of non-monotonicity.

    In order to solve this issue, rather than penalizing monotonicity, we will define this as a failure region. The penalty function will return finite values for feasible designs and will return `np.nan` for designs that are statistically impossible.

    The Bayesian optimization loop will then fit two models: (1) a Gaussian process regression (GPR) model for the feasible designs and their objective function and (2) a variational Gaussian process (VPR) classification model with Bernoulli likelihood to model the failure region.

    The steps are as follows:
    1. Generate the 3 initial points, Pocock, O'Brien-Fleming, and triagular bounds and penalty values.
    2. The objective function $f(\cdot)\in\mathbb{R}^+$ takes as input the study design $D$ and outputs $y=\{f(D),1\}$ if the design is feasible and $y=\{\texttt{np.nan}, 0\}$ if the study design is statistically impossible.
    3. Fit the two models (1) GPR for the real-valued $f(D)$ and (2) a VPR for the indicator $\{0,1\}$ for statistical possibility.
    4. Run the Bayesian optimization loop

    The goal of this type of model is to increase the number of feasible designs output by the model in a way that differs from only penalizing statistically impossible designs with large $f(D)$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simplified example of failure region objective function output
    """)
    return


@app.cell
def _(np):
    # a simplified example of step 2's objective function
    def objective_function(D):
        if np.isfinite(D):
            return (3, 1)
        else:
            return (np.nan, 0)

    return (objective_function,)


@app.cell
def _(objective_function):
    # example output for a feasible design D1 = 3
    D1 = 3
    objective_function(D1)
    return


@app.cell
def _(np, objective_function):
    # example output for a statistically impossible desing D2 = np.nan
    D2 = np.nan
    objective_function(D2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The $\{0,1\}$ output is used to train the classification model and the real value is used to train the regression model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Setup required imports
    """)
    return


@app.cell
def _():
    # scientific computing imports
    import numpy as np
    import pandas as pd
    from scipy import stats
    from scipy import optimize
    import matplotlib.pyplot as plt

    return (np,)


@app.cell
def _():
    import tensorflow as tf

    return


@app.cell
def _():
    # imports for GP regression
    import gpflow

    return


@app.cell
def _():
    # imports for Bayes opt
    from trieste.ask_tell_optimization import (
        AskTellOptimizer,
        AskTellOptimizerNoTraining,
    )
    from trieste.bayesian_optimizer import Record
    from trieste.data import Dataset
    from trieste.models.gpflow.models import GaussianProcessRegression
    from trieste.objectives import ScaledBranin
    from trieste.objectives.utils import mk_observer
    from trieste.space import Box

    return


if __name__ == "__main__":
    app.run()
