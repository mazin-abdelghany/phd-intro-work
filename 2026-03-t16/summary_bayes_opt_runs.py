import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In Bayesian optimization, the goal is to optimize a blackbox function, $f$. In our case, this function takes inputs of
    $$
    D = \{n, u_1, \ell_1, u_2, \ell_2, \cdots, u_k, \ell_k \}
    $$

    where

    - $D$ is the design (an $n$-tuple set),
    - $n$ is the sample size at each analysis point (total for both groups),
    - $u_1$ and $\ell_1$ are the upper and lower bounds for the first stage, and
    - $k$ is the total number of stages

    and outputs several values of interest, $T$ (an $m$-tuple set), including (but not limited to)

    - $\alpha$, the type I error
    - $\beta$, the type II error
    - $\mathbb{E}[N \, | \, \boldsymbol{\delta}]$, the expected sample size (ESS) $N$ over a range of differences, $\delta$, between groups $\boldsymbol{\delta} = \{\delta_1, \delta_2, \dots, \delta_j \}$

    Above, $N = k*n$ and

    The expected sample size is calculated across a set of possible true treatment effects $\boldsymbol{\delta} = \{\delta_1, \delta_2, \dots, \delta_j \}$ as a function of the design elements $D$: (1) number of analyses $\{1, 2, \dots, k\}$, (2) number of patients at each analysis $\mathbf{n} = \{n_1, n_2, \dots, n_k\}$, and (3) the upper and lower bounds $\mathbf{u} = (u_1, u_2, \dots, u_k)$ and $\boldsymbol{\ell} = (\ell_1, \ell_2, \dots, \ell_k)$.

    It is calculated as
    $$
    \mathbb{E}[N \, | \, \boldsymbol{\delta}]=\sum_{i=1}^k n_i P(\text{trial stops after analysis }i \, | \, \boldsymbol{\delta})
    $$

    In order to generated an optimized clinical trial design, the following steps must be followed:
    1. Generate several feasible trial designs, $\boldsymbol{D}=\{D_1, D_2, \cdots, D_p\}$.
    2. Generate the corresponding outputs, $\boldsymbol{T}=\{T_1, T_2, \cdots, T_p\}$.
    3. Fit a Gaussian process regression model to estimate the blackbox function $f: D \to T$.
    4. Perform a step of Bayesian optimization to find the next trial design of interest $D_i$.
    5. Obtain the corresponding outputs that correspond to this design $T_i$.
    6. Refit the Gaussian process regression model on the new data $n$-tuples $\{(\boldsymbol{D}, \boldsymbol{T}), (D_i, T_i)\}$

    Repeat until termination policy is reached.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order to generated an optimized clinical trial design, the following steps must be followed:

    1. Generate several feasible trial designs, $\boldsymbol{D}=\{D_1, D_2, \cdots, D_p\}$.
    2. Generate the corresponding outputs, $\boldsymbol{T}=\{T_1, T_2, \cdots, T_p\}$.
    3. Fit a Gaussian process regression model to estimate the blackbox function $f: D \to T$.
    4. Perform a step of Bayesian optimization to find the next trial design of interest $D_i$.
    5. Obtain the corresponding outputs that correspond to this design $T_i$.
    6. Refit the Gaussian process regression model on the new data $n$-tuples $\{(\boldsymbol{D}, \boldsymbol{T}), (D_i, T_i)\}$

    Repeat until termination policy is reached.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In all of the below runs, the following was true.

    **The data presented below were generated using the following settings:**
    - 3-stage design
    - Target $\alpha = 0.05$
    - Target $\beta = 0.1$, or power of 0.9
    - Clinically meaningful difference, $\delta=1$
    - Assumed variance, $\sigma^2=3$

    **Objective function (loss):**
    $$
    \mathcal{L}(T_i,\mu) =\mu (\alpha' - \alpha)^2 + \mu(\beta' - \beta)^2 + \frac{\max\big(\mathbb{E}[N \, | \, \boldsymbol{\delta}]\big)}{\mu}
    $$
    where:
    - $T_i$ is a vector that contains the design's calculated type I and type II error, and the per stage N, that is, $T_i=\{\alpha',\beta',N\}$.
    - $\mu$ is the single stage sample size for the design.
    - $\alpha$ is the target type I error.
    - $\beta$ is the target type II error.
    - $\mathbb{E}[N \, | \, \boldsymbol{\delta}]$ is defined above.
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
    from py_group_sequential_designs import feasibility_penalty as fp
    from py_group_sequential_designs import boundary_manipulations as fmt_bd
    from py_group_sequential_designs import function_to_minimize as fn_min
    from py_group_sequential_designs import generate_gpr_input as gen_input
    from py_group_sequential_designs import simulate as sim
    from py_group_sequential_designs import sample_size as ss

    return (fmt_bd,)


@app.cell
def _(plt):
    # get colors for plotting
    colors = plt.cm.tab20.colors
    return (colors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # February data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Small box
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayesian optimization settings
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - GPR kernel was Matern52
    - Upper bound of search space was `[ 3,  2.9,  2.8,  2.7, 2.6, 60]`
    - Lower bound of search space was `[-3, -2.9, -2.8, -2.7, -2.6, 5]`
    - `acquisition_rule = EfficientGlobalOptimization`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Import the data
    """)
    return


@app.cell
def _(pd):
    small_box_feb_bounds = pd.read_csv(
        filepath_or_buffer="/tf/2026-02-t12/small_box_boundaries.csv",
        header=0,
        names=["upper1","upper2","upper3","lower1","lower2","n"]
    )
    return (small_box_feb_bounds,)


@app.cell
def _(pd):
    small_box_feb_obj_f = pd.read_csv(
        filepath_or_buffer="/tf/2026-02-t12/small_box_penalty.csv",
        header=0,
        names=["obj_f"]
    )
    return (small_box_feb_obj_f,)


@app.cell
def _(pd, small_box_feb_bounds, small_box_feb_obj_f):
    small_box_feb = pd.concat([small_box_feb_bounds, small_box_feb_obj_f], axis=1)
    return (small_box_feb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check for monotonicity
    """)
    return


@app.cell
def _(fmt_bd, np, small_box_feb):
    _monotonic = []

    for _i in range(small_box_feb.shape[0]):
        _bounds = small_box_feb.loc[_i, ["upper1","upper2","upper3","lower1","lower2","n"]]
        _bounds_arr = np.array(_bounds)
        _monotonic.append(fmt_bd.check_monotonicity(n_analyses = 3, bounds = _bounds_arr))

    small_box_feb["monotonic"] = _monotonic
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analyze only the monotonic bounds
    """)
    return


@app.cell
def _(small_box_feb):
    small_box_feb_remove_known = small_box_feb.iloc[3:]
    return (small_box_feb_remove_known,)


@app.cell
def _(small_box_feb_remove_known):
    small_box_feb_remove_known[small_box_feb_remove_known["monotonic"] == True]["obj_f"].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Of 1000 runs, 12 monotonic, minimum penalty 3.06 - 121
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Big box
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayesian optimization settings
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - GPR kernel was Matern52
    - Upper bound of search space was `[ 6,  6,  6,  6,  6, 60]`
    - Lower bound of search space was `[-6, -6, -6, -6, -6, 2]`
    - `acquisition_rule = EfficientGlobalOptimization`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Import the data
    """)
    return


@app.cell
def _(pd):
    big_box_feb_bounds = pd.read_csv(
        filepath_or_buffer="/tf/2026-02-t12/box6-6-2-60_boundaries.csv",
        header=0,
        names=["upper1","upper2","upper3","lower1","lower2","n"]
    )
    return (big_box_feb_bounds,)


@app.cell
def _(pd):
    big_box_feb_obj_f = pd.read_csv(
        filepath_or_buffer="/tf/2026-02-t12/box6-6-2-60_penalty.csv",
        header=0,
        names=["obj_f"]
    )
    return (big_box_feb_obj_f,)


@app.cell
def _(big_box_feb_bounds, big_box_feb_obj_f, pd):
    big_box_feb = pd.concat([big_box_feb_bounds, big_box_feb_obj_f], axis=1)
    return (big_box_feb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check for monotonicity
    """)
    return


@app.cell
def _(big_box_feb, fmt_bd, np):
    _monotonic = []

    for _i in range(big_box_feb.shape[0]):
        _bounds = big_box_feb.loc[_i, ["upper1","upper2","upper3","lower1","lower2","n"]]
        _bounds_arr = np.array(_bounds)
        _monotonic.append(fmt_bd.check_monotonicity(n_analyses = 3, bounds = _bounds_arr))

    big_box_feb["monotonic"] = _monotonic
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analyze only the monotonic bounds
    """)
    return


@app.cell
def _(big_box_feb):
    big_box_feb_remove_known = big_box_feb.iloc[3:].reset_index()
    return (big_box_feb_remove_known,)


@app.cell
def _(big_box_feb_remove_known):
    big_box_feb_remove_known[big_box_feb_remove_known["monotonic"] == True]["obj_f"].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Of 1000 runs, 11 monotonic, minimum penalty 1.64 - 2.73
    """)
    return


@app.cell
def _(big_box_feb_remove_known):
    big_box_feb_best = big_box_feb_remove_known[
        (big_box_feb_remove_known["monotonic"] == True) & (big_box_feb_remove_known["obj_f"] < 3)
    ].reset_index()
    return (big_box_feb_best,)


@app.cell
def _(big_box_feb_best):
    big_box_feb_best.shape
    return


@app.cell
def _(big_box_feb_best, colors, plt):
    _fig, _ax = plt.subplots()

    for _i in range(big_box_feb_best.shape[0]):
        _ax.plot([1,2,3], big_box_feb_best.loc[_i, ["upper1", "upper2", "upper3"]], color = colors[_i])
        _ax.plot([1,2,3], big_box_feb_best.loc[_i, ["lower1", "lower2","upper3"]], color = colors[_i])

    _ax.plot([1,2,3], [2.11957748, 1.87345951, 1.83560794], lw=3, color="red")
    _ax.plot([1,2,3], [6.28553399e-16, 1.12407571e+00, 1.83560794e+00], lw=3, color="red")

    _fig
    return


@app.cell
def _(big_box_feb_best, mo):
    slider1 = mo.ui.slider(start=0, stop=big_box_feb_best.shape[0]-1, step=1)
    slider1
    return (slider1,)


@app.cell
def _(big_box_feb_best, plt, slider1):
    plt.plot([1,2,3], big_box_feb_best.loc[slider1.value, ["upper1", "upper2", "upper3"]])
    plt.plot([1,2,3], big_box_feb_best.loc[slider1.value, ["lower1", "lower2","upper3"]])

    plt.plot([1,2,3], [2.11957748, 1.87345951, 1.83560794], lw=3, color="red")
    plt.plot([1,2,3], [6.28553399e-16, 1.12407571e+00, 1.83560794e+00], lw=3, color="red")

    plt.ylim(-6, 6)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # March data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Big box with monotonic penalty
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayesian optimization settings
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Penalty modified to include $\mathcal{L}=25$ if not monotonic, else same as above.
    - GPR kernel was Matern52
    - Upper bound of search space was `[ 3, 3, 3, 3, 3, 30]`
    - Lower bound of search space was `[-3,-3,-3,-3,-3, 10]`
    - `acquisition_rule = EfficientGlobalOptimization`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Import the data
    """)
    return


@app.cell
def _(pd):
    big_box_march = pd.read_csv(
        filepath_or_buffer="/tf/2026-03-t15/data/big_box_monotonicity.csv",
        header=0,
        names=["upper1","upper2","upper3","lower1","lower2","n", "obj_f"]
    )
    return (big_box_march,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check for monotonicity
    """)
    return


@app.cell
def _(big_box_march, fmt_bd, np):
    _monotonic = []

    for _i in range(big_box_march.shape[0]):
        _bounds = big_box_march.loc[_i, ["upper1","upper2","upper3","lower1","lower2","n"]]
        _bounds_arr = np.array(_bounds)
        _monotonic.append(fmt_bd.check_monotonicity(n_analyses = 3, bounds = _bounds_arr))

    big_box_march["monotonic"] = _monotonic
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analyze the monotonic bounds
    """)
    return


@app.cell
def _(big_box_march):
    big_box_march_remove_known = big_box_march.iloc[3:]
    return (big_box_march_remove_known,)


@app.cell
def _(big_box_march_remove_known):
    big_box_march_remove_known[big_box_march_remove_known["monotonic"] == True]["obj_f"].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Of 1790 runs, 11 monotonic, minimum penalty 1.14 - 138
    """)
    return


@app.cell
def _(big_box_march_remove_known):
    big_box_march_best = big_box_march_remove_known[
        (big_box_march_remove_known["monotonic"] == True) & (big_box_march_remove_known["obj_f"] < 3)
    ].reset_index()
    return (big_box_march_best,)


@app.cell
def _(big_box_march_best):
    big_box_march_best.shape
    return


@app.cell
def _(big_box_march_best, colors, plt):
    _fig, _ax = plt.subplots()

    for _i in range(big_box_march_best.shape[0]):
        _ax.plot([1,2,3], big_box_march_best.loc[_i, ["upper1", "upper2", "upper3"]], color = colors[_i])
        _ax.plot([1,2,3], big_box_march_best.loc[_i, ["lower1", "lower2","upper3"]], color = colors[_i])

    _ax.plot([1,2,3], [2.11957748, 1.87345951, 1.83560794], lw=3, color="red")
    _ax.plot([1,2,3], [6.28553399e-16, 1.12407571e+00, 1.83560794e+00], lw=3, color="red")

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pocock region with monotonicity
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayesian optimization settings
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Penalty modified to include $\mathcal{L}=25$ if not monotonic, else same as above.
    - GPR kernel was Matern52
    - Upper bound of search space was `[1.72998486,  1.72998486,  1.72998486, -2.25438538, -2.25438538, 10]`
    - Lower bound of search space was `[2.25438538,  2.25438538,  2.25438538, -1.72998486, -1.72998486, 30]`
    - `acquisition_rule = EfficientGlobalOptimization`
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Import the data
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
    ## Check for monotonicity
    """)
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
    ## Analyze the monotonic bounds
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pocock region without monotonicity
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayesian optimization settings
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - GPR kernel was Matern52
    - Upper bound of search space was `[1.72998486,  1.72998486,  1.72998486, -2.25438538, -2.25438538, 10]`
    - Lower bound of search space was `[2.25438538,  2.25438538,  2.25438538, -1.72998486, -1.72998486, 30]`
    - `acquisition_rule = EfficientGlobalOptimization`
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Import the data
    """)
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
    ## Check for monotonicity
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analyze the monotonic bounds
    """)
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
    # Separate upper and lower boxes - force monotonicity
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayesian optimization settings
    """)
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
    ## Import the data
    """)
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
    ## Check for monotonic bounds
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analyze the monotonic bounds
    """)
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
    # Triangular region with monotonicity
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayesian optimization settings
    """)
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
    ## Import the data
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check for monotonicity
    """)
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
    ## Analyze the monotonic bounds
    """)
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
    # Triangular region without monotonicity
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayesian optimization settings
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Import the data
    """)
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
    ## Check for monotonicity
    """)
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
    ## Analyze the monotonic bounds
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
    # Failure region Bayes opt
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run 1
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
    ## Run 2
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
    ## Run 3
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


if __name__ == "__main__":
    app.run()
