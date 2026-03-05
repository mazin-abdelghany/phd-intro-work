import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo._runtime.context.get_context().marimo_config["runtime"]["output_max_bytes"] = 10000000000
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    return go, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Objective function
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The penalty function used as the objective to find the best design parameter boundaries is set using the following:

    1. Calculate the trial properties for the candidate design, includes.
       - $\alpha_{\texttt{found}}$
       - $\beta_{\texttt{found}}$
    3. Penalty parameter is instantiated, $p$, set to the single stage sample size.
    4. Number of restarts, $r$
    5. Worst case expected sample size, $ESS_w$

    \[
    f = \mathbb{I}\{\alpha_{\texttt{found}} > \alpha\} \left(p + p\left(\frac{\alpha_{\texttt{found}}-\alpha}{\alpha}\right)\right) + \mathbb{I}\{\beta_{\texttt{found}} > \beta\}\left(p + p\left(\frac{\beta_{\texttt{found}}-\beta}{\beta}\right)\right) +
    \]

    \[
    \mathbb{I}\{(\alpha_{\texttt{found}} > \alpha\,\,\texttt{or}\,\,\beta_{\texttt{found}} > \beta) \,\,\texttt{and}\,\,(r\ge-1)\}\left(\frac{p}{10}\right) + ESS_w
    \]
    """)
    return


@app.function
def objective_function(
        alpha,
        alpha_found,
        beta,
        beta_found,
        p,
        r,
        ess_worst
):
    function_value = 0

    if alpha_found > alpha:
        function_value += p * (p * ((alpha_found-alpha)/alpha) )
    if beta_found > beta:
        function_value += p * (p * ((beta_found-beta)/beta) )
    if (alpha_found > alpha or beta_found > beta) and r >= -1:
        function_value += p/10

    function_value += ess_worst

    return function_value


@app.cell
def _(np):
    alpha_found = np.linspace(0, 1, 200)
    beta_found = np.copy(alpha_found)

    p = 154
    r = -2
    return alpha_found, beta_found, p, r


@app.cell
def _(alpha_found):
    len(alpha_found)
    return


@app.cell
def _(alpha_found, beta_found, np):
    objective_function_z = np.empty(shape=(len(alpha_found), len(beta_found)))
    return (objective_function_z,)


@app.cell
def _(objective_function_z):
    objective_function_z.shape
    return


@app.cell
def _(alpha_found, beta_found, objective_function_z, p, r):
    # create values when r is -2
    for _i, alpha_val in enumerate(alpha_found):
        for _j, beta_val in enumerate(beta_found):
            objective_function_z[_i, _j] = objective_function(
                alpha = 0.05,
                alpha_found = alpha_val,
                beta = 0.1,
                beta_found = beta_val, 
                p = p, 
                r = r, 
                ess_worst= 120
            )
    return


@app.cell
def _(alpha_found, beta_found, np):
    objective_function_z_r1 = np.ones(shape=(len(alpha_found), len(beta_found))) * 10000
    return (objective_function_z_r1,)


@app.cell
def _(alpha_found, beta_found, objective_function_z_r1, p):
    # create values when r is 1
    for _i, _alpha_val in enumerate(alpha_found):
        for _j, _beta_val in enumerate(beta_found):
            objective_function_z_r1[_i, _j] = objective_function(
                alpha = 0.05,
                alpha_found = _alpha_val,
                beta = 0.1,
                beta_found = _beta_val, 
                p = p, 
                r = 1, 
                ess_worst= 120
            )
    return


@app.cell
def _(alpha_found, beta_found, np):
    X, Y = np.meshgrid(alpha_found, beta_found)
    return X, Y


@app.cell
def _(X, Y, go, objective_function_z, objective_function_z_r1):
    fig_3d = go.Figure()

    fig_3d.add_trace(go.Surface(z=objective_function_z, x=X, y=Y, name="Surface 1"))
    fig_3d.add_trace(go.Surface(z=objective_function_z_r1, x=X, y=Y, name="Surface 2"))

    fig_3d.update_scenes(
        xaxis_title_text="alpha",
        yaxis_title_text="beta",
        zaxis_title_text="penalty"
    )
    fig_3d.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simulated annealing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loop 1
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the simulated annealing function, the steps are:

    Before the first loop within the simulated annealing code, the initial design is set as the triangular design, call it $D$, and $f_{\texttt{min}}$ is set as the current minimum objective function value calculated by $f$ above.

    Perform this <span style="color:red"><b>loop</span></b> $n_{\texttt{generate}}$ times:
    1. Generate a candidate design, $D'$
    2. Calculate the objective function value for this design, $f'$.
    3. Reduce the size of the search space $B$ by $\rho_{\texttt{sigma}}$
    4. Increment the counter for the number of generated designs (limit is 10,000)
    5. Perform simulated annealing:
        - Generate a $U(0, 1)$ random variable, $x$
        - <span style="color:red"><b>IF</span></b> $\exp\left\{-\frac{(f' - f)}{T}\right\}\ge x$:
            - Set $f'$ to $f$
            - Reduce $T$ by $\rho_{\texttt{cost}}$, calculated as $T\cdot\rho_{\texttt{cost}}$
            - Set the generated candidate design $D'$ to the current design $D$
            - <span style="color:red"><b>IF</span></b> $f' < f_{\texttt{min}}$:
                - Save the current design $D$ as the "best" design $D_{\texttt{min}}$
                - Set $f_{\texttt{min}}$ to $f'$
                - Reset a counter of number of loops since objective function reduction to 0
            - <span style="color:red"><b>ELSE</span></b> increment number since objective function reduction
        - <span style="color:red"><b>ELSE</span></b> increment number since objective function reduction
    6. Every 25th run, reset $D$ to $D_{\texttt{min}}$ and reset $f$ to $f_{\texttt{min}}$

    After $n_{\texttt{generate}}$ times:
    1. Set $D$ to $D_{\texttt{min}}$
    2. Set $f$ to $f_{\texttt{min}}$
    3. Reset $T$ to its starting value
    4. Reset box shrinker to starting value
    5. Reset $n_{\texttt{generate}}$ to 0
    6. Increment number of restarts

    Once these are reset, the loop restarts with the design $D$ set at the current minimum $D_{\texttt{min}}$. The simulated annealing temperature $T$ and the search space $B$ are reset and the search begins again. This entire loop is completed at least $n_{\texttt{restarts}}$ times.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loop 2
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    After the above loop is finished,

    1. The sample size changed to an integer using `floor()`.
    2. The minimum objective function value, $f_{\texttt{min}}$, is recalculated with this new integer sample size.
    3. $n_{\texttt{generate}}$ is reset to 0.
    4. $n_{\texttt{restarts}}$ is reduced by 4.

    The loop restarts with the sample size fixed exactly as above except that in step 6:
    6. Every 10th run, reset $D$ to $D_{\texttt{min}}$ and reset $f$ to $f_{\texttt{min}}$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The path to the minimum
    """)
    return


@app.cell
def _(pd):
    designs = pd.read_csv(filepath_or_buffer="/tf/2026-03-t14/designs.txt", sep=" ",
                          index_col=False,
                          names=["sample_size", "lower1", "upper1", "lower2", "upper2", "lower3", "upper3"])
    return (designs,)


@app.cell
def _(designs):
    designs
    return


@app.cell
def _(pd):
    objective_function_vals = pd.read_csv(filepath_or_buffer="/tf/2026-03-t14/objective_function_vals.txt", sep=" ",
                                          index_col=False, names = ["obj_func_val"])
    return (objective_function_vals,)


@app.cell
def _(pd):
    temperature = pd.read_csv(filepath_or_buffer="/tf/2026-03-t14/temperature.txt", sep=" ",
                                          index_col=False, names = ["temp"])
    return (temperature,)


@app.cell
def _(mo):
    slider = mo.ui.slider(start=1, stop=3500, label="Slider", value=3)
    return (slider,)


@app.cell
def _(slider):
    slider
    return


@app.cell
def _(designs, np, objective_function_vals, plt, slider, temperature):
    _fig, _ax = plt.subplots()

    _ax.plot([1,2,3], [2.11957748, 1.87345951, 1.83560794], lw=3, color="red")
    _ax.plot([1,2,3], [6.28553399e-16, 1.12407571e+00, 1.83560794e+00], lw=3, color="red")

    _ax.plot([1,2,3], designs.iloc[slider.value, [1, 3, 5]])
    _ax.plot([1,2,3], designs.iloc[slider.value, [2, 4, 6]])
    _ax.text(2.3, -2, str("temp = ")+str(np.array(temperature.iloc[slider.value])))
    _ax.text(2.3, -1.5, str("obj func = ")+str(np.array(objective_function_vals.iloc[slider.value])))
    _ax.set_ylim([-4, 4])

    _fig
    return


@app.cell
def _(objective_function_vals, plt, slider, temperature):
    _fig, _ax = plt.subplots(figsize=(15,6))

    _ax.plot(objective_function_vals)
    _ax2 = _ax.twinx()
    _ax2.plot(temperature, color = "orange", lw=3)
    _ax2.scatter(x = slider.value, y = temperature.iloc[slider.value], marker="o", color = "purple", s=100,
                 zorder = 3)

    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
