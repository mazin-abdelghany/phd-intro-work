import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


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

    1. The sample size is set to an integer.
    2. The minimum objective function value, $f_{\texttt{min}}$, is recalculated with this integer sample size.
    3. $n_{\texttt{generate}}$ is reset to 0.
    4. $n_{\texttt{restarts}}$ is reduced by 4.

    The loop restarts
    """)
    return


if __name__ == "__main__":
    app.run()
