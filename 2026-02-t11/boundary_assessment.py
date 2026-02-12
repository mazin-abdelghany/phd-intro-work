import marimo

__generated_with = "0.19.10"
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

    return (sim,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step penalty - Squared exponential
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Input:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```
    kernel = gpflow.kernels.SquaredExponential()
    kernel.variance = gpflow.Parameter(value = 100, trainable = False)
    kernel.lengthscales = gpflow.Parameter(value = [1, 1, 1, 1, 1, 50], trainable = True)

    likelihood = gpflow.likelihoods.Gaussian()
    likelihood.variance = gpflow.Parameter(value = 1e-1, trainable = False)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Output:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```
    array([ 2.10857995, -1.19535156, -5.21828924,  0.49096243,  0.75162218,
            8.08091176])
    ```
    """)
    return


@app.cell
def _(sim):
    # boundaries "corrected" for order
    sim.group_sequential_designs(
        upper_bounds = [ 0.49096243,  0.75162218, 2.10857995 ],
        lower_bounds = [-5.21828924, -1.19535156,   2.10857995],
        n_patients = 8,
        alt_hypothesis = 1,
        variance = 3
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step penalty - Matern 52
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Input:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```
    mat_kernel = gpflow.kernels.Matern52()
    mat_kernel.variance = gpflow.Parameter(value = 100, trainable = True)
    mat_kernel.lengthscales = gpflow.Parameter(value = [1, 1, 1, 1, 1, 50], trainable = True)

    mat_likelihood = gpflow.likelihoods.Gaussian()
    mat_likelihood.variance = gpflow.Parameter(value = 1e-1, trainable = False)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Output:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```
    array([-6.        , -6.        ,  1.04320073,  5.19429949,  3.1639038 ,
            4.        ])
    ```
    """)
    return


@app.cell
def _(sim):
    # boundaries "corrected" for order
    sim.group_sequential_designs(
        upper_bounds = [5.19429949,  3.1639038 ,  1.04320073],
        lower_bounds = [-6, -6,   1.04320072],
        n_patients = 4,
        alt_hypothesis = 1,
        variance = 3
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Smooth penalty - Matern 52
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Input:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```
    sm_mat_kernel = gpflow.kernels.Matern52()
    sm_mat_kernel.variance = gpflow.Parameter(value = 100, trainable = True)
    sm_mat_kernel.lengthscales = gpflow.Parameter(value = [1, 1, 1, 1, 1, 10], trainable = True)

    sm_mat_likelihood = gpflow.likelihoods.Gaussian()
    sm_mat_likelihood.variance = gpflow.Parameter(value = 1, trainable = False)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Output:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```
    array([ 5.11394986, -2.136278  , -4.65706045, -1.79768349,  0.86269316,
            6.99561733])
    ```
    """)
    return


@app.cell
def _(sim):
    # boundaries "corrected" for order
    sim.group_sequential_designs(
        upper_bounds = [ -1.79768349,  0.86269316, 5.11394986],
        lower_bounds = [-4.65706045, -2.136278,  5.11394986],
        n_patients = 8,
        alt_hypothesis = 1,
        variance = 3
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
