import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    return go, np, plt


@app.cell
def _(plt):
    plt.rcParams["figure.dpi"]=400
    return


@app.cell
def _():
    target_alpha = 0.05
    target_beta = 0.1
    return target_alpha, target_beta


@app.cell
def _(np):
    alpha_grid = np.linspace(0, 1, 300)
    beta_grid = np.linspace(0, 1, 300)
    return alpha_grid, beta_grid


@app.function
# vectorized, will need to be modified for loops
def flat_range_obj(alpha, target_alpha, epsilon=0.01):
    obj_value = abs(alpha-target_alpha)
    mask = (abs(alpha-target_alpha) <= epsilon)
    obj_value[mask] = 0
    return obj_value


@app.function
# vectorized, will need to be modified for loops
def must_be_at_target_obj(alpha, target_alpha, epsilon=0.01):
    obj_value = abs(alpha-target_alpha)
    mask = ((-epsilon <= alpha - target_alpha) & (alpha-target_alpha <= 0))
    obj_value[mask] = 0
    return obj_value


@app.cell
def _(alpha_grid, plt, target_alpha):
    plt.plot(alpha_grid, abs(target_alpha-alpha_grid))
    plt.plot(alpha_grid, (target_alpha-alpha_grid)**2)
    plt.plot(alpha_grid, flat_range_obj(alpha_grid, target_alpha, epsilon=0.01))
    plt.plot(alpha_grid, must_be_at_target_obj(alpha_grid, target_alpha, epsilon=0.01))
    plt.xlim(0,0.2)
    plt.ylim(0,0.1)
    plt.show()
    return


@app.function
def objective1_loop(
    alpha, 
    beta,
    target_alpha,
    target_beta,
    alpha_epsilon = 0.01,
    beta_epsilon = 0.05
):

    alpha_met = (-alpha_epsilon <= alpha - target_alpha) & (alpha-target_alpha <= 0)
    beta_met = (-beta_epsilon <= beta - target_beta) & (beta-target_beta <= 0)

    if (alpha_met and beta_met):
        return 0
    else:
        return 150*(abs(alpha-target_alpha) + abs(beta-target_beta))


@app.cell
def _(alpha_grid, beta_grid, np, target_alpha, target_beta):
    objective1 = np.empty(shape=(len(alpha_grid), len(beta_grid)))
    for _i, _alpha in enumerate(alpha_grid):
        for _j, _beta in enumerate(beta_grid):
            objective1[_i,_j] = objective1_loop(_alpha, _beta, target_alpha, target_beta)
    return (objective1,)


@app.cell
def _(alpha_grid, beta_grid, np):
    X,Y=np.meshgrid(alpha_grid, beta_grid, indexing="ij")
    return X, Y


@app.cell
def _(X, Y, go, objective1):
    _fig = go.Figure(data=[go.Surface(z=objective1, x=X, y=Y)])
    _fig.show()
    return


@app.function
def objective2_loop(
    alpha, 
    beta,
    target_alpha,
    target_beta,
    alpha_epsilon = 0.01,
    beta_epsilon = 0.05
):

    return (alpha-target_alpha)**2 + (beta-target_beta)**2


@app.cell
def _(alpha_grid, beta_grid, np, target_alpha, target_beta):
    objective2 = np.empty(shape=(len(alpha_grid), len(beta_grid)))
    for _i, _alpha in enumerate(alpha_grid):
        for _j, _beta in enumerate(beta_grid):
            objective2[_i,_j] = objective2_loop(_alpha, _beta, target_alpha, target_beta)
    return (objective2,)


@app.cell
def _(X, Y, go, objective2):
    _fig = go.Figure(data=[go.Surface(z=objective2, x=X, y=Y)])
    _fig.show()
    return


@app.function
def objective3_loop(
    alpha, 
    beta,
    target_alpha,
    target_beta,
    alpha_epsilon = 0.01,
    beta_epsilon = 0.05
):

    return 150*((alpha-target_alpha)**4 + (beta-target_beta)**4)


@app.cell
def _(alpha_grid, beta_grid, np, target_alpha, target_beta):
    objective3 = np.empty(shape=(len(alpha_grid), len(beta_grid)))
    for _i, _alpha in enumerate(alpha_grid):
        for _j, _beta in enumerate(beta_grid):
            objective3[_i,_j] = objective3_loop(_alpha, _beta, target_alpha, target_beta)
    return (objective3,)


@app.cell
def _(X, Y, go, objective1, objective3):
    _fig = go.Figure(data=[
        go.Surface(z=objective3, x=X, y=Y, coloraxis="coloraxis"),
        #go.Surface(z=objective2, x=X, y=Y),
        go.Surface(z=objective1, x=X, y=Y, coloraxis="coloraxis")
    ]
                    )
    _fig.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
