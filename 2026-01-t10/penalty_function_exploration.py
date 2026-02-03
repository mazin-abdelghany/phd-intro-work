import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    import marimo as mo
    mo._runtime.context.get_context().marimo_config["runtime"]["output_max_bytes"] = 10000000000
    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import plotly.graph_objects as go
    from scipy import stats
    return cm, go, np, plt


@app.cell
def _():
    from py_group_sequential_designs import boundaries as bd
    from py_group_sequential_designs import feasibility_penalty as fp
    from py_group_sequential_designs import format_boundaries_after_ask as fmt_bd
    from py_group_sequential_designs import function_to_minimize as fn_min
    from py_group_sequential_designs import generate_gpr_input as gen_input
    from py_group_sequential_designs import simulate as sim
    from py_group_sequential_designs import sample_size as ss
    return fp, ss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploring the feasibility penalty
    """)
    return


@app.cell
def _(ss):
    # some set defaults
    num_analyses = 3
    target_alpha = 0.05
    target_power = 0.9
    important_diff_delta = 1
    assumed_variance = 3

    # to obtain mu (sample size at one stage)
    mu = ss.sample_size_means(
        ratio=1,
        variance=assumed_variance,
        power=target_power,
        alpha=target_alpha,
        delta=important_diff_delta
    )
    return mu, target_alpha, target_power


@app.cell
def _(np):
    # generate betas to loop over
    beta_range = np.arange(start=0.01, stop=0.99, step=0.001)
    return (beta_range,)


@app.cell
def _(beta_range, fp, mu, np, target_alpha, target_power):
    # generate several penalty lines
    penalty_plot_1 = np.empty(len(beta_range))
    for i_1, beta_1 in enumerate(beta_range):
        penalty_plot_1[i_1] = fp.feasibility_penalty(
            mu = mu,
            power = target_power,
            alpha = target_alpha,
            beta_prime = beta_1,
            alpha_prime = 0.9
        )

    penalty_plot_2 = np.empty(len(beta_range))
    for i_1, beta_1 in enumerate(beta_range):
        penalty_plot_2[i_1] = fp.feasibility_penalty(
            mu = mu,
            power = target_power,
            alpha = target_alpha,
            beta_prime = beta_1,
            alpha_prime = 0.5
        )

    penalty_plot_3 = np.empty(len(beta_range))
    for i_1, beta_1 in enumerate(beta_range):
        penalty_plot_3[i_1] = fp.feasibility_penalty(
            mu = mu,
            power = target_power,
            alpha = target_alpha,
            beta_prime = beta_1,
            alpha_prime = 0.1
        )

    penalty_plot_4 = np.empty(len(beta_range))
    for i_1, beta_1 in enumerate(beta_range):
        penalty_plot_4[i_1] = fp.feasibility_penalty(
            mu = mu,
            power = target_power,
            alpha = target_alpha,
            beta_prime = beta_1,
            alpha_prime = 0.01
        )
    return penalty_plot_1, penalty_plot_2, penalty_plot_3, penalty_plot_4


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A single penalty line
    """)
    return


@app.cell
def _(beta_range, penalty_plot_1, plt):
    # plot a single penalty line at alpha = 0.5
    fig_sing, ax_sing = plt.subplots()
    ax_sing.plot(beta_range, penalty_plot_1)
    ax_sing.set_title("Penalty function, alpha fixed at 0.5")
    ax_sing.set_ylabel("Penalty")
    ax_sing.set_xlabel("Beta")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multiple penalty lines
    """)
    return


@app.cell
def _(
    beta_range,
    penalty_plot_1,
    penalty_plot_2,
    penalty_plot_3,
    penalty_plot_4,
    plt,
):
    # plot 4 penalty lines at different alphas
    fig, ax = plt.subplots()

    ax.plot(beta_range, penalty_plot_4, label = "alpha=0.9")
    ax.plot(beta_range, penalty_plot_1, label = "alpha=0.5")
    ax.plot(beta_range, penalty_plot_2, label = "alpha=0.1")
    ax.plot(beta_range, penalty_plot_3, label = "alpha=0.01")

    ax.set_title("Penalty function, alpha fixed at legend values")
    ax.set_ylabel("Penalty")
    ax.set_xlabel("Beta")
    ax.legend(bbox_to_anchor=(1.01,1.02))

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create a penalty contour and surface plot
    """)
    return


@app.cell
def _(np):
    alpha_range = np.arange(start=0.01, stop=0.99, step=0.001)
    return (alpha_range,)


@app.cell
def _(alpha_range, beta_range, np):
    penalty_plot_z = np.empty(shape=(len(beta_range), len(alpha_range)))
    return (penalty_plot_z,)


@app.cell
def _(penalty_plot_z):
    penalty_plot_z.shape
    return


@app.cell
def _(
    alpha_range,
    beta_range,
    fp,
    mu,
    penalty_plot_z,
    target_alpha,
    target_power,
):
    for i, beta in enumerate(beta_range):
        for j, alpha in enumerate(alpha_range):
            penalty_plot_z[i,j] = fp.feasibility_penalty(
                mu = mu,
                power = target_power,
                alpha = target_alpha,
                beta_prime = beta,
                alpha_prime = alpha
            )
    return


@app.cell
def _(alpha_range, beta_range, np):
    X, Y = np.meshgrid(alpha_range, beta_range)
    return X, Y


@app.cell
def _(X):
    X
    return


@app.cell
def _(Y):
    Y
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Contour plot
    """)
    return


@app.cell
def _(X, Y, cm, penalty_plot_z, plt):
    # purple to green colormap
    cmap = cm.PRGn

    # initialize the figure
    fig_cont, ax_cont = plt.subplots()

    # contour colors
    cset1 = ax_cont.contourf(
        X, Y, penalty_plot_z, levels = 100,
        cmap = cmap.resampled(49)
    )

    # contour lines
    ax_cont.contour(
        X, Y, penalty_plot_z, 
        levels = 10, colors = 'k'
    )

    # set the plot characteristics
    ax_cont.set_xlim(-0.1, 1.1)
    ax_cont.set_ylim(-0.1, 1.1)
    ax_cont.set_xlabel("alpha")
    ax_cont.set_ylabel("beta")

    # add the colorbar to the plot
    fig_cont.colorbar(cset1, ax=ax_cont,
                      label="Penalty")

    plt.show()
    return (cmap,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Surface plot
    """)
    return


@app.cell
def _(X, Y, go, penalty_plot_z):
    fig_3d = go.Figure(data = go.Surface(z=penalty_plot_z, x=X, y=Y))
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
    The above surface has two issues (1) there is a long list of zeros along the line $\alpha \approx 0.01$ and there is a much larger change in the surface in response to $\alpha$ compared to $\beta$. This explains the issue that the Bayesian optimization algorithm is having. It is finding boundaries that are infeasible by optimizing for small $\alpha$ and $\beta$. A new feasibility penalty seems to be required to ensure that the optimization can proceed reasonably.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create a new surface
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### First dimension of surface
    """)
    return


@app.function
def alpha_test(alpha, alpha_prime):
    if alpha_prime > alpha: 
        return (alpha_prime - alpha)**2
    return 1


@app.cell
def _(alpha_range, np):
    alpha_surface = np.empty(len(alpha_range))
    for i2, next_alpha in enumerate(alpha_range):
        alpha_surface[i2] = alpha_test(alpha = 0.2, alpha_prime = next_alpha)
    return (alpha_surface,)


@app.cell
def _(alpha_range, alpha_surface, plt):
    plt.plot(alpha_range, alpha_surface)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Second dimension of surface
    """)
    return


@app.function
def new_surface(
        alpha_prime,
        beta_prime,
        alpha,
        beta):
    if (alpha_prime > alpha) and (beta_prime > beta):
        return 150 * ((alpha_prime - alpha)**2 + (beta_prime - beta)**2)
    return 150


@app.cell
def _(alpha_range, beta_range, np):
    new_surface1 = np.empty(shape=(len(alpha_range), len(beta_range)))
    for i3, alphas in enumerate(alpha_range):
        for j3, betas in enumerate(beta_range):
            new_surface1[i3,j3] = new_surface(
                alpha = 0.05, beta = 0.1, 
                beta_prime=betas, alpha_prime=alphas
            )
    return (new_surface1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Surface contour
    """)
    return


@app.cell
def _(X, Y, cmap, new_surface1, plt):
    # initialize the figure
    fig_cont2, ax_cont2 = plt.subplots()

    # contour colors
    cset1_2 = ax_cont2.contourf(
        X, Y, new_surface1, levels = 100,
        cmap = cmap.resampled(49)
    )

    # contour lines
    #ax_cont2.contour(
    #    X, Y, new_surface1, 
    #    levels = 10, colors = 'k'
    #)

    # set the plot characteristics
    ax_cont2.set_xlabel("alpha")
    ax_cont2.set_ylabel("beta")

    # add the colorbar to the plot
    fig_cont2.colorbar(cset1_2, ax=ax_cont2,
                      label="Penalty")

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Surface plot in 3 dimensions
    """)
    return


@app.cell
def _(X, Y, go, new_surface1):
    fig_3d_new = go.Figure(data = go.Surface(z=new_surface1, x=X, y=Y))
    fig_3d_new.update_scenes(
        xaxis_title_text="alpha",
        yaxis_title_text="beta",
        zaxis_title_text="penalty"
    )
    fig_3d_new.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
