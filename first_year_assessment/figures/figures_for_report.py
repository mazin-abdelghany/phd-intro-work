import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Loss function figures
    """)
    return


@app.cell
def _(np):
    alpha_range = np.linspace(0, 1, num=200)
    beta_range = np.linspace(0, 1, num=200)
    return alpha_range, beta_range


@app.cell
def _():
    target_alpha = 0.05
    target_beta = 0.1
    return (target_alpha,)


@app.cell
def _(alpha_range, target_alpha):
    loss_alpha = (alpha_range - target_alpha)**2
    return (loss_alpha,)


@app.cell
def _(alpha_range, loss_alpha, plt):
    fig, ax = plt.subplots(figsize=(9,2.5))

    ax.plot(alpha_range, loss_alpha, color = "darkorange", label="loss function")
    ax.set_title("Loss function with $\\alpha'$")
    ax.set_ylabel("Loss($\\alpha'$)")
    ax.set_xlabel("$\\alpha'$")
    ax.axvline(0.05, color = "purple", linewidth = 1, linestyle="--", label = "minimum")
    ax.legend()

    fig.savefig("/tf/first_year_assessment/figures/loss_alpha.png", dpi=300, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(alpha_range, beta_range, np):
    X, Y = np.meshgrid(alpha_range, beta_range)

    loss_ab = (X - 0.05)**2 + (Y-0.1)**2
    return X, Y, loss_ab


@app.cell
def _(X, Y, loss_ab, plt):
    _fig, _ax = plt.subplots(figsize=(10.3,2.5))

    # number of contour lines
    levels = 15

    # filled contours
    cf = _ax.contourf(X, Y, loss_ab, levels=levels, cmap="viridis")

    # contour lines
    cs = _ax.contour(X, Y, loss_ab, levels=levels, colors="black", linewidths=1.2)

    # colorbar
    _fig.colorbar(cf, ax=_ax, label="Loss($\\alpha',\\beta'$)")

    # minimum point
    _ax.plot(0.05, 0.1, marker="o", color="white", markersize=8)
    _ax.text(0.065, 0.081, " min", color="white")

    # labels
    _ax.set_title("Loss function with $\\alpha'$ and $\\beta'$")
    _ax.set_xlabel("$\\alpha'$")
    _ax.set_ylabel("$\\beta'$")

    _fig.savefig("/tf/first_year_assessment/figures/loss_alpha_beta.png", dpi=300, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reverse parameterisation exploration
    """)
    return


@app.cell
def _(np):
    def reverse_to_boundaries(params, K):
        params = np.asarray(params).flatten()
        c = params[0]

        delta_u = params[1::2][::-1]
        delta_l = params[2::2][::-1]

        upper_bounds = np.array([c + np.sum(delta_u[k:]) for k in range(K)])
        lower_bounds = np.array([c - np.sum(delta_l[k:]) for k in range(K)])

        return upper_bounds, lower_bounds

    return (reverse_to_boundaries,)


@app.cell
def _():
    #         c  u_4  l_4  u_3  l_3  u_2  l_2  u_1  l_1
    params = [3, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1]

    delta_u = params[1::2][::-1]
    delta_l = params[2::2][::-1]

    for k in range(5):
        print("k=",k+1)
        print("upper:")
        print(delta_u[k:])
        print(sum(delta_u[k:]))
        print(3+sum(delta_u[k:]))
        print("lower:")
        print(delta_l[k:])
        print("\n")
    return


@app.cell
def _(reverse_to_boundaries):
    reverse_to_boundaries(params=[3, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1], K=5)
    return


if __name__ == "__main__":
    app.run()
