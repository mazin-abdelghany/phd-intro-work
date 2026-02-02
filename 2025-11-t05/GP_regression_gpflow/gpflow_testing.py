import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import gpflow
    return gpflow, np, plt


@app.cell
def _(np, plt):
    X = np.array(
        [
            [0.865], [0.666], [0.804], [0.771], [0.147], [0.866], [0.007], [0.026],
            [0.171], [0.889], [0.243], [0.028],
        ]
    )
    Y = np.array(
        [
            [1.57], [3.48], [3.12], [3.91], [3.07], [1.35], [3.80], [3.82], [3.49],
            [1.30], [4.00], [3.82],
        ]
    )

    plt.plot(X, Y, "kx", mew=2)
    return X, Y


@app.cell
def _(X):
    X
    return


@app.cell
def _(Y):
    Y
    return


@app.cell
def _(X, Y, gpflow):
    model = gpflow.models.GPR(
        (X, Y),
        kernel=gpflow.kernels.SquaredExponential(),
    )
    return (model,)


@app.cell
def _(gpflow, model):
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)
    return


@app.cell
def _(model, np):
    Xnew = np.array([[0.5]])
    model.predict_f(Xnew)
    return


@app.cell
def _(np):
    Xplot = np.linspace(-0.1, 1.1, 100)[:, None]
    return (Xplot,)


@app.cell
def _(Xplot, model):
    f_mean, f_var = model.predict_f(Xplot, full_cov=False)
    y_mean, y_var = model.predict_y(Xplot)
    return f_mean, f_var, y_mean, y_var


@app.cell
def _(f_mean, f_var, np, y_mean, y_var):
    f_lower = f_mean - 1.96 * np.sqrt(f_var)
    f_upper = f_mean + 1.96 * np.sqrt(f_var)
    y_lower = y_mean - 1.96 * np.sqrt(y_var)
    y_upper = y_mean + 1.96 * np.sqrt(y_var)
    return f_lower, f_upper, y_lower, y_upper


@app.cell
def _(X, Xplot, Y, f_lower, f_mean, f_upper, plt, y_lower, y_upper):
    plt.plot(X, Y, "kx", mew=2, label="input data")
    plt.plot(Xplot, f_mean, "-", color="C0", label="mean")
    plt.plot(Xplot, f_lower, "--", color="C0", label="f 95% confidence")
    plt.plot(Xplot, f_upper, "--", color="C0")
    plt.fill_between(
        Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color="C0", alpha=0.1
    )
    plt.plot(Xplot, y_lower, ".", color="C0", label="Y 95% confidence")
    plt.plot(Xplot, y_upper, ".", color="C0")
    plt.fill_between(
        Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="C0", alpha=0.1
    )
    plt.legend()
    return


if __name__ == "__main__":
    app.run()
