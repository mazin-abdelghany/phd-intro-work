import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(np):
    np.random.seed(438092798)
    return


@app.cell
def _(plt):
    plt.rcParams["figure.dpi"] = 150
    return


@app.cell
def _(np):
    def accept_prob(f_old, f_new, temp):
        if f_new < f_old:
            return 1
        else:
            return np.exp( (-(f_new - f_old)) / temp )

    return (accept_prob,)


@app.function
def obj_func(x):
    return x**4 + x**3 - 3*x**2 + 1


@app.cell
def _(mo):
    x_old = mo.ui.slider(-2.75, 2, 0.01, label="Old x", value=-0.45)
    return (x_old,)


@app.cell
def _(mo):
    x_new = mo.ui.slider(-2.75, 2, 0.01, label="New x: ", value=1.55)
    return (x_new,)


@app.cell
def _(mo):
    temp = mo.ui.slider(0.1, 50, 0.1, label="Temp", value=50)
    return (temp,)


@app.cell
def _(mo, temp, x_new, x_old):
    mo.hstack(
        [
            mo.vstack([x_old, mo.md(f"Has value: {x_old.value}")]),
            mo.vstack([x_new, mo.md(f"Has value: {x_new.value}")]),
            mo.vstack([temp, mo.md(f"Has value: {temp.value}")])
        ]
    )
    return


@app.cell
def _(np):
    xx = np.linspace(-2.75, 2, 500)
    yy = obj_func(xx)
    return xx, yy


@app.cell
def _(accept_prob, np, plt, temp, x_new, x_old, xx, yy):
    fig, ax = plt.subplots(figsize=(12,6))

    ax.scatter(x_old.value, obj_func(x_old.value), zorder=3, label = "old point", color = "purple", s=100)
    ax.scatter(x_new.value, obj_func(x_new.value), zorder=3, label = "new point", color = "darkorange", s=100)
    ax.text(0, 10, "Temp: " + str(temp.value))
    ax.text(0, 9, "P(accept): " + str(
        np.round(accept_prob(f_new=obj_func(x_new.value), f_old=obj_func(x_old.value), temp=temp.value),3)
    ))

    ax.plot(xx, yy, label = "objective function", color = "darkblue")
    ax.set_xlim(-2.75 , 2)
    ax.set_ylim(-6,15)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_title("Probability of accepting proposals")

    ax.legend(loc = "lower right")

    plt.gca()
    return


@app.cell
def _(np):
    temperature = np.linspace(2, 0.01, 1000)
    return (temperature,)


@app.cell
def _(np):
    def neighbor(x, alpha):
        if np.random.uniform(0,1) > 0.5:
            return x + np.random.uniform(0.001, alpha)
        else:
            return x - np.random.uniform(0.001, alpha)

    return (neighbor,)


@app.cell
def _(accept_prob, neighbor, np, temperature):
    x = []
    y = []

    x_temp = []
    y_temp = []

    accept_x = np.random.uniform(-2.5, 2)
    start_y = obj_func(accept_x)

    x_temp.append(accept_x)
    y_temp.append(start_y)

    for i, t in enumerate(temperature):

        new_x = neighbor(accept_x, t)
        eff = obj_func(new_x)

        x_temp.append(new_x)
        y_temp.append(eff)

        if accept_prob(f_new=y_temp[i+1], f_old=y_temp[i], temp=t) > np.random.uniform(0, 1):
            accept_x = new_x
            x.append(new_x)
            y.append(eff)

    print(x[-1])

    x = np.array(x)
    y = np.array(y)
    return x, y


@app.cell
def _(mo, x):
    iteration = mo.ui.slider(0, len(x)-1, label="Iteration no.:")
    return (iteration,)


@app.cell
def _(iteration, mo):
    mo.vstack([iteration, mo.md(f"Has value: {iteration.value}")])
    return


@app.cell
def _(iteration, plt, x, xx, y, yy):
    _fig, _ax = plt.subplots(figsize=(12,6))

    _ax.plot(xx, yy, color = "darkblue")
    _ax.vlines(x[iteration.value],-6,y[iteration.value], color = "purple")
    _ax.scatter(x[iteration.value], y[iteration.value], color = "purple", zorder=3, s=80)
    _ax.set_xlim(-2.75 , 2)
    _ax.set_ylim(-6,15)

    _ax.set_xlabel("$x$")
    _ax.set_ylabel("$f(x)$")
    _ax.set_title("Simulated annealing in action")

    _fig
    return


@app.cell
def _(iteration, plt, x):
    _fig, _ax = plt.subplots(figsize=(12,6), dpi=80)

    _ax.plot(x, label = "simulated annealing path")
    _ax.scatter(iteration.value, x[iteration.value], color = "purple", zorder=3, s=80)
    _ax.axhline(-1.6558, color = "darkorange", label = "global optimum")
    _ax.axhline(0.905, color = "purple", label = "local optimum")
    _ax.legend(loc = "lower right")
    _ax.set_xlabel("Iteration")
    _ax.set_ylabel("$x$ value")
    _ax.set_title("Example path of simulated annealing")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
