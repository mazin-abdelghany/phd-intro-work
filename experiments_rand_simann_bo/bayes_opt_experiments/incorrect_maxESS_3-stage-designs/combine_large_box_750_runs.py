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
    import pandas as pd

    return np, pd


@app.cell
def _(pd):
    one = pd.read_csv("/workspace/experiments_rand_simann_bo/bayes_opt_experiments/large_box_bo_smooth_45x750.csv")
    two = pd.read_csv("/workspace/experiments_rand_simann_bo/bayes_opt_experiments/large_box_bo_smooth_5x750.csv")
    return one, two


@app.cell
def _():
    # correct the index for the 5x750 run
    space_needed_for_label = len(str(750))
    label_range = 10**(space_needed_for_label)

    # generate the indices using the pattern described above
    index_list = [
        i 
        for start in range(label_range, (50+1)*label_range, label_range)
        for i in range(start + 1, start + (750+1))
    ]
    return (index_list,)


@app.cell
def _(index_list, np, two):
    two["index"] = np.array(index_list[-3750:])
    return


@app.cell
def _(one, pd, two):
    pd.concat(
        [one.iloc[:, 1:], two.iloc[:, 1:]]
    ).to_csv("/workspace/experiments_rand_simann_bo/bayes_opt_experiments/large_box_bo_smooth_50x750.csv")
    return


if __name__ == "__main__":
    app.run()
