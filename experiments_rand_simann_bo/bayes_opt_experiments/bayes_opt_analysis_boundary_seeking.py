import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    return np, pd


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Read in data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5-stages
    """)
    return


@app.cell
def _(pd):
    scaled_trieste = pd.read_csv(filepath_or_buffer="/tf/experiments_rand_simann_bo/bayes_opt_experiments/bo_smooth_50x500_x_min_max_y_z_scaled_0.001_500haltons.csv")
    return (scaled_trieste,)


@app.cell
def _():
    upper_keys = ["upper" + str(i+1) for i in range(5)]
    lower_keys = ["lower" + str(i+1) for i in range(4)]
    lower_keys = lower_keys + ["upper5"]
    return lower_keys, upper_keys


@app.cell
def _(np):
    reversed_bounds = np.empty((25000,9))
    return (reversed_bounds,)


@app.cell
def _(fmt_bd, lower_keys, reversed_bounds, scaled_trieste, upper_keys):
    for i in range(len(scaled_trieste)):
        reversed_bounds[i] = fmt_bd.boundaries_to_reverse(
            upper_bounds=scaled_trieste[upper_keys].to_numpy()[i],
            lower_bounds=scaled_trieste[lower_keys].to_numpy()[i]
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3-stages
    """)
    return


@app.cell
def _(pd):
    three_stage = pd.read_csv(filepath_or_buffer="/tf/experiments_rand_simann_bo/bayes_opt_experiments/3-stage-designs/large_box_bo_smooth_50x500.csv")
    return (three_stage,)


@app.cell
def _(np):
    three_stage_reversed_bounds = np.empty((25000,5))
    return (three_stage_reversed_bounds,)


@app.cell
def _(lower_keys, upper_keys):
    upper_keys3 = upper_keys[0:3]
    lower_keys3 = lower_keys[0:2] + ["upper3"]
    return lower_keys3, upper_keys3


@app.cell
def _(
    fmt_bd,
    lower_keys3,
    scaled_trieste,
    three_stage,
    three_stage_reversed_bounds,
    upper_keys3,
):
    for j in range(len(scaled_trieste)):
        three_stage_reversed_bounds[j] = fmt_bd.boundaries_to_reverse(
            upper_bounds=three_stage[upper_keys3].to_numpy()[j],
            lower_bounds=three_stage[lower_keys3].to_numpy()[j]
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # What percent of bounds include boundaries?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5-stage designs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### At upper limit
    """)
    return


@app.cell
def _(np, reversed_bounds):
    np.round(sum(reversed_bounds == 4.901) / 25000 * 100, 1)
    return


@app.cell
def _(np, reversed_bounds):
    np.round(sum(reversed_bounds == 4) / 25000 * 100, 1)
    return


@app.cell
def _(np, reversed_bounds):
    np.round(sum(reversed_bounds == 160) / 25000 * 100, 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### At lower limit
    """)
    return


@app.cell
def _(np, reversed_bounds):
    np.round(sum(reversed_bounds == -1.099) / 25000 * 100, 1)
    return


@app.cell
def _(np, reversed_bounds):
    np.round(sum(reversed_bounds == 0) / 25000 * 100, 1)
    return


@app.cell
def _(np, reversed_bounds):
    np.round(sum(reversed_bounds == 20) / 25000 * 100, 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3-stage designs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### At 4
    """)
    return


@app.cell
def _(np, three_stage_reversed_bounds):
    np.round(sum(three_stage_reversed_bounds == 4) / 25000 * 100, 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### At 0
    """)
    return


@app.cell
def _(np, three_stage_reversed_bounds):
    np.round(sum(three_stage_reversed_bounds == 0) / 25000 * 100, 1)
    return


if __name__ == "__main__":
    app.run()
