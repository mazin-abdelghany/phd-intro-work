import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    # pip install /tf/pyGroupSequentialDesigns/
    return


@app.cell
def _():
    from py_group_sequential_designs import (
        boundaries, feasibility_penalty, format_boundaries_after_ask,
        function_to_minimize, generate_gpr_input, simulate, sample_size
    )
    return (
        boundaries,
        feasibility_penalty,
        format_boundaries_after_ask,
        function_to_minimize,
        generate_gpr_input,
        sample_size,
        simulate,
    )


@app.cell
def _(boundaries):
    boundaries.calculate_of_boundaries()
    return


@app.cell
def _(boundaries):
    boundaries.calculate_pocock_boundaries()
    return


@app.cell
def _(boundaries):
    boundaries.calculate_triangular_boundaries()
    return


@app.cell
def _(format_boundaries_after_ask):
    format_boundaries_after_ask.format_boundaries_after_ask(n_analyses=3, result = [[1,2,3,4,5,6]])
    return


@app.cell
def _(feasibility_penalty):
    feasibility_penalty.feasibility_penalty()
    return


@app.cell
def _(sample_size):
    sample_size.find_sample_size()
    return


@app.cell
def _(sample_size):
    sample_size.max_ess()
    return


@app.cell
def _(sample_size):
    sample_size.sample_size_means()
    return


@app.cell
def _(function_to_minimize):
    function_to_minimize.function_to_minimize(2, 3)
    return


@app.cell
def _(generate_gpr_input):
    generate_gpr_input.generate_gpr_input(3, [1,2,3], [4,5,6], 20)
    return


@app.cell
def _(simulate):
    simulate.group_sequential_designs()
    return


if __name__ == "__main__":
    app.run()
