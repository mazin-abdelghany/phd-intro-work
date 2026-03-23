import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import gpflow

    return (gpflow,)


@app.cell
def _():
    import numpy as np
    import tensorflow as tf

    np.random.seed(1234)
    tf.random.set_seed(1234)
    return np, tf


@app.cell
def _():
    import trieste

    return (trieste,)


@app.cell
def _(np, tf, trieste):
    def masked_branin(x):
        mask_nan = np.sqrt((x[:, 0] - 0.5) ** 2 + (x[:, 1] - 0.4) ** 2) < 0.3
        y = np.array(trieste.objectives.Branin.objective(x))
        y[mask_nan] = np.nan
        return tf.convert_to_tensor(y.reshape(-1, 1), x.dtype)

    return (masked_branin,)


@app.cell
def _():
    from trieste.space import Box

    search_space = Box([0, 0], [1, 1])
    return (search_space,)


@app.cell
def _(masked_branin, search_space):
    from trieste.experimental.plotting import plot_function_plotly

    fig = plot_function_plotly(
        masked_branin, search_space.lower, search_space.upper
    )
    fig.show()
    return (fig,)


@app.cell
def _(masked_branin, np, tf, trieste):
    OBJECTIVE = "OBJECTIVE"
    FAILURE = "FAILURE"


    def observer(x):
        y = masked_branin(x)
        mask = np.isfinite(y).reshape(-1)
        return {
            OBJECTIVE: trieste.data.Dataset(x[mask], y[mask]),
            FAILURE: trieste.data.Dataset(x, tf.cast(np.isfinite(y), tf.float64)),
        }

    return FAILURE, OBJECTIVE, observer


@app.cell
def _(np, observer):
    observer(x=np.array([[0.5,0.5]]))
    return


@app.cell
def _(initial_data):
    initial_data["FAILURE"]
    return


@app.cell
def _(initial_data):
    initial_data["OBJECTIVE"]
    return


@app.cell
def _(observer, search_space):
    num_init_points = 15
    initial_data = observer(search_space.sample(num_init_points))
    return initial_data, num_init_points


@app.cell
def _(FAILURE, OBJECTIVE, initial_data, search_space):
    from trieste.models.gpflow import build_gpr, build_vgp_classifier

    regression_model = build_gpr(
        initial_data[OBJECTIVE], search_space, likelihood_variance=1e-7
    )
    classification_model = build_vgp_classifier(
        initial_data[FAILURE], search_space, noise_free=True
    )
    return classification_model, regression_model


@app.cell
def _(gpflow, regression_model):
    gpflow.utilities.print_summary(regression_model)
    return


@app.cell
def _(classification_model, gpflow):
    gpflow.utilities.print_summary(classification_model)
    return


@app.cell
def _(FAILURE, OBJECTIVE, classification_model, regression_model):
    from gpflow.keras import tf_keras

    from trieste.models import TrainableProbabilisticModel
    from trieste.models.gpflow.models import (
        GaussianProcessRegression,
        VariationalGaussianProcess,
    )
    from trieste.models.optimizer import BatchOptimizer
    from trieste.types import Tag


    models: dict[Tag, TrainableProbabilisticModel] = {
        OBJECTIVE: GaussianProcessRegression(regression_model),
        FAILURE: VariationalGaussianProcess(
            classification_model,
            BatchOptimizer(tf_keras.optimizers.Adam(1e-3)),
            use_natgrads=True,
        ),
    }
    return (models,)


@app.cell
def _(models: "dict[Tag, TrainableProbabilisticModel]"):
    models
    return


@app.cell
def _(FAILURE, OBJECTIVE, tf):
    from trieste.acquisition.rule import EfficientGlobalOptimization
    from trieste.acquisition import (
        SingleModelAcquisitionBuilder,
        ExpectedImprovement,
        Product,
    )


    class ProbabilityOfValidity(SingleModelAcquisitionBuilder):
        def prepare_acquisition_function(self, model, dataset=None):
            def acquisition(at):
                mean, _ = model.predict_y(tf.squeeze(at, -2))
                return mean

            return acquisition


    ei = ExpectedImprovement()
    pov = ProbabilityOfValidity()
    acq_fn = Product(ei.using(OBJECTIVE), pov.using(FAILURE))
    rule = EfficientGlobalOptimization(acq_fn)  # type: ignore
    return (rule,)


@app.cell
def _(
    OBJECTIVE,
    initial_data,
    models: "dict[Tag, TrainableProbabilisticModel]",
    observer,
    rule,
    search_space,
    tf,
    trieste,
):
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

    num_steps = 20
    result = bo.optimize(
        num_steps, initial_data, models, rule
    ).final_result.unwrap()

    arg_min_idx = tf.squeeze(
        tf.argmin(result.datasets[OBJECTIVE].observations, axis=0)
    )
    print(f"query point: {result.datasets[OBJECTIVE].query_points[arg_min_idx, :]}")
    return (result,)


@app.cell
def _(result):
    result.datasets["FAILURE"]
    return


@app.cell
def _(FAILURE, masked_branin, num_init_points, result, search_space):
    import matplotlib.pyplot as plt
    from trieste.experimental.plotting import (
        plot_gp_2d,
        plot_function_2d,
        plot_bo_points,
    )

    mask_fail = (
        result.datasets[FAILURE].observations.numpy().flatten().astype(int) == 0
    )
    _fig, _ax = plot_function_2d(
        masked_branin,
        search_space.lower,
        search_space.upper,
        grid_density=20,
        contour=True,
    )
    plot_bo_points(
        result.datasets[FAILURE].query_points.numpy(),
        ax=_ax[0, 0],
        num_init=num_init_points,
        mask_fail=mask_fail,
    )
    plt.show()
    return mask_fail, plot_bo_points, plot_gp_2d, plt


@app.cell
def _(OBJECTIVE, fig, num_init_points, result, search_space, tf):
    from trieste.experimental.plotting import (
        plot_model_predictions_plotly,
        add_bo_points_plotly,
    )

    _arg_min_idx = tf.squeeze(
        tf.argmin(result.datasets[OBJECTIVE].observations, axis=0)
    )

    _fig = plot_model_predictions_plotly(
        result.models[OBJECTIVE],
        search_space.lower,
        search_space.upper,
    )
    _fig = add_bo_points_plotly(
        x=result.datasets[OBJECTIVE].query_points[:, 0].numpy(),
        y=result.datasets[OBJECTIVE].query_points[:, 1].numpy(),
        z=result.datasets[OBJECTIVE].observations.numpy().flatten(),
        num_init=num_init_points,
        idx_best=_arg_min_idx,
        fig=fig,
        figrow=1,
        figcol=1,
    )

    fig.show()
    return


@app.cell
def _(
    FAILURE,
    mask_fail,
    num_init_points,
    plot_bo_points,
    plot_gp_2d,
    plt,
    result,
    search_space,
):
    _fig, _ax = plot_gp_2d(
        result.models[FAILURE].model,
        search_space.lower,
        search_space.upper,
        grid_density=20,
        contour=True,
        figsize=(12, 5),
        predict_y=True,
    )

    plot_bo_points(
        result.datasets[FAILURE].query_points.numpy(),
        num_init=num_init_points,
        ax=_ax[0, 0],
        mask_fail=mask_fail,
    )

    plot_bo_points(
        result.datasets[FAILURE].query_points.numpy(),
        num_init=num_init_points,
        ax=_ax[0, 1],
        mask_fail=mask_fail,
    )

    plt.show()
    return


if __name__ == "__main__":
    app.run()
