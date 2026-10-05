#!/usr/bin/env python3
"""Agent-editable experiment: feature engineering + model. Read program.md first."""


def build_features(ctx, points):
    """Return one numeric row per point, in the original order; never read labels here."""
    import pandas as pd

    slope = ctx.output("slope.sdat")
    ctx.run([
        "saga_cmd", f"--cores={ctx.cpus}", "ta_morphometry", "0",
        "-ELEVATION", ctx.dem, "-SLOPE", slope, "-UNIT_SLOPE", "1", "-METHOD", "6",
    ], inputs=[ctx.dem], outputs=[slope])
    features = {"elevation": ctx.raster(ctx.dem), "slope": ctx.raster(slope)}
    # Arbitrary antecedent features belong here, without materialising raster cubes.
    if "precipitation" in ctx.cfg["dynamic"]["sources"]:
        days = min(7, ctx.cfg["dynamic"]["max_history_days"])
        rain = ctx.history("precipitation", days)
        features[f"rain_{days}d"] = rain[:, 1:days + 1].sum(axis=1)
    return pd.DataFrame(features, index=points.index)


def make_model(*, seed, cpus):
    """Any sklearn-compatible binary classifier with fit/predict_proba/classes_."""
    from interpret.glassbox import ExplainableBoostingClassifier
    return ExplainableBoostingClassifier(
        interactions=0, max_rounds=500, learning_rate=0.03,
        outer_bags=4, validation_size=0.15, early_stopping_rounds=30,
        random_state=seed, n_jobs=cpus,
    )


if __name__ == "__main__":
    # Keep this entry point: the fixed runner supervises a snapshot in artifacts/.
    from prepare import main
    main(default_run=True)
