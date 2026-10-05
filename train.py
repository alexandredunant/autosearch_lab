#!/usr/bin/env python3
"""Static landslide susceptibility experiment. The agent edits this file (see program.md).

Feature rasters are made beforehand by one-off scripts in tmp/ and saved, with that script
as recipe.sh/recipe.py, in artifacts/features/<name>/. Here you only choose them and the model.
"""
from interpret.glassbox import ExplainableBoostingClassifier

from prepare import CPUS, SEED, evaluate

FEATURES = {  # name -> raster (any grid; warped onto the DEM)
    "elevation": "data/grids/dem.tif",
    "landcover": "data/vectors/CorineLandCover_polygon.tif",
}
CATEGORICAL = ("landcover",)


def make_model() -> ExplainableBoostingClassifier:
    """A fresh, unfitted glass-box model per call."""
    return ExplainableBoostingClassifier(
        interactions=0, feature_types=["nominal" if n in CATEGORICAL else "continuous" for n in FEATURES],
        random_state=SEED, n_jobs=CPUS)


if __name__ == "__main__":
    evaluate(FEATURES, make_model, CATEGORICAL)
