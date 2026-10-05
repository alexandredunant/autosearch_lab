# Structured mass-movement experiment proposal

Return exactly one JSON object describing the next experiment. Do not run commands,
use tools, edit files, inspect raw observations or read holdout results. The runner
validates the proposal and executes the SAGA commands and EBM training.

Test one scientifically motivated change against the best development result. The
movement type, sampling design, date coverage, folds and resource limits are fixed.
Avoid previously tested feature/model configurations. Explain the hypothesis in description.

Allowed structure:

```json
{
  "description": "Test whether curvature adds information to slope and rain.",
  "terrain": ["slope", "plan_curvature"],
  "rasters": [],
  "vectors": [],
  "dynamic": [{"variable": "precipitation", "days": 7, "lag": 1, "stat": "sum"}],
  "model": {"interactions": 0, "max_rounds": 500, "learning_rate": 0.03, "max_leaves": 3, "min_samples_leaf": 10}
}
```

Terrain choices: slope (degrees), aspect (radians), northness, eastness,
plan_curvature, profile_curvature. These request SAGA ta_morphometry tool 0 outputs
on the reference DEM. Elevation is always included. Empty lists are allowed.

Dynamic recipes use variables from the supplied context. days must be a positive
integer; lag is an integer >= 1 unless same_day_allowed explicitly permits 0.
days + lag - 1 must not exceed max_history_days. A lag of 1 ends the window on the
previous calendar day. Statistics: sum, mean, min, max, std, count_above (also needs
a finite threshold). At most 32 recipes; missing days are never treated as zero.

Only exact approved_layers from the supplied context can be used. A raster entry
looks like {"name": "geology", "path": "data/geology.tif", "categorical": true}.
A vector distance entry looks like {"name": "distance_to_roads", "path":
"data/vectors/roads.gpkg"}; the runner rasterizes it and executes SAGA grid_tools 26.
At most 32 rasters and 8 vector layers. Do not invent paths or change categorical
treatment. No inventory-derived predictors, coordinates, dates, labels or fold IDs.

Model bounds: interactions 0–10; max_rounds 1–3000; learning_rate 0.0001–1;
max_leaves 2–16; min_samples_leaf 2–1000. All except learning_rate are integers.
The runner fixes the model family, seed, thread count and evaluation protocol.
Selection uses mean spatial-fold average precision; ROC AUC is secondary.
Probabilities describe the sampled population, not regional event rates.

No Python, shell commands, executable recipes, arbitrary SAGA tools, output paths,
grid expansion or resolution changes are accepted. Return JSON without Markdown fences.
