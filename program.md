# autoresearch: static landslide susceptibility

You are an autonomous researcher. Goal: the best **interpretable static susceptibility
model** for the events in `config.yml`, scored by **`val_auroc`** (mean AUROC over frozen
k-means spatial folds, higher is better). The deliverable is one map:
`artifacts/maps/susceptibility_best.tif` from the best kept run.

## Files

- `prepare.py`: fixed. It holds the sampling (Moreno et al. 2025 JGR-ES, 2024 STOTEN; Corine
  mask), the evaluation and mapping, and the ledger. Read it once, never edit it.
- `config.yml`: the human's study design and limits. Never edit it.
- `train.py`: the experiment. It lists the feature rasters and builds the model.
- `tmp/`: your scratch space for one-off scripts.
- `results.tsv`: the ledger, untracked by git. Only `python prepare.py log` writes it.

## Setup

1. Agree a run tag with the human, then `git checkout -b autoresearch/<tag>`.
2. Read `README.md`, `config.yml`, `prepare.py` and `train.py`. Run `mkdir -p tmp`.
3. If `artifacts/prepared/` is missing, run `python prepare.py`, which takes seconds.
4. If `results.tsv` has no row for the current study, run the unchanged baseline first.

## How to work: small scripts, not frameworks

Do each single operation with a short script you write in `tmp/`: inspect a layer,
compute a terrain derivative, rasterize a vector, or average a climate archive. When a
script produces a feature raster:

1. Write the output to `artifacts/features/<name>/`.
2. Save the exact script beside it as `artifacts/features/<name>/recipe.sh` or `recipe.py`.
   `evaluate()` refuses features without a recipe. It copies the recipe's `saga_cmd` lines
   into `results.tsv`, so every run records exactly what it computed.
3. Add `"<name>": "artifacts/features/<name>/<file>"` to `FEATURES` in `train.py`.

A feature directory is immutable once a logged run uses it. For a variant, make a new
`<name>`. Before computing anything, check `ls artifacts/features/` and `results.tsv`. The
raster may already exist, or the idea may already have failed. Rasters may use any grid or
CRS. `evaluate()` warps them onto the DEM, using nearest-neighbour for `CATEGORICAL`.

Inputs in `data/`:

| Input | Notes |
|---|---|
| `grids/dem.tif` | Reference grid, 250 m, EPSG:32632 |
| `vectors/CorineLandCover_polygon.tif` | CLC level-3 codes, 30 m; the file lacks its CRS, which is EPSG:25832 |
| `vectors/roads.gpkg`, `buildings.gpkg`, `electrical_infrastructure.shp` | Lines, building centroids, lines |
| `dynamic/{precipitation,temperature}/*.nc` | Daily 1980–2023, 250 m, own extent (variable `DATE`) |
| `dynamic/lightning/*.nc` | Daily flash density 2012–2024, DEM extent |

**Dynamic data.** The map is static, so a NetCDF archive enters as a map: a climatology, an
extreme or threshold index, seasonality or variability. Points have no dates, and any feature
built from event dates leaks the inventory. Measured costs on this machine:

- **Streaming.** One monthly precipitation or temperature file decompresses to about 50 MB
  in about 0.2 s. A full 1980–2023 pass with rolling windows takes about 2 minutes, at about
  1 GB of memory.
- **Whole series.** Keeping the whole daily series of every DEM-extent cell, for example for
  per-pixel percentiles, takes about 16 GB and 4 minutes. That is one variable at a time and
  float32 only; two variables at once exceed the memory limit.
- **Layout.** Each day is stored as one compressed full-grid chunk. Reading a spatial tile
  still decompresses whole days, so processing tile by tile re-reads the archive per tile.

Stream month by month and carry rolling-window state across files. Estimate the cost first
(files × 0.2 s per pass). Save the result once as a feature, so later experiments reuse it.

The model must stay glass-box. The baseline is an Explainable Boosting Machine; additive
terms with a few pairwise interactions are fine. The human must be able to read every
shape function behind the final map.

## SAGA GIS

Put SAGA calls in a bash recipe, with literal paths and one `saga_cmd` per line. Start from
this template:

```bash
#!/usr/bin/env bash
# artifacts/features/slope/recipe.sh: run from the repository root
set -euo pipefail
ulimit -f 2000000                           # no file above ~2 GB (limits.file_gb)
ulimit -d 20000000                          # no process above ~20 GB of data (limits.memory_gb)
unset GDAL_DRIVER_PATH GDAL_DATA PROJ_DATA  # system SAGA must not load conda's GDAL plugins
mkdir -p artifacts/features/slope
timeout 30m saga_cmd --cores=4 ta_morphometry 0 -ELEVATION data/grids/dem.tif -SLOPE artifacts/features/slope/slope.sdat -UNIT_SLOPE 1
```

Vector distances: rasterize onto the DEM grid, then compute proximity:

```bash
timeout 30m saga_cmd --cores=4 grid_gridding 0 -INPUT data/vectors/roads.gpkg -OUTPUT 0 -TARGET_DEFINITION 1 -TARGET_TEMPLATE data/grids/dem.tif -GRID tmp/roads.sdat
timeout 30m saga_cmd --cores=4 grid_tools 26 -FEATURES tmp/roads.sdat -DISTANCE artifacts/features/dist_roads/dist.sdat
```

Find tools yourself: `saga_cmd ta_morphometry` lists a library's tools, and
`saga_cmd ta_morphometry 14` prints one tool's usage. Useful libraries:

- `ta_morphometry`: 0 slope/aspect/curvatures, 1 convergence, 14 standardized height, 16 TRI, 18 TPI.
- `ta_hydrology`: 0 flow accumulation, 15 SAGA wetness, 19 specific catchment area, 20 TWI, 22 LS.
- Others: `ta_channels`, `ta_lighting`, `grid_filter`, `statistics_grid` (1 focal statistics),
  `grid_gridding` and `grid_tools`.

## Rules

- **No leakage.** `train.py` and recipes never read `artifacts/prepared/samples.csv`, which
  holds labels and folds. No feature may derive from the event inventory, such as event
  density or distance to events. Coordinates are not predictors.
- **Reporting bias.** The inventory records events near infrastructure. Distance to roads or
  buildings can raise `val_auroc` by learning where landslides get reported, not where they
  happen. Prefer physical predictors. If you test one, say so in the description.
- **Comparable runs.** Never change the sample, folds or metric. Never edit `prepare.py` or
  `config.yml`.

## Security and resources

These limits are hard. A violation ends the run.

- **Write only** to `train.py`, `run.log`, `tmp/` and `artifacts/features/`. Write
  `artifacts/` and `results.tsv` otherwise only through `prepare.py`. Never write, move or
  delete anything in `data/`, `legacy/`, `artifacts/legacy/` or `.git/` internals.
- **Disk.** `artifacts/` plus `tmp/` must stay below `limits.artifacts_gb`, which is 40 GB;
  `evaluate()` refuses to run above it. No file may exceed 2 GB: keep `ulimit -f` in every
  recipe. Check `python prepare.py usage` before large jobs such as full-archive NetCDF
  reductions or fine grids. Above 80 %, run `python prepare.py prune`, which deletes feature
  directories no kept run uses, and empty `tmp/`. Use `rm` only inside `tmp/` and
  `artifacts/features/`.
- **Time.** Long computations are normal, such as NetCDF statistics or big EBMs, but nothing may
  run longer than `limits.timeout_min`, which is 30 minutes. Wrap `saga_cmd`, recipe scripts
  and training runs in `timeout 30m`. Any Python process that imports `prepare` is also killed
  after 30 minutes. If your harness caps a command's duration (Claude Code: 10 minutes in the
  foreground), start long jobs in the background and wait for them to finish. A command your
  harness cut short is not a crash of the idea. Use at most `limits.cpus` cores, which is 4.
- **Memory.** No process may allocate more than `limits.memory_gb`, which is 20 GB of a
  31 GB machine. Keep `ulimit -d` in every recipe. Python processes that import `prepare` get
  `MemoryError` beyond it. A recipe that hits the limit must stream, not ask for more.
- **Nothing outside the box.** No network access, no package installs, no `sudo`, no
  background processes, and one experiment at a time.

## The experiment loop

LOOP FOREVER:

1. Read `results.tsv` (descriptions, features and SAGA commands) and
   `artifacts/last_explanation.md`, the TalkToEBM text of every shape function. Pick ONE
   untried idea that these support. An identical experiment is refused automatically.
2. Make any new feature with a `tmp/` script, and save its recipe. Edit `train.py`, then
   `git commit -am "<idea>"`.
3. Run `timeout 30m python train.py > run.log 2>&1`, in the background if it may be long.
4. Run `grep "^val_auroc:\|^best_kept" run.log`. If it prints nothing, read `tail -n 50 run.log`.
   Fix trivial mistakes and rerun. Otherwise log a crash.
5. Log with `python prepare.py log <keep|discard|crash> "<what changed; result; why; next>"`.
   Keep only if `val_auroc` beats `best_kept_val_auroc` by at least 0.001. The first baseline
   is a keep. Keep also saves the map.
6. On keep, the branch advances. On discard or crash, run `git reset --hard HEAD~1`. Leave the
   feature directory in place: the ledger points to it.

**Token economy.** Keep the context small. Never print rasters, NetCDF files, whole logs
or `samples.csv`. Use `grep`, `head`, `tail` and short summaries in your `tmp/` scripts.

**Simplicity.** A tiny gain that adds ugly complexity is not worth it. Removing a feature for
equal AUROC is a win. Spatial folds are noisy, so check `val_auroc_sd` and the per-fold scores
in `artifacts/last_run.json`.

**NEVER STOP.** Once the loop has begun, do not ask the human whether to continue. They may
be asleep. If you run out of ideas, revisit the papers' predictors, the importances and near
misses, or try something radical. The loop runs until the human interrupts you.
