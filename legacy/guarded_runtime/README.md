# autosearch_lab

A small mass-movement modelling workflow for **landslides, debris flows and rockfalls**,
one process per study. Start with a DEM, dated event points and optional daily archives
and raw layers. The LLM proposes structured experiments; fixed code builds features,
trains an Explainable Boosting Machine, and evaluates frozen spatial folds.

| File | Role |
|---|---|
| `config.yml` | Your inputs, sampling design, approved layers and resource limits |
| `experiment.yml` | Feature choices and bounded EBM settings; the experiment surface |
| `prepare.py` | Reproducible non-events, date coverage and evaluation splits |
| `features.py` | Validated SAGA operations and small batches of dynamic histories |
| `train.py` | Train, compare experiments, request proposals and evaluate holdout |
| `guard.py` | Shared disk accounting, per-file limit, CPU/memory/time supervision |
| `program.md`, `loop.sh` | Proposal instructions and optional bounded LLM loop |

## Start

Use Python 3.10+, install `requirements.txt`, and install SAGA GIS (`saga_cmd`, tested
with 9.12.6). Set your paths in `config.yml`, then:

```bash
python prepare.py             # freeze observations and splits
python train.py               # run the small baseline in experiment.yml
python train.py --list        # completed experiments and development scores
python guard.py usage         # generated-data usage, including old caches
```

Edit `experiment.yml` and rerun `train.py` to test an idea. Duplicate configurations
reuse the completed run. No LLM is needed for this workflow. If Codex CLI is installed
and logged in, `bash loop.sh 5` requests at most five additional structured proposals
using read-only sessions. Responses are validated as data; they are never executed as code.
The runner snapshots each proposal. It does not give the LLM an open-ended build session.

At the end, `python train.py --evaluate` evaluates the best development model on the
reserved holdout once and closes that study to further search. The ordinary training
command never scores the holdout. Use `--config /path/config.yml` and
`--experiment /path/experiment.yml` for another study; outputs live beside its config.

## Inputs and features

- DEM: single-band GeoTIFF, projected CRS in metres, north-up; it defines the grid.
- Events: georeferenced Points with an accurate `date` column, one movement type only.
  Undated records still exclude nearby non-events. Set the surveyed period explicitly
  when known; otherwise the dates of recorded events define the period.
- Dynamic sources: daily NetCDF `(time, y, x)`, regular x/y coordinates, declared CRS,
  one timestamp per day and no overlapping files. An empty `sources: {}` is allowed.
- Optional surveyed/terrain masks: raster value 1 means eligible. Existing rasters and
  vectors used as predictors must be registered under `config.yml: predictors`.

Terrain choices are slope, aspect, northness, eastness, plan and profile curvature
(SAGA `ta_morphometry 0`). Approved vector layers can supply distance grids (SAGA
`grid_tools 26`). Outputs must match the DEM. Dynamic recipes offer sum, mean, minimum,
maximum, standard deviation and threshold counts over chosen antecedent windows.
The supplied experiment demonstrates the schema. Additional SAGA operations require
an explicit addition to the trusted builder, not an arbitrary command from the LLM.

## Non-events and scientific limits

The sampler is a point-based adaptation of [Moreno et al. (2025)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024JF008219)
and [Steger et al. (2026)](https://nhess.copernicus.org/articles/26/3637/2026/):

1. Both classes share the valid DEM, surveyed/terrain masks and land-cover exclusions.
2. Non-events occupy unique cells at least 150 m from every inventoried event;
   occupied event cells are excluded even if their centres exceed that distance.
3. Control dates are independent of event dates. Year-month quotas are proportional
   to eligible calendar days, before coverage and optional wet-day filtering.
4. Every observation needs complete histories for all configured dynamic sources.
   Missing days are not zero rainfall. Fixed spatial blocks define CV and holdout.

The default includes all recorded events. `wet_days` optionally filters both classes;
the commented three-day sum is a daily-data approximation, not Moreno's hourly
72-hour criterion. Steger's daily 10 mm criterion is also shown in the config.
These choices change the target population and final class/month balance. Day 1 is
the previous day; using observation-day weather requires `same_day: true` and can
include post-event information. That is a retrospective design, not an early forecast.

NHESS's basin selection, 10 km spacing, within-basin 30-day thinning, GPP runout and
areal predictor aggregation are **not reproduced** by the point sampler. Surveyed
areas and process masks are supplied inputs, not inferred from the event density.
Without them, the run is exploratory. Do not exclude steep or bare-rock terrain
indiscriminately for rockfall, or flat runout terrain for impact inventories.

Generated controls are presumed non-events, not verified absences. Repeated search
can overfit CV; spatial evaluation does not establish future-year or storm transfer.
Scores and probabilities are conditional on the chosen sample, not calibrated
regional event rates. `sampling_report.json` records filtering losses and date counts.

## Resource limits and outputs

One shared **40 decimal GB generated-data budget** covers all of `artifacts/`, including
scratch, logs, models and previous caches. Raw inputs are excluded. By default each
job reserves 4 GB, each file is capped at 1 GB, and jobs use at most two CPU cores,
with a 6 GB memory watchdog and 900-second timeout. Limits are in `config.yml`.
The runner rejects oversized grids, samples, histories and invalid experiment schemas.
Failed job scratch is removed; completed runs are preserved and never silently evicted.

The total disk and process-tree memory checks are watchdogs (every 0.2 seconds and on
exit), not filesystem/cgroup quotas; they can overshoot between checks. A strict disk
ceiling needs an administrator-provided filesystem quota. The supported experiment
interface executes only installed builders and validated input selections; bypassing
the runner with arbitrary code is outside these protections.

Outputs are under `artifacts/<process>/<study id>/`: `prepared/`, one directory per
completed experiment in `runs/`, optional `proposals/`, and final `holdout/`. Each run
includes the exact experiment, metrics, feature matrix, EBM model, term importances,
development predictions and provenance. Changing the sampling design or preparation
code starts a different study; scores from different studies are not interchangeable.

The original scripts remain in `legacy/` as historical reference; do not execute the
old unsafe loop. Data provenance/licensing notes are in `legacy/DATA.md`. Existing
data and results are preserved. Run the synthetic regression checks with:

```bash
python -m unittest discover -s tests -v
```
