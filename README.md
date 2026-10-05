# autosearch_lab

An [autoresearch](https://github.com/karpathy/autoresearch)-style loop for **static
landslide susceptibility**: a coding agent iterates on features and an Explainable
Boosting Machine, and the human gets one interpretable map at the end.

| File | Who edits | Role |
|---|---|---|
| `config.yml` | human | Inputs, Corine mask classes, sampling design, disk/time limits |
| `prepare.py` | nobody | Sampling, fixed spatial-CV metric, map, TalkToEBM report, ledger |
| `train.py` | agent | Feature rasters to use, EBM settings |
| `program.md` | human | Agent instructions: SAGA usage, rules, security limits, loop |
| `install.sh` | — | `.venv` (uv), SAGA check, sample preparation |

## Quick start

```bash
bash install.sh                 # Python env, SAGA check, builds artifacts/prepared/
source .venv/bin/activate
python train.py                 # baseline: elevation + Corine land cover
```

Then start the agent in this directory, e.g. with Claude Code on a mid-size model:

```bash
claude --model sonnet "Read program.md and start a new run."
```

Sonnet is the recommended trade-off: each iteration is short (write a small script, run,
read a bounded report) and does not need the largest model, while SAGA and raster code
still benefit from a capable one. Haiku is cheaper but makes more tool and leakage
mistakes. Approve `python`, `bash tmp/...`, `saga_cmd` and `git` commands with "always
allow" during the first iterations so the loop can then run unattended.

## Design

**Sampling** follows the spatial protocol of Moreno et al.
([2025, JGR-ES](https://doi.org/10.1029/2024JF008219);
[2024, STOTEN](https://doi.org/10.1016/j.scitotenv.2023.169166)), adapted to a static model:

1. The domain is the valid DEM minus trivial terrain from Corine Land Cover: bare rock,
   glaciers and water. A flat-slope rule and a surveyed-area mask are optional extras.
2. Presences are inventoried events in the domain, one per 250 m cell.
3. Absences are random domain cells at least 150 m from any event, 10 per presence.
4. Ten k-means spatial folds (as `sperrorest::partition_kmeans`) give `val_auroc`.

The papers' space-time parts (random dates, antecedent rainfall windows, wet-day filter)
are deliberately not used: the target is one static map. Daily archives remain available
as climatology predictors.

**Features** are rasters. The agent writes a one-off script in `tmp/`, saves the output in
`artifacts/features/<name>/` together with the script as `recipe.sh`/`recipe.py`, and
lists it in `train.py`. `evaluate()` refuses features without a recipe and copies every
`saga_cmd` line into `results.tsv`, so each experiment's exact SAGA commands are listed
and an identical experiment (same `train.py` and recipes) is refused.

**Model** is compulsorily an `interpret` `ExplainableBoostingClassifier`. Every run writes
`artifacts/last_explanation.md` with [TalkToEBM](https://interpret.ml/TalkToEBM/): term
importances and each shape function as text, which the agent reads to choose the next
experiment. This needs no API key; set `T2EBM_LLM` (e.g. `gpt-4o`) and `OPENAI_API_KEY`
to append TalkToEBM's own LLM summary.

## Outputs

```
results.tsv                       ledger: commit, val_auroc, status, description, features, SAGA commands
artifacts/prepared/               frozen sample, folds, domain.tif, report.json
artifacts/features/<name>/        feature rasters + recipe
artifacts/maps/susceptibility_best.tif   the final map (latest keep)
artifacts/maps/explanation_best.md       its TalkToEBM report; ebm_best.json the model
```

Map values are relative susceptibility conditional on the sampling design (presences vs
presumed absences in the Corine domain), not calibrated event probabilities.

## Limits

`config.yml: limits` sets a 40 GB budget for `artifacts/` + `tmp/`, enforced by
`evaluate()`, and recipes cap single files with `ulimit -f` and SAGA calls with `timeout`.
`python prepare.py usage` shows usage, `python prepare.py prune` deletes feature
directories no kept run uses. These are guardrails for a cooperative agent, not a
sandbox: run it under an account that cannot modify `data/`.

Earlier pipelines are kept for reference in `legacy/` (the dynamic-feature loop and the
guarded runtime) and their outputs in `artifacts/legacy/`.
