# autosearch_lab

Self-contained autonomous research loop that searches for the best
**interpretable static susceptibility model** — where a point-mapped natural
hazard tends to happen — using Explainable Boosting Machines, in the spirit of
[karpathy/autoresearch](https://github.com/karpathy/autoresearch): an agent
edits code, trains, keeps or discards against a git-backed ledger, and repeats
to convergence.

This is the "back to basics" redesign of `autosearch_spacetime`: instead of
selecting from a bundle of ~14 precomputed static grids, the agent starts from
raw primitives only — a DEM, dynamic climate NetCDF (precipitation,
temperature), and vector layers (roads, buildings, Corine land cover,
electrical infrastructure) — and proposes its **own** raster-generation code
(`features.py`) as part of each experiment: terrain derivatives, distance-to-
feature surfaces, custom dynamic aggregates, whatever the hypothesis needs.
The EBM training / spatial-CV / explainability workflow is unchanged.

## What's inside

| File | Role |
|------|------|
| `prepare.py` | Fixed infrastructure: sampling, spatial CV, categorical exclusion-mask tooling, features.py execution/validation/caching, EBM explanation, ledger helper. Never edited. |
| `train.py` | Feature selection (`FEATURES`) + derived aggregates (`DERIVED`) + EBM config. Agent-owned. |
| `features.py` | Agent-owned raster-generation code (`RECIPES`) — the new surface; see its docstring for the contract. |
| `program.md` | The agent's operating contract, including the one-time per-branch exclusion-mask setup step. |
| `config.yml` | The hazard, sampling knobs, search schedule, feature-build timeout. The author's — never edited by the agent except the one documented setup exception. |
| `loop.sh` | Outer driver — relaunches the agent CLI until the search converges. |
| `postprocess.py` | Manual, post-loop reporting (susceptibility maps, the ledger progress plot) — not part of the per-experiment loop. |
| `data/` | Physical copy (not a symlink) of `dem.tif`, `dynamic/`, `vectors/`, `points_*.geojson` — see [DATA.md](DATA.md). Self-contained: the source repos it was copied from can be deleted. |

## Run it

```bash
pip install -r requirements.txt
# data/ is already populated (physical copy) — see DATA.md
python prepare.py --describe-raster-classes --source <path>   # first: judge the exclusion mask (see program.md Setup)
python prepare.py --build-exclude-mask   # once sampling.exclude_raster/exclude_classes are set in config.yml
./loop.sh                                # the agent builds the matrix (prepare.py) if missing, then searches
```

Any CLI `loop.sh` supports drives it — `AGENT=` one of `codex` (default),
`claude`, `opencode` (free DeepSeek), `gemini`, or `ollama` (local). All state
lives in git + the ledger, so you can stop, switch CLIs, and resume from the
same row.

## Deliverables

```bash
python postprocess.py --predict-map      # -> artifacts/maps/susceptibility_<hazard>.tif
python postprocess.py --plot-progress    # -> progress.png (search trajectory)
python prepare.py --describe-primitives   # inventory of raw data available to features.py
python prepare.py --prune-feature-cache   # reclaim disk from unreachable cached rasters
```

- `artifacts/experiments/experiments.tsv` — the ledger: every experiment, its
  score, and a narrative description.
- `artifacts/run_best/explain.md` — what the best model learned (term
  importances + shape functions in plain text).
- `artifacts/models/best_model.pkl` — the winner.
- `artifacts/features_cache/` — cached `features.py` recipe rasters, capped at
  15GB total and keyed on `features.py`'s own source so a code edit always
  invalidates stale entries; never recomputed once cached for the current hash.

## Reproducibility: verified baseline

Commit `58e7306` ("Clean baseline: fresh history under this session's infra
changes") is the search's starting point (ledger iteration 0, `status=baseline`,
`FEATURES`/`DERIVED`/`RECIPES` all empty). It was independently reproduced in
an isolated `git worktree` — genuinely empty `artifacts/derived/`,
`features_cache/`, `features_scratch/`, `features/` (no gitignored cache
carried over), raw `data/` untouched (read-only, shared via symlink) — running
only `python prepare.py` (feature matrix) then `python train.py` (no
`--warm-derived-cache` needed: the null baseline's `DERIVED` list is empty).
Result: `val_pr_auc=0.167498`, `val_roc_auc=0.500000`, `val_pr_gain_auc=0.0`,
`spec_hash=a5656eeed786` — exactly matching the committed ledger row. The
run's own `assert_new_experiment()` duplicate guard was also independently
confirmed: re-running against the live ledger correctly refuses the repeat.
