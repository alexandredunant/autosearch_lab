# Archived agent instructions — historical reference only

You are an autonomous ML engineer working in one repository. Your goal: the best
interpretable **static susceptibility model** — maximize `val_pr_auc` with an
`interpret.glassbox.ExplainableBoostingClassifier` on the single hazard process
in `project.processes`. You run many experiments in a row, one after another.
**Do not voluntarily stop**: once the loop has begun, do not pause to ask the
human whether to continue — keep going experiment after experiment until the
discard budget is exhausted.

## Any CLI, any session — state lives in git
It does not matter which agentic CLI you are (Codex, Claude Code, opencode,
Gemini, …) or whether this is a fresh session. **Everything you need is in git
and the ledger**, so the run is fully resumable:
- You may be interrupted at any time — a token/quota limit, a crash, a closed
  terminal. That is fine and expected.
- A later session resumes with zero loss: the **same CLI once its quota
  refreshes, or a *different* CLI**, reads git + the ledger and continues.
- So never treat "I'm running low on quota" as a reason to rush or improvise.
  Finish the current experiment cleanly (or don't start a new one); the next
  session picks up exactly where the ledger ends.

## The four files
- `prepare.py` — fixed infrastructure (data, sampling, evaluation protocol,
  derived aggregates, features.py execution/validation/caching, categorical
  exclusion-mask tooling, EBM explanation, ledger helper). Never edit.
- `train.py` — yours. Feature selection (`FEATURES`), derived aggregates
  (`DERIVED`), EBM config, rationale, pipeline.
- `features.py` — yours. Where you propose raster-creation code (terrain
  derivatives, distance-to-road/building surfaces, custom dynamic aggregates,
  or anything else the data supports) as `RECIPES`. See its own docstring for
  the exact contract.
- `program.md` — this contract. Never edit. (`config.yml` — the author's,
  with ONE documented exception: the Setup section below.)

Not part of this loop: `postprocess.py` (susceptibility-map export, the
ledger progress plot) is a manually-run, post-loop reporting tool the human
runs — not read or run by the per-experiment loop.

## The ledger is the only narrative
The ledger lives at **`artifacts/experiments/experiments.tsv`**. There is no
lab-notes file. Each row's **description is verbose** — one to three sentences
stating what you changed, the result, *why* it did or didn't help (cite a term
importance or fold-variance observation), and the next axis you'd try. Later
experiments read those descriptions back. Write the description as your note.

## Setup (first run per hazard branch — ONCE, before the null baseline)
This is the one documented exception to "config.yml is the author's, never
edited by the agent" — scoped strictly to this one-time pass. Once done,
`config.yml` is frozen for the rest of this branch's run exactly as before;
you never revisit `sampling.exclude_raster`, `sampling.exclude_classes`,
or `mask_exclude.tif` again, and neither does anything in `features.py`.

1. Run `python prepare.py --describe-primitives` and pick a categorical
   raster relevant to the hazard, by name and content — land cover, geology,
   demographic zoning, soil type, or anything else that's present. This
   framework doesn't assume any specific product or domain.
2. Run `python prepare.py --describe-raster-classes --source <path>`. It lists
   every class code present on the reference grid with pixel counts — no
   label table is bundled here, so use your own knowledge of that
   classification scheme's codes (e.g. Corine's well-known level-3
   nomenclature) to interpret them.
3. Judge which classes are physically implausible non-event locations **for
   this branch's specific hazard** — the judgment depends on the hazard, not
   a fixed list. For example: wildfire plausibly excludes water, glaciers,
   and bare rock (no fuel); landslide's exclusion list is typically narrower
   (open water and glaciers, rarely more). Use your own reasoning about the
   hazard process, not a template.
4. Write the raster path to `config.yml`'s `sampling.exclude_raster` and the
   chosen codes to `sampling.exclude_classes`, then commit:
   `git add config.yml && git commit -m "setup: exclusion mask for <hazard>
   (raster: <path>, classes: <codes>, reasoning: <one sentence>)"`.
5. Run `python prepare.py --build-exclude-mask` — this writes
   `data/grids/mask_exclude.tif`, picked up automatically by the sampling
   domain. It is derived data (gitignored under `data/`), not committed.
6. Proceed to the compulsory first step below (build the feature matrix), then
   the null baseline, exactly as on every subsequent session.

## Recover state (do this every time you start or resume)
Read directly — there is no digest helper.
- **COMPULSORY FIRST STEP — the feature matrix must exist.** Check
  `artifacts/features/<process>/matrix.npz` (`<process>` = the key in
  `config.yml`'s `project.processes`); if it is missing, run
  `python prepare.py` and wait for it to finish. No experiment can run
  without it, and it is never committed — a fresh checkout or a hazard
  switch always starts without one.
- **`data/grids/mask_exclude.tif` is gitignored (lives under `data/`), so a
  branch switch does NOT restore the right one for the branch you land on.**
  `config.yml`'s `sampling.exclude_raster`/`exclude_classes` ARE
  branch-tracked and switch correctly — but the mask file itself needs
  regenerating to match:
  after any `git checkout <branch>`, run `python prepare.py
  --build-exclude-mask` before `python prepare.py`, even if `matrix.npz`
  already looks present from a stale mask. When in doubt, regenerate both.
- `git log --oneline -15` and `git status` — current branch and commit.
- `cat artifacts/experiments/experiments.tsv` — every experiment and its
  verbose description; the best score is the highest `keep`/`baseline` row, and
  D is the count of consecutive `discard`/`crash` rows since the last `keep`.
  `timestamp`/`agent`/`model` are recorded automatically by `log_experiment()`
  (not something you supply) — useful context if the run switched CLIs
  mid-way (e.g. a usage-limit fallback), but not something to reason about
  or optimize for.
- `cat artifacts/run_best/explain.md` (if present) — what the current best
  model learned: term importances and shape functions.
- Read `train.py` and `features.py` — the current best configuration.
Keep it lean: read each once, do not re-cat large files repeatedly, and never
echo full training logs into the conversation.

## Hard rules
1. Model family is fixed: EBM only. Interpretability is the point.
2. Metric is fixed: `val_pr_auc` from `train.py` — mean PR-AUC over the spatial
   K-folds — is the ONLY decision metric. Never change folds, metric, seed, or
   aggregation. `val_roc_auc` is secondary/informational.
3. Keep only when `val_pr_auc >= best + 0.001`; else discard.
4. Hazard-agnostic: the process is whatever `config.yml` names; features are
   whatever rasters, archives, and vector layers exist. Hardcode no assumptions.
5. Never re-run a logged experiment; `train.py` computes a `spec_hash` (which
   also covers `features.py`'s own source — a raster-generation code change
   alone counts as a new experiment) and refuses duplicates. Every experiment
   must change the RESOLVED feature set, `DERIVED`, `features.py`'s recipes,
   or `MODEL_CONFIG` — no-op `exclude` patterns that match nothing are rejected.
6. Don't ask the human anything. Pick the strongest non-duplicate idea and run.
7. Coordinates are not predictors. `metadata.csv` carries `x`, `y` so you can
   understand and group the sample; a model that scored rows by *where* they
   were drawn from directly would be reading our sampling design rather than
   the terrain. Use them to reason, never as a feature.
8. **`features.py` must never see labels.** It has no access to
   `data/points_*.geojson` and must never import `load_points`,
   `build_sample`, `sample_absences`, or `spatial_folds` from `prepare` —
   `prepare.py` refuses to execute a `features.py` that does (a crash, not a
   silent pass). A recipe builds a predictor from raw primitives only (DEM,
   dynamic archives, vectors); it is never told where the events are.
9. `features.py` recipes must use `NaN` for nodata (not a sentinel like
   `-9999`), consistent with every other predictor in this repo.
10. Disk budgets are enforced, not advisory: a single cached feature raster
    over 2x `dem.tif`'s file size, or a cache addition that would push
    `artifacts/features_cache/` over 15GB total, is refused (a crash). Run
    `python prepare.py --prune-feature-cache` to reclaim space from recipes no
    longer in `RECIPES`, or from stale hashes of kept ones — this never loses
    a live recipe's current cache, only unreachable entries.

## What an observation is
Every row is a location; `label` says whether the process is known to have
occurred there. Non-events are **generated**: random background locations in
the valid domain (optionally restricted by the categorical exclusion mask from
Setup), kept at least `event_buffer_m` from every known event, fixed seed.
Scores are therefore *relative* — as susceptibility always is.

## The predictors: static grids, derived aggregates, and features.py recipes
Three tiers, one `FEATURES` namespace (fnmatch include/exclude) — a name
collision between tiers is a hard error, never a silent shadow:
- **Static grids** — `data/grids/dem.tif` is the only seed grid (the reference
  grid). List available names from `artifacts/features/<process>/` after the
  first `prepare.py` run.
- **Antecedent dynamic-archive features** (also base columns in
  `artifacts/features/<process>/`, named `<var>_sum_<N>d`/`<var>_mean_<N>d`,
  e.g. `precipitation_sum_7d`) — sum/mean of each dynamic variable over the N
  most recent days up to and including each observation's OWN date (real
  event dates for presences; assigned pseudo-dates, resampled from real event
  dates with a spatiotemporal buffer, for background — see
  `build_antecedent_features()`/`assign_absence_dates()` in `prepare.py`).
  Unlike `DERIVED`, these are date-linked per observation, not a single
  climatology raster shared by every point — this is what actually lets a
  climate variable's *triggering* signal show up, not just its long-term
  spatial average. Precomputed once by `prepare.py`, not configurable per
  experiment beyond which columns you `include`. The search-order rule below
  applies to these exactly as it does to `DERIVED`: they're dynamic-archive
  features, so test them before adding new static RECIPES.
- **Derived aggregates (`DERIVED`, in `train.py`)** — a fixed, declarative form
  for reducing `data/dynamic/<var>/*.nc` to one static per-cell summary:
  `{"var": <archive>, "stat": "mean"|"min"|"max"|"std", "months": [1-12]}`
  (`months` optional). Run `python prepare.py --describe-dynamic` to see what
  archives exist. Good for a simple aggregate; for anything else, use RECIPES.
  **The no-`months` (full-archive) form of all four stats, for every
  discovered variable, is pre-built by `loop.sh` before your session starts**
  (`python prepare.py --warm-derived-cache`) — trying any of those is always
  an instant cache hit, never a fresh archive scan. A `months`-filtered
  variant is NOT pre-built (the combination space is unbounded) and still
  triggers a fresh scan on first use, carrying the timeout exposure in the
  Run step below.
- **`features.py` recipes (`RECIPES`)** — arbitrary raster-generation code
  starting from the raw primitives: `data/grids/dem.tif`, `data/dynamic/`,
  `data/vectors/{roads.gpkg, buildings.gpkg, CorineLandCover_polygon.tif,
  electrical_infrastructure.*}`. Run `python prepare.py --describe-primitives`
  to see what's available. This is where terrain derivatives (slope, aspect,
  curvature, TWI, LS-factor, ...), distance-to-feature surfaces, or custom
  dynamic aggregates the `DERIVED` schema can't express belong. **Prefer
  `saga_cmd` (SAGA GIS)** for terrain analysis and distance transforms —
  fast (whole-DEM terrain analysis is a few seconds) and robust enough to run
  live inside this recipe's own subprocess-isolated `feature_build_timeout_s`
  call, rather than hand-rolling one with numpy/scipy. Every recipe's output
  is validated against the reference grid (no silent reprojection) and cached
  by a hash of `features.py`'s own source — editing the file, even one
  recipe, invalidates every cached raster; a recipe is never recomputed once
  cached for its current hash.

### Paths available to `build_<name>(ctx)` recipes
`ctx` is plain data, no methods — to shell out to an external tool, call
`subprocess.run(cmd, timeout=N, cwd=ctx.scratch_dir, check=True)` directly.
- `ctx.dem_path` — the reference grid (`data/grids/dem.tif` or its
  alphabetical-fallback equivalent).
- `ctx.dynamic_dir` — `data/dynamic/`, one subdirectory per variable.
- `ctx.vectors_dir` — `data/vectors/`, whatever layers actually exist.
- `ctx.scratch_dir` — wiped before every run; throwaway intermediates only.
- `ctx.reference_profile` — dict with `crs`, `transform`, `height`, `width`:
  what a returned raster MUST match. Return a bare `np.ndarray` of that exact
  shape and prepare.py writes the GeoTIFF for you (simplest, safest); return
  a `Path` only if a CLI chain naturally emits one — it's then validated
  strictly against this profile (no silent reprojection).

## One-experiment loop (repeat until the budget is exhausted)
1. **State**: recover as above. Note the best score and D.
2. **Choose ONE change**, deduced from the evidence (term importances + the
   ledger descriptions). Read the phase boundaries from `config.yml`
   (`search: phase_feature / phase_hyper / phase_cluster / discard_budget`)
   and obey them strictly by D:
   - **D < phase_feature → FEATURE PHASE. Change ONLY the feature space**
     (`FEATURES`, `DERIVED`, and/or `features.py`'s `RECIPES`). Changing
     `MODEL_CONFIG` (interactions, bins, learning rate, bags, early stopping)
     in this phase is FORBIDDEN — leave it exactly as the current best.
     Exhaust the feature space first: different include/exclude combinations,
     new or removed recipes, add or remove derived aggregates, prune the weak
     terms `explain.md` flags, restore dropped ones. Do NOT reach for
     hyperparameters early just because features stopped improving — that is
     precisely when to try a *different* feature idea.
     **Search-order rule: test every full-archive `DERIVED` aggregate (the
     pre-warmed mean/min/max/std for each discovered dynamic variable — no
     `months` filter, already an instant cache hit) and every antecedent
     dynamic-archive column (`<var>_sum_<N>d`/`<var>_mean_<N>d`, see below)
     before adding any new static `features.py` RECIPE (DEM- or
     vector-derived).** Greedy forward
     selection is order-dependent — a dynamic variable correlated with
     terrain (e.g. precipitation with elevation, via orographic effect) will
     look artificially weak if terrain is already locked in first, because
     its shared variance was already claimed. Testing dynamic candidates
     early gives them a fair, less order-biased read on their own marginal
     value. This does not mean dynamic features must be *kept* — a
     fairly-tested one that still doesn't clear the keep threshold is
     genuinely uninformative, not just disadvantaged by ordering. Once every
     variable's 4 full-archive `DERIVED` stats AND every antecedent column
     have been tested (kept or discarded), static RECIPES and month-filtered
     `DERIVED` variants are both fair game,
     in any order — `months` filtering is combinatorially unbounded, so it is
     exploratory like any other feature idea, not part of this ordering gate.
   - phase_feature ≤ D < phase_hyper → HYPERPARAMETER PHASE: now change
     `MODEL_CONFIG` (features may stay fixed).
   - phase_hyper ≤ D < phase_cluster → CLUSTERING PHASE: change ONLY the
     feature space again, restricted to spatial/temporal clustering-derived
     `features.py` recipes — e.g. distance to the nearest spatial cluster
     centroid of observations (k-means/DBSCAN on x/y), a categorical spatial
     zone ID, or a seasonal/temporal cluster membership term. Same
     `RECIPES` mechanism, same leakage guard, same keep/discard rule on
     `val_pr_auc` as any other recipe — this is a restricted idea-space, not
     a different evaluation method. `MODEL_CONFIG` stays fixed at whatever
     the hyperparameter phase left it at.
   - phase_cluster ≤ D < discard_budget → anything untested (any kind).
   - D ≥ discard_budget → STOP and create `.autoresearch_done`.
3. **Commit the candidate**: edit `train.py` and/or `features.py`, then
   `git add train.py features.py && git commit -m "Candidate: <short desc>"`.
4. **Run**: `python -u train.py > run.log 2>&1` then
   `grep -E "^val_(pr|roc|pr_gain)_auc:" run.log`. `val_pr_auc` is the decision
   metric; `val_roc_auc` and `val_pr_gain_auc` (AUPRG, see `prg_metric.py`) are
   secondary/informational. A `features.py` recipe that hangs
   or exceeds `search.feature_build_timeout_s` is killed automatically and
   surfaces as a crash — treat it the same as any other crash. Kill runs over
   20 min = `crash`.
   **If your tool-calling harness lets you set a timeout/duration for this
   shell command, set it explicitly to at least 20 minutes.** Do not rely on
   whatever default your harness applies to a single command — several
   agentic CLIs default to well under a minute, which is shorter than a full
   5-fold EBM fit can legitimately take. A command killed by your own
   harness's default timeout is indistinguishable from a real crash in the
   ledger (it dies mid-fold with no metrics emitted) and wastes the
   experiment for a reason that has nothing to do with the candidate under
   test. The 20-minute ceiling above is the one that should end a run, not an
   unrelated tool default shorter than that.
5. **Keep or discard** on `val_pr_auc` only. Capture first (the hash vanishes on
   discard): `CAND=$(git rev-parse --short HEAD)`,
   `SPEC=$(grep "^spec_hash:" run.log | awk '{print $2}')`,
   `ROC=$(grep "^val_roc_auc:" run.log | awk '{print $2}')`, and
   `PRG=$(grep "^val_pr_gain_auc:" run.log | awk '{print $2}')`.
   - keep: copy `artifacts/run_current/` → `artifacts/run_best/` and the
     bundle → `artifacts/models/best_model.pkl`.
   - discard/crash: `git reset --hard HEAD~1`.
6. **Record** with the helper (never hand-edit the TSV). The `--description` is
   your narrative (1–3 sentences: what changed, result, why, next idea):
   `python prepare.py --log-experiment --commit "$CAND" --score <val_pr_auc>
   --roc-auc "$ROC" --pr-gain-auc "$PRG" --status <keep|discard|crash|baseline>
   --description "<what changed; result; why; next axis>" --spec-hash "$SPEC"`
   (`--score 0.0 --status crash` for crashes; omit `--roc-auc`/`--pr-gain-auc` on crash).
7. **Commit the record**: keep → `git add -A && git commit --amend --no-edit`;
   discard/crash → `git add artifacts/experiments && git commit -m "log
   experiment <N>: val_pr_auc <score> (<status>, <desc>)"`.
8. **Go straight to the next experiment.** Do not voluntarily stop or ask.

## Fresh run (empty ledger)
If `artifacts/experiments/experiments.tsv` has only its header (or does not
exist): the Setup section above has already run (config.yml's
`exclude_classes` is set, `mask_exclude.tif` exists) and the feature
matrix is already in place (the compulsory first step above), and the first
experiment is the null baseline (`FEATURES = {"include": [], "exclude": []}`,
`DERIVED = []`, `features.py`'s `RECIPES = []`), logged with `--status
baseline`. The ledger refuses any first row that is not the baseline.
