#!/usr/bin/env python3
"""Fixed utilities for the autoresearch loop. The agent must never edit this file.

Everything that is not the experiment lives here. This is the "back to basics,
more room for the agent" redesign of autosearch_spacetime/autosearch_susceptibility:
1. Data contract + configuration: hazard processes are inferred from file
   naming; author settings come from an optional config.yml with code defaults.
2. Data preparation CLI: hazard-occurrence POINTS (data/points_<key>.geojson)
   plus static predictor GRIDS (data/grids/*.tif) become per-process feature
   matrices under artifacts/features/. Non-events are GENERATED: random
   background locations in the valid domain, buffered away from every known
   event, fixed seed, optionally restricted by an exclusion mask built from
   any categorical raster -- land cover, geology, demographic zoning,
   whatever fits the hazard (see "Class-based exclusion mask" below). If the
   points file carries a "date" column, build_antecedent_features() also adds
   per-observation, date-linked sum/mean columns of every dynamic archive
   variable over 1/7/30/60-day antecedent windows (background points get a
   pseudo-date via assign_absence_dates(), see DATA.md) -- unlike DERIVED
   below, these can capture a short-lived triggering signal, not just a
   long-term spatial climatology.
3. Fixed evaluation protocol: spatial K-fold cross-validation (KMeans blocks
   on coordinates) sets the decision metric val_pr_auc that train.py must use
   unchanged. This is the goalpost of the search.
4. Derived static grids: derived_grid() reduces a time-varying archive
   (data/dynamic/<var>/*.nc) to a per-cell static aggregate GeoTIFF (cached),
   so temporal data can enter the model only as another static predictor.
   --describe-dynamic documents what archives exist.
5. Agent-authored feature recipes: unlike the old DERIVED-only design, the
   agent may write ARBITRARY raster-generation code in features.py (terrain
   derivatives from the DEM, distance-to-road/building surfaces, custom
   dynamic aggregates, ...). Each recipe runs in an isolated, timed subprocess;
   its output is validated against the reference grid (no silent
   reprojection), cached by a hash of features.py's own source + a data
   fingerprint (so a code change always invalidates stale cache), and is
   never recomputed once cached. Two disk-budget guardrails apply: no single
   cached raster may exceed 2x dem.tif's file size, and artifacts/features_cache/
   as a whole is capped at 15GB (--prune-feature-cache reclaims space).
6. Class-based exclusion mask: the agent picks a categorical raster from
   --describe-primitives's inventory (by file name/content -- this framework
   assumes nothing about which product, e.g. Corine, is in use, if any),
   inspects its codes with --describe-raster-classes --source <path>, and judges,
   ONCE per hazard branch during setup, which classes are physically
   implausible non-event locations for that specific hazard (using its own
   knowledge of that classification scheme's codes -- no lookup table is
   bundled here). It writes the raster path to config.yml's
   sampling.exclude_raster and the codes to sampling.exclude_classes,
   then --build-exclude-mask turns that into data/grids/mask_exclude.tif (read
   automatically by valid_domain()). This is the one documented, setup-only
   exception to "config.yml is the author's, never edited by the agent" —
   once written, it is frozen for the rest of that branch's run.
7. EBM explanation: explain.md (term importances + TalkToEBM graph-to-text)
   so every experiment's model stays inspectable.
8. Ledger: log_experiment() owns artifacts/experiments/experiments.tsv --
   iteration numbering, delta arithmetic, best-score bookkeeping, and the
   spec_hash duplicate-experiment check -- so no agent session can hand-mangle
   the TSV or silently re-run a logged experiment.

Prediction maps and the progress plot: see postprocess.py.

Data contract (hazard-agnostic, inferred from naming):
  data/points_<key>.geojson   one hazard process per file: event POINTS. <key>
                              becomes a `project.processes` entry -- discovered
                              from whatever files are actually present, not a
                              fixed list of hazards.
  data/grids/dem.tif          the reference grid: defines the sampling domain
                              and output resolution. "dem" is a naming
                              convention (used when present); with no file by
                              that name, discover_grids() falls back to
                              whichever *.tif sorts first alphabetically -- any
                              single-band static raster works.
  data/grids/mask_exclude.tif optional; nonzero cells are excluded from the
                              sampling domain (see "Class-based exclusion mask")
  data/dynamic/<var>/*.nc     OPTIONAL time-varying rasters, one directory per
                              variable; the directory name IS the variable
                              name, over dims (DATE, y, x) -- discovered
                              generically, no fixed variable list
  data/vectors/*              OPTIONAL vector/raster layers of any kind (roads,
                              buildings, land cover, geology, ...), in whatever
                              format and under whatever names the dataset
                              actually has (.gpkg, .shp, .tif, ...) --
                              discovered generically via --describe-primitives.
                              features.py (rewritten per dataset) is what
                              actually names and consumes specific files here;
                              this module hardcodes none of them.
Only dem.tif (or its alphabetical-fallback) and the points files are
structurally required. Everything else is optional and named by convention or
discovery, never assumed -- this file must run unmodified against a
differently-named dataset in a different region for a different hazard.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

# pip-installed rasterio wheels bundle their own PROJ database; a conda env's
# PROJ_DATA pointing at an older proj.db breaks every CRS lookup. Prefer the
# bundled data when it exists (no-op for conda-installed rasterio).
_spec = importlib.util.find_spec("rasterio")
if _spec and _spec.origin:
    _bundled_proj = Path(_spec.origin).parent / "proj_data"
    if _bundled_proj.exists():
        os.environ["PROJ_DATA"] = str(_bundled_proj)
        os.environ.pop("PROJ_LIB", None)

# A .shp missing its .shx sidecar (a real-world data-quality issue in some
# shapefile sources, not specific to any one dataset here) would otherwise
# fail to open; GDAL can rebuild the index on the fly instead.
os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import yaml  # noqa: E402
from rasterio.transform import xy as transform_xy  # noqa: E402
from rasterio.warp import Resampling, reproject  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from prg_metric import auprg_score  # noqa: E402

log = logging.getLogger("prepare")

# ============================================================
# Configuration — code defaults, optional config.yml override
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent
POINTS_PATTERN = "points_*.geojson"
EXCLUDE_MASK_NAME = "mask_exclude.tif"
LEDGER_PATH = Path("artifacts/experiments/experiments.tsv")
BEST_SCORE_PATH = Path("artifacts/experiments/best_score.txt")


LEDGER_COLUMNS = ("iteration", "commit", "val_pr_auc", "val_roc_auc", "delta", "status",
                  "process_scores", "description", "spec_hash",
                  "timestamp", "agent", "model", "val_pr_gain_auc")
NULL_FEATURE_NAME = "__null_feature__"

# Time-varying archives live under data/dynamic/<var>/*.nc. The model is
# static: an archive participates only as per-cell temporal aggregates
# (derived_grid), cached as GeoTIFFs under artifacts/derived/.
DYNAMIC_DIRNAME = "dynamic"
DERIVED_DIR = Path("artifacts/derived")
DERIVED_SCRATCH_DIR = Path("artifacts/derived_scratch")  # mirrors FEATURES_SCRATCH_DIR
DERIVED_STATS = ("mean", "min", "max", "std")

# Agent-authored feature recipes (features.py) — see module docstring point 5.
FEATURES_CACHE_DIR = Path("artifacts/features_cache")
FEATURES_SCRATCH_DIR = Path("artifacts/features_scratch")
FEATURES_CACHE_BUDGET_BYTES = 15 * 1024**3  # 15GB, user-specified
FEATURES_MODULE_NAME = "features.py"
# features.py must never import these prepare.py symbols or reference the
# points files by name — see scan_features_module().
FEATURES_FORBIDDEN_IMPORTS = {"load_points", "build_sample", "sample_absences", "spatial_folds"}
FEATURES_FORBIDDEN_SUBSTRINGS = ("points_", ".geojson")

@dataclass(frozen=True)
class ProjectConfig:
    name: str
    data_dir: Path
    grids_dir: Path
    artifact_dir: Path
    processes: tuple[str, ...]
    random_state: int
    # Sampling (presence vs generated spatial background):
    background_ratio: int         # background points per presence
    event_buffer_m: float         # how far a background point must sit from any event
    # Antecedent dynamic-archive features (see build_antecedent_features()):
    # for each dynamic variable, sum and mean over the N most recent days up
    # to and including each observation's own date (real event dates for
    # presences; assigned pseudo-dates for background, see assign_absence_dates()).
    antecedent_windows_days: tuple[int, ...]
    # A background point's assigned pseudo-date is rejected and redrawn if any
    # real event within antecedent_buffer_radius_m falls within
    # antecedent_buffer_days of it -- avoids labeling "regional wet conditions
    # that also triggered a nearby real event" as a clean negative.
    antecedent_buffer_days: int
    antecedent_buffer_radius_m: float
    # Setup-time-only, see module docstring point 6. exclude_raster is whichever
    # data/vectors/*.tif the agent judged to be a relevant categorical layer --
    # land cover, geology, demographic zoning, or any other classification --
    # by inspecting --describe-primitives's inventory; not assumed by this
    # framework to be any specific product or domain.
    exclude_raster: str | None
    exclude_classes: tuple[int, ...]
    # Fixed evaluation protocol:
    k_folds: int              # spatial K-fold blocks
    # Search schedule (author-owned; the agent reads it, never changes it):
    phase_feature: int
    phase_hyper: int
    phase_cluster: int
    discard_budget: int
    feature_build_timeout_s: float  # per-recipe wall-clock budget, see build_feature_grid()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def infer_processes(data_dir: Path) -> tuple[str, ...]:
    """One hazard process per data/points_<key>.geojson file."""
    keys = tuple(
        path.stem[len("points_"):] for path in sorted(data_dir.glob(POINTS_PATTERN))
    )
    if not keys:
        raise FileNotFoundError(f"No {POINTS_PATTERN} event files found in {data_dir}")
    return keys


def load_config(path: str | Path = "config.yml") -> ProjectConfig:
    """Load project config; without a config file every setting falls back to defaults."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    payload: dict[str, Any] = {}
    if config_path.exists():
        payload = yaml.safe_load(config_path.read_text()) or {}
    project = payload.get("project", {})
    sampling = payload.get("sampling", {})
    search = payload.get("search", {})
    data_dir = _resolve(REPO_ROOT, project.get("data_dir", "data"))
    return ProjectConfig(
        name=str(project.get("name", "hazard_autoresearch")),
        data_dir=data_dir,
        grids_dir=_resolve(REPO_ROOT, project.get("grids_dir", "data/grids")),
        artifact_dir=_resolve(REPO_ROOT, project.get("artifact_dir", "artifacts")),
        processes=tuple(project.get("processes", ())) or infer_processes(data_dir),
        random_state=int(project.get("random_state", 42)),
        background_ratio=int(sampling.get("background_ratio", 5)),
        event_buffer_m=float(sampling.get("event_buffer_m", 150.0)),
        antecedent_windows_days=tuple(int(w) for w in sampling.get("antecedent_windows_days", (1, 7, 30, 60))),
        antecedent_buffer_days=int(sampling.get("antecedent_buffer_days", 730)),
        antecedent_buffer_radius_m=float(sampling.get("antecedent_buffer_radius_m", 2000.0)),
        exclude_raster=sampling.get("exclude_raster") or None,
        exclude_classes=tuple(int(c) for c in sampling.get("exclude_classes", ())),
        k_folds=int(sampling.get("k_folds", 5)),
        phase_feature=int(search.get("phase_feature", 10)),
        phase_hyper=int(search.get("phase_hyper", 20)),
        phase_cluster=int(search.get("phase_cluster", 25)),
        discard_budget=int(search.get("discard_budget", 30)),
        feature_build_timeout_s=float(search.get("feature_build_timeout_s", 900)),
    )


def parse_processes(config: ProjectConfig, value: str | None) -> list[str]:
    keys = config.processes if not value or value == "all" else tuple(
        item.strip() for item in value.split(",") if item.strip()
    )
    missing = [key for key in keys if key not in config.processes]
    if missing:
        raise KeyError(f"Unknown process key(s): {missing}. Available: {sorted(config.processes)}")
    return list(keys)


# ============================================================
# Fixed evaluation protocol — the goalposts of the search
# ============================================================


def scalar_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    out = {
        "pr_auc": float("nan"), "roc_auc": float("nan"),
        "pr_gain_auc": float("nan"), "prevalence": float(np.mean(y_true)),
    }
    if len(np.unique(y_true)) == 2:
        out["pr_auc"] = float(average_precision_score(y_true, prob))
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
        # Secondary/informational only (see prg_metric.py): PR-AUC's arithmetic-mean
        # area can rank a classifier above another with strictly better expected F1.
        # AUPRG is tracked in the ledger to watch for disagreement, not to decide.
        out["pr_gain_auc"] = auprg_score(y_true, prob)
    return out


def spatial_folds(
    y: np.ndarray, metadata: pd.DataFrame, k: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Spatial K-fold: KMeans blocks on coordinates, each block once as validation.

    Generalization to unseen terrain is the primary goalpost, so the model is
    always validated on locations spatially disjoint from its training data.
    Blocks whose validation part is single-class are skipped.
    """
    xy = metadata[["x", "y"]].to_numpy(dtype=np.float64)
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(xy)
    folds = []
    for block in range(k):
        val_idx = np.flatnonzero(labels == block)
        train_idx = np.flatnonzero(labels != block)
        if len(np.unique(y[val_idx])) == 2 and len(np.unique(y[train_idx])) == 2:
            folds.append((train_idx, val_idx))
    if len(folds) < 2:
        raise ValueError("Fewer than 2 usable spatial folds; check the sample")
    return folds


# ============================================================
# Experiment identity — mechanical duplicate rejection
# ============================================================


def experiment_hash(spec: dict[str, Any]) -> str:
    """Deterministic identity of an experiment specification."""
    payload = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def assert_new_experiment(spec_hash: str, ledger_path: str | Path = LEDGER_PATH) -> None:
    """Refuse to run an experiment whose spec_hash is already in the ledger.

    Two identical experiments must never exist: this is the mechanical guard
    against repeat/infinite loops, independent of the agent's own memory.
    """
    ledger = Path(ledger_path)
    if not ledger.exists():
        return
    lines = ledger.read_text().strip().splitlines()
    if not lines:
        return
    header = lines[0].split("\t")
    if "spec_hash" not in header:
        return
    col = header.index("spec_hash")
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) > col and parts[col] == spec_hash:
            raise RuntimeError(
                f"Duplicate experiment: spec_hash {spec_hash} already logged in "
                f"{ledger} (iteration {parts[0]}). Design a different experiment."
            )


def _looks_hex(value: str, min_len: int, max_len: int) -> bool:
    return (min_len <= len(value) <= max_len
            and all(c in "0123456789abcdef" for c in value))


def log_experiment(
    process: str,
    commit: str,
    score: float,
    status: str,
    description: str,
    spec_hash: str,
    roc_auc: float = float("nan"),
    pr_gain_auc: float = float("nan"),
    ledger_path: Path = LEDGER_PATH,
    best_score_path: Path = BEST_SCORE_PATH,
) -> str:
    """Append one mechanically formatted row to the experiment ledger.

    The agent supplies the values; this function owns the bytes. Iteration
    numbering, delta arithmetic, tab separators, and best-score bookkeeping
    are computed here so no session can hand-mangle the TSV. `timestamp`,
    `agent`, and `model` are filled in automatically (UTC ISO-8601 wall
    clock; AUTORESEARCH_AGENT/AUTORESEARCH_MODEL env vars, "unknown" if
    unset) rather than trusted to the agent's own self-report — loop.sh sets
    those two env vars before launching each session, so a mid-run agent
    switch (e.g. hitting one provider's usage limit) is recorded accurately
    regardless of what the model itself believes it is. `val_pr_gain_auc`
    (AUPRG, see prg_metric.py) is informational only, tracked to watch for
    disagreement with the val_pr_auc decision metric; it never affects
    keep/discard status or best_score bookkeeping. Returns the row.
    """
    if status not in ("baseline", "keep", "discard", "crash"):
        raise SystemExit(f"status must be baseline|keep|discard|crash, got '{status}'")
    if not _looks_hex(commit, 7, 40):
        raise SystemExit(f"--commit '{commit}' does not look like a git hash")
    if not _looks_hex(spec_hash, 12, 12):
        raise SystemExit(f"--spec-hash '{spec_hash}' must be the 12-char hash from run.log")
    description = " ".join(description.split())
    if not description:
        raise SystemExit("--description must not be empty")

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    if not ledger_path.exists() or not ledger_path.read_text().strip():
        ledger_path.write_text("\t".join(LEDGER_COLUMNS) + "\n")
    iteration = len(ledger_path.read_text().strip().splitlines()) - 1

    # The reference score comes first: the search starts from the null baseline
    # so every later delta is measured against something (see program.md).
    if status != "baseline" and iteration == 0:
        raise SystemExit(
            "the first ledger row must be the null baseline (--status baseline): "
            "every later experiment is measured against it. See program.md."
        )

    best = float(best_score_path.read_text()) if best_score_path.exists() else None
    delta = 0.0 if best is None else score - best
    if status == "keep" and best is not None and delta < 0.001:
        raise SystemExit(
            f"status 'keep' requires score >= best + 0.001 (score {score:.6f}, "
            f"best {best:.6f}); log this as 'discard'"
        )
    if status in ("baseline", "keep"):
        best_score_path.write_text(f"{score:.6f}\n")

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    agent = os.environ.get("AUTORESEARCH_AGENT", "unknown")
    model = os.environ.get("AUTORESEARCH_MODEL", "unknown")
    row = "\t".join((
        str(iteration), commit, f"{score:.6f}", f"{roc_auc:.6f}", f"{delta:+.6f}", status,
        f"{process}={score:.6f}", description, spec_hash,
        timestamp, agent, model, f"{pr_gain_auc:.6f}",
    ))
    with ledger_path.open("a") as handle:
        handle.write(row + "\n")
    return row


# ============================================================
# Feature matrices — persistence and name-pattern selection
# ============================================================


def save_artifact(
    out_dir: Path,
    *,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    feature_names: list[str],
    schema: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "matrix.npz", X=X, y=y, feature_names=np.array(feature_names, dtype=object))
    metadata.to_csv(out_dir / "metadata.csv", index=False)
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True))


def load_artifact(process_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str], dict[str, Any]]:
    matrix_path = process_dir / "matrix.npz"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing prepared matrix: {matrix_path}. Run prepare.py first.")
    data = np.load(matrix_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int8)
    feature_names = data["feature_names"].tolist()
    metadata = pd.read_csv(process_dir / "metadata.csv")
    schema = json.loads((process_dir / "schema.json").read_text())
    return X, y, metadata, feature_names, schema


def select_feature_names(feature_names: list[str], selection: dict[str, Any]) -> list[str]:
    """Select features by fnmatch name patterns — no built-in vocabulary.

    A name is selected when it matches any 'include' pattern and no 'exclude'
    pattern. An empty 'include' list is the intentional null baseline and
    yields the single constant null feature.
    """
    include = list(selection.get("include", []))
    exclude = list(selection.get("exclude", []))
    # An exclude pattern that removes nothing is a no-op: it perturbs the spec
    # text (and thus spec_hash) without changing the model the experiment
    # trains, letting an identical experiment masquerade as new. Reject dead
    # patterns so every logged experiment is a genuine change of the feature set.
    dead = [p for p in exclude if not any(fnmatch(name, p) for name in feature_names)]
    if dead:
        raise ValueError(
            f"exclude patterns match no available feature: {dead}. Every exclude "
            "must remove at least one real feature — drop the no-op pattern or "
            "choose a different experiment axis (see program.md)."
        )
    selected = [
        name
        for name in feature_names
        if any(fnmatch(name, pattern) for pattern in include)
        and not any(fnmatch(name, pattern) for pattern in exclude)
    ]
    if include and not selected:
        raise ValueError(
            "FEATURES matched zero available features; an empty 'include' list "
            "is the explicit null baseline"
        )
    return selected or [NULL_FEATURE_NAME]


def selected_matrix(
    X: np.ndarray,
    feature_names: list[str],
    selected: list[str],
    *,
    strict: bool = True,
) -> np.ndarray:
    """Column-subset X to the selected names; the null feature is a zero column.

    With strict=False, selected names absent from feature_names also become
    zero columns (one-hot levels can be missing from a prediction block).
    """
    index = {name: i for i, name in enumerate(feature_names)}
    missing = [name for name in selected if name != NULL_FEATURE_NAME and name not in index]
    if missing and strict:
        raise KeyError(f"Selected features not present in matrix: {missing[:10]}")
    columns = []
    for name in selected:
        if name in index:
            columns.append(X[:, [index[name]]].astype(np.float32))
        else:
            columns.append(np.zeros((X.shape[0], 1), dtype=np.float32))
    return np.hstack(columns).astype(np.float32)


# ============================================================
# Static predictor grids
# ============================================================


def discover_grids(grids_dir: Path) -> dict[str, Path]:
    """Predictor rasters by feature name (file stem); the exclusion mask is not a feature."""
    grids = {
        path.stem: path
        for path in sorted(grids_dir.glob("*.tif"))
        if path.name != EXCLUDE_MASK_NAME
    }
    if not grids:
        raise FileNotFoundError(
            f"No predictor rasters (*.tif) in {grids_dir}. The static pipeline "
            "needs at least one grid (dem.tif)."
        )
    return grids


def is_categorical(name: str) -> bool:
    return name.startswith("cat_")


def reference_grid(grids: dict[str, Path]) -> Path:
    """The grid that defines the sampling domain and the prediction resolution:
    'dem' when present, else the first grid alphabetically."""
    return grids.get("dem", grids[sorted(grids)[0]])


def valid_domain(grids: dict[str, Path], grids_dir: Path) -> tuple[np.ndarray, Any, Any]:
    """Boolean mask of sampleable cells on the reference grid, its transform and CRS.

    Valid = reference grid has data, minus optional mask_exclude.tif nonzero cells.
    """
    ref = reference_grid(grids)
    with rasterio.open(ref) as src:
        band = src.read(1, masked=True)
        mask = ~np.ma.getmaskarray(band)
        if src.nodata is None and np.issubdtype(band.dtype, np.floating):
            mask &= np.isfinite(band.filled(np.nan))
        transform, crs = src.transform, src.crs
    exclude_path = grids_dir / EXCLUDE_MASK_NAME
    if exclude_path.exists():
        with rasterio.open(exclude_path) as src:
            exclude = src.read(1, masked=True).filled(0)
        if exclude.shape != mask.shape:
            raise ValueError(f"{EXCLUDE_MASK_NAME} shape differs from the reference grid")
        mask &= exclude == 0
    if not mask.any():
        raise ValueError("Sampling domain is empty (reference grid all nodata/excluded)")
    return mask, transform, crs


def random_points_in_domain(
    n: int, mask: np.ndarray, transform, crs, rng: np.random.Generator
) -> gpd.GeoSeries:
    """Uniform random points within valid cells (jittered inside each cell)."""
    rows, cols = np.nonzero(mask)
    pick = rng.choice(len(rows), size=n, replace=True)
    xs, ys = transform_xy(transform, rows[pick], cols[pick], offset="center")
    xs = np.asarray(xs) + (rng.random(n) - 0.5) * transform.a
    ys = np.asarray(ys) + (rng.random(n) - 0.5) * transform.e
    return gpd.GeoSeries(gpd.points_from_xy(xs, ys), crs=crs)


def sample_grids(points: gpd.GeoSeries, grids: dict[str, Path]) -> pd.DataFrame:
    """Sample every predictor raster at the given points (reprojected per raster).

    Numeric grids give one float column; categorical grids ('cat_' stems) give
    codes to be one-hot encoded downstream. Nodata becomes NaN.
    """
    out: dict[str, np.ndarray] = {}
    for name, path in grids.items():
        with rasterio.open(path) as src:
            pts = points.to_crs(src.crs)
            coords = [(geom.x, geom.y) for geom in pts]
            values = np.array([v[0] for v in src.sample(coords)], dtype=np.float64)
            if src.nodata is not None:
                values[values == src.nodata] = np.nan
        out[name] = values
    return pd.DataFrame(out, index=points.index)


def encode_features(sampled: pd.DataFrame) -> pd.DataFrame:
    """Numeric columns as float32; categorical ('cat_') columns one-hot encoded."""
    parts: list[pd.DataFrame] = []
    for name in sampled.columns:
        if is_categorical(name):
            codes = sampled[name].map(
                lambda v: "__MISSING__" if pd.isna(v) else str(int(v))
            )
            onehot = pd.get_dummies(codes, prefix=f"{name}_", prefix_sep="", dtype=np.float32)
            parts.append(onehot)
        else:
            parts.append(sampled[[name]].astype(np.float32))
    return pd.concat(parts, axis=1)


# ============================================================
# Sampling — presences vs generated spatial background
# ============================================================
#
# Susceptibility is a purely spatial contrast: where the process occurred vs
# random locations in the valid domain that host no known event (kept at least
# event_buffer_m away, fully seeded). The map, and every score, is relative —
# as susceptibility always is.


def load_points(config: ProjectConfig, key: str) -> gpd.GeoDataFrame:
    path = config.data_dir / f"points_{key}.geojson"
    points = gpd.read_file(path)
    if not (points.geometry.geom_type == "Point").all():
        raise ValueError(f"{path}: expected Point geometries only")
    if "date" in points.columns:
        points["date"] = pd.to_datetime(points["date"])
    return points


def sample_absences(
    presences: gpd.GeoSeries,
    n_absences: int,
    *,
    domain_mask: np.ndarray,
    transform,
    crs,
    buffer_m: float,
    rng: np.random.Generator,
    max_tries: int = 50,
) -> gpd.GeoSeries:
    """Random absence locations in the domain, >= buffer_m from any presence.

    Non-events are generated, not observed: random locations in the valid
    domain (reference grid minus exclusion mask), kept away from every known
    event, fully seeded.
    """
    presence_xy = np.column_stack([presences.to_crs(crs).x, presences.to_crs(crs).y])
    tree = cKDTree(presence_xy)
    kept: list = []
    for _ in range(max_tries):
        need = n_absences - len(kept)
        if need <= 0:
            break
        candidates = random_points_in_domain(
            int(np.ceil(need * 1.5)), domain_mask, transform, crs, rng
        )
        distances, _ = tree.query(np.column_stack([candidates.x, candidates.y]))
        kept.extend(candidates[distances >= buffer_m][:need])
    if len(kept) < n_absences:
        raise ValueError(
            f"Could only place {len(kept)}/{n_absences} absences with a "
            f"{buffer_m} m event buffer; domain too small or buffer too large"
        )
    return gpd.GeoSeries(kept, crs=crs)


def assign_absence_dates(
    absence_geometry: gpd.GeoSeries,
    presence_geometry: gpd.GeoSeries,
    presence_dates: pd.Series,
    *,
    buffer_days: int,
    buffer_radius_m: float,
    rng: np.random.Generator,
    max_tries: int = 200,
) -> np.ndarray:
    """Assign each background point a pseudo-date for antecedent-condition
    sampling (see build_antecedent_features()).

    Dates are resampled with replacement from the real event dates -- this
    preserves the actual seasonal/yearly distribution of events with no
    extra modeling assumptions, rather than e.g. uniform-random dates that
    would let the model partly learn "season" as a shortcut. A candidate is
    rejected and redrawn if any real event within buffer_radius_m falls
    within buffer_days of it: a regional storm can trigger one landslide
    while a few km away, at the same time, conditions look identical but no
    slide was recorded there -- without this buffer, that location would be
    mislabeled a clean negative under genuinely landslide-conducive conditions.
    """
    event_dates = presence_dates.to_numpy()
    event_xy = np.column_stack([presence_geometry.x, presence_geometry.y])
    tree = cKDTree(event_xy)
    absence_xy = np.column_stack([absence_geometry.x, absence_geometry.y])
    neighbor_lists = tree.query_ball_point(absence_xy, r=buffer_radius_m)
    buffer_td = np.timedelta64(buffer_days, "D")

    assigned = np.empty(len(absence_geometry), dtype="datetime64[ns]")
    for i, neighbors in enumerate(neighbor_lists):
        if not neighbors:
            assigned[i] = rng.choice(event_dates)
            continue
        nearby_dates = event_dates[neighbors]
        for _ in range(max_tries):
            candidate = rng.choice(event_dates)
            if np.all(np.abs(nearby_dates - candidate) >= buffer_td):
                assigned[i] = candidate
                break
        else:
            raise ValueError(
                f"Could not find a conflict-free pseudo-date for background point "
                f"{i} after {max_tries} tries ({len(neighbors)} real events within "
                f"{buffer_radius_m}m; buffer_days={buffer_days}). Consider a smaller "
                "antecedent_buffer_radius_m/antecedent_buffer_days, or fewer "
                "background points relative to event density."
            )
    return assigned


def _finalize(
    geometry: gpd.GeoSeries,
    y: np.ndarray,
    grids: dict[str, Path],
    dates: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Sample the grids, drop nodata rows, encode -> (features, labels, metadata).

    `dates` (if given, one per row, aligned with `geometry`/`y` before
    filtering) is carried into metadata's "date" column for
    build_antecedent_features() -- real event dates for presences, assigned
    pseudo-dates for background (see assign_absence_dates())."""
    sampled = sample_grids(geometry, grids)
    valid = ~sampled.isna().any(axis=1).to_numpy()
    sampled = sampled[valid]
    geometry = geometry[valid].reset_index(drop=True)
    y = y[valid]
    if dates is not None:
        dates = dates[valid]
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("Sample degenerated to a single class after nodata filtering")
    encoded = encode_features(sampled)
    metadata = pd.DataFrame({
        "x": geometry.x.to_numpy(),
        "y": geometry.y.to_numpy(),
        "label": y.astype(int),
    })
    if dates is not None:
        metadata["date"] = dates
    return encoded, y.astype(np.int8), metadata


def build_sample(
    points: gpd.GeoDataFrame,
    config: ProjectConfig,
    *,
    domain_mask: np.ndarray,
    transform,
    crs,
    grids: dict[str, Path],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Presences vs generated spatial background, sampled on every grid."""
    rng = np.random.default_rng(config.random_state)
    presences = points.geometry.to_crs(crs).reset_index(drop=True)
    background = sample_absences(
        presences, config.background_ratio * len(presences),
        domain_mask=domain_mask, transform=transform, crs=crs,
        buffer_m=config.event_buffer_m, rng=rng,
    )
    geometry = gpd.GeoSeries(list(presences) + list(background), crs=crs)
    y = np.concatenate([
        np.ones(len(presences), dtype=np.int8),
        np.zeros(len(background), dtype=np.int8),
    ])
    dates = None
    if "date" in points.columns:
        presence_dates = pd.to_datetime(points["date"]).reset_index(drop=True)
        background_dates = assign_absence_dates(
            background, presences, presence_dates,
            buffer_days=config.antecedent_buffer_days,
            buffer_radius_m=config.antecedent_buffer_radius_m,
            rng=rng,
        )
        dates = np.concatenate([presence_dates.to_numpy(), background_dates])
    return _finalize(geometry, y, grids, dates=dates)


def prepare_one(config: ProjectConfig, key: str, out: Path) -> dict:
    grids = discover_grids(config.grids_dir)
    domain_mask, transform, crs = valid_domain(grids, config.grids_dir)
    points = load_points(config, key)
    encoded, y, metadata = build_sample(
        points, config,
        domain_mask=domain_mask, transform=transform, crs=crs, grids=grids,
    )
    antecedent = build_antecedent_features(config, metadata, crs)
    for name, column in antecedent.items():
        encoded[name] = column
    schema = {
        "process": key,
        "grids": sorted(grids),
        "dynamic_vars": sorted(discover_dynamic(config.data_dir)),
        "crs": str(crs),
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "background_ratio": config.background_ratio,
        "event_buffer_m": config.event_buffer_m,
        "antecedent_windows_days": list(config.antecedent_windows_days),
        "antecedent_buffer_days": config.antecedent_buffer_days,
        "antecedent_buffer_radius_m": config.antecedent_buffer_radius_m,
        "antecedent_features": sorted(antecedent),
        "k_folds": config.k_folds,
    }
    out_dir = out / key
    save_artifact(
        out_dir,
        X=encoded.to_numpy(dtype=np.float32), y=y, metadata=metadata,
        feature_names=encoded.columns.tolist(), schema=schema,
    )
    return {
        "process": key,
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "n_features": int(encoded.shape[1]),
        "out_dir": str(out_dir),
    }


# ============================================================
# Derived static grids — temporal archives reduced to per-cell aggregates
# ============================================================
#
# The model is static, so a time-varying archive can enter it only as a static
# per-cell summary. derived_grid() performs that reduction — one streaming pass
# over data/dynamic/<var>/*.nc — and caches the result as a GeoTIFF that is
# then sampled exactly like any grid in data/grids/, for training and for the
# map alike (one code path, no train/serve skew). This declarative mechanism
# still exists for simple aggregates; features.py (below) generalizes it to
# arbitrary raster-generation code.


def discover_dynamic(data_dir: Path) -> dict[str, Path]:
    """Dynamic variables by name: one directory per variable under data/dynamic/.

    The directory name IS the variable name (mirroring grids/<name>.tif). A
    directory with no NetCDF is ignored. Empty result = no dynamic layer.
    """
    root = data_dir / DYNAMIC_DIRNAME
    if not root.is_dir():
        return {}
    return {
        sub.name: sub
        for sub in sorted(root.iterdir())
        if sub.is_dir() and any(sub.glob("*.nc"))
    }


def _grid_axes(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    """Ascending (x, y) coordinate vectors of a dynamic NetCDF grid."""
    x = np.asarray(dataset["x"].values, dtype=np.float64)
    y = np.asarray(dataset["y"].values, dtype=np.float64)
    return x, y


def _resolve_var(dataset: Any, var: str) -> str:
    """The data-variable to read for `var`, tolerating inconsistent file naming.

    Prefer an exact match on the directory name; otherwise fall back to the sole
    time-varying data variable (dims include DATE), so an archive whose files
    disagree on the internal name — a full name vs. an abbreviation — still
    reads cleanly.
    Grid-mapping variables (no DATE dimension) are ignored.
    """
    if var in dataset.data_vars:
        return var
    timed = [name for name, arr in dataset.data_vars.items() if "DATE" in arr.dims]
    if len(timed) == 1:
        return timed[0]
    raise KeyError(
        f"cannot resolve variable '{var}' in a NetCDF: data variables with a DATE "
        f"dimension are {timed}. Rename the variable to '{var}' or leave "
        "exactly one time-varying variable per file."
    )


def build_antecedent_features(
    config: ProjectConfig, metadata: pd.DataFrame, crs: str,
) -> dict[str, np.ndarray]:
    """Sum and mean of every dynamic variable over each configured antecedent
    window (config.antecedent_windows_days), ending on and including each
    observation's own date -- real event dates for presences, assigned
    pseudo-dates for background (see assign_absence_dates()). Window w means
    "the w most recent days up to and including this observation's date", so
    w=1 is exactly the day itself.

    One streaming pass per variable over its whole archive -- the same cost
    profile as warm_derived_cache(), NOT per-observation -- accumulating each
    observation's per-window sum/count as the scan passes each file's dates.
    Returns {} if there is no dynamic archive or no "date" column (e.g. a
    hazard whose points file carries no date).
    """
    variables = discover_dynamic(config.data_dir)
    if not variables or "date" not in metadata.columns:
        return {}
    import xarray as xr

    n = len(metadata)
    # Normalize to calendar-day resolution: dynamic archives commonly timestamp
    # a daily value at noon (observed here: 12:00:00, vs. midnight event dates),
    # so exact-instant comparison would silently drop the reference day itself
    # from every window (all of it for a 1-day window). Comparing whole days
    # makes the window well-defined regardless of either side's time-of-day
    # convention.
    ref_dates = pd.to_datetime(metadata["date"]).to_numpy().astype("datetime64[D]")
    windows = sorted(set(int(w) for w in config.antecedent_windows_days))
    points = gpd.GeoSeries(gpd.points_from_xy(metadata["x"], metadata["y"]), crs=crs)

    out: dict[str, np.ndarray] = {}
    for var in sorted(variables):
        sums = {w: np.zeros(n, dtype=np.float64) for w in windows}
        counts = {w: np.zeros(n, dtype=np.int64) for w in windows}
        col_idx = row_idx = None
        for path in sorted(variables[var].glob("*.nc")):
            with xr.open_dataset(path) as dataset:
                if col_idx is None:
                    # Dynamic archives are assumed already in the reference grid's CRS
                    # (same convention as derived_grid()/_write_derived_raster() --
                    # no reprojection is done for this layer, just a different extent).
                    grid_x, grid_y = _grid_axes(dataset)
                    col_idx = np.argmin(np.abs(grid_x[:, None] - points.x.to_numpy()[None, :]), axis=0)
                    row_idx = np.argmin(np.abs(grid_y[:, None] - points.y.to_numpy()[None, :]), axis=0)
                block = (
                    dataset[_resolve_var(dataset, var)]
                    .transpose("DATE", "y", "x")
                    .to_numpy()
                    .astype(np.float64)
                )
                file_dates = pd.to_datetime(dataset["DATE"].values).to_numpy().astype("datetime64[D]")
            values = block[:, row_idx, col_idx]  # (T_file, n): this file's value at every observation's cell
            finite = np.isfinite(values)
            for w in windows:
                window_start = ref_dates - np.timedelta64(w - 1, "D")
                in_window = (
                    (file_dates[:, None] >= window_start[None, :])
                    & (file_dates[:, None] <= ref_dates[None, :])
                    & finite
                )
                sums[w] += np.where(in_window, values, 0.0).sum(axis=0)
                counts[w] += in_window.sum(axis=0)
        for w in windows:
            count = counts[w]
            has_data = count > 0
            out[f"{var}_sum_{w}d"] = np.where(has_data, sums[w], np.nan).astype(np.float32)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"{var}_mean_{w}d"] = np.where(has_data, sums[w] / count, np.nan).astype(np.float32)
    return out


def derived_name(var: str, stat: str, months: list[int] | None = None) -> str:
    """Feature name of a derived aggregate; doubles as its raster's file stem."""
    suffix = "" if not months else "_m" + "-".join(str(int(m)) for m in sorted(set(months)))
    return f"{var}_{stat}{suffix}"


def derived_grid(
    config: ProjectConfig,
    var: str,
    stat: str,
    months: list[int] | None = None,
) -> Path:
    """Reduce a time-varying archive to one static per-cell aggregate GeoTIFF.

    A single streaming pass over data/dynamic/<var>/*.nc computes, per cell,
    the `stat` ("mean", "min", "max" or "std") of the variable over every
    timestep in the archive, optionally restricted to the given calendar
    `months`. The raster is cached at artifacts/derived/<name>.tif and reused
    on later calls; delete the file to recompute after the archive changes.
    It lives on the archive's own grid (reference-grid CRS, north-up) and is
    sampled exactly like any static predictor grid.

    The build itself runs in an isolated subprocess under a wall-clock timeout
    (config.feature_build_timeout_s, shared with build_feature_grid()) killed
    by process GROUP on timeout — mirrors build_feature_grid()'s protection,
    since a full archive scan (hundreds of NetCDF files) on a busy network
    filesystem can otherwise hang the whole training run indefinitely.

    A cache miss triggers ONE subprocess that computes all of DERIVED_STATS
    for this (var, months) — not just the requested stat — since the archive
    scan itself (not the per-cell reduction) is the expensive part and is
    identical either way. See warm_derived_cache() for pre-building these
    outside any agent tool-call timeout.
    """
    if stat not in DERIVED_STATS:
        raise ValueError(f"stat must be one of {DERIVED_STATS}, got '{stat}'")
    if months and not all(1 <= int(m) <= 12 for m in months):
        raise ValueError(f"months must be calendar months 1-12, got {months}")
    variables = discover_dynamic(config.data_dir)
    if var not in variables:
        raise KeyError(
            f"No dynamic archive '{var}' under {config.data_dir / DYNAMIC_DIRNAME}. "
            f"Available: {sorted(variables)}"
        )
    name = derived_name(var, stat, months)
    out_path = REPO_ROOT / DERIVED_DIR / f"{name}.tif"
    if out_path.exists():
        return out_path

    # Build ALL of DERIVED_STATS for this (var, months) in one subprocess: the
    # archive scan (hundreds of NetCDF files) is the expensive part and is
    # identical regardless of which single stat was asked for, so computing
    # only one and discarding the other three mid-scan wastes 3 out of every
    # 4 full scans. Whichever of the 4 aren't cached yet get (re)built; any
    # that are already cached are simply left alone.
    scratch_dir = REPO_ROOT / DERIVED_SCRATCH_DIR / name
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(REPO_ROOT / "prepare.py"), "--run-derived-all",
           "--var", var, "--derived-out-dir", str(scratch_dir)]
    if months:
        cmd += ["--months", ",".join(str(int(m)) for m in sorted(set(months)))]
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, start_new_session=True)
    try:
        ret = proc.wait(timeout=config.feature_build_timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
        raise SystemExit(
            f"derived '{name}' exceeded feature_build_timeout_s="
            f"{config.feature_build_timeout_s}s and was killed (process group)"
        )
    if ret != 0:
        raise SystemExit(f"derived '{name}' build failed (exit {ret}); see output above")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for other_stat in DERIVED_STATS:
        other_name = derived_name(var, other_stat, months)
        other_out = REPO_ROOT / DERIVED_DIR / f"{other_name}.tif"
        scratch_file = scratch_dir / f"{other_name}.tif"
        if not other_out.exists() and scratch_file.exists():
            scratch_file.rename(other_out)
    shutil.rmtree(scratch_dir, ignore_errors=True)

    if not out_path.exists():
        raise SystemExit(f"derived '{name}' build reported success but wrote no output")
    return out_path


def _scan_dynamic_archive(config: ProjectConfig, var: str, months: list[int] | None):
    """One streaming pass over data/dynamic/<var>/*.nc, returning the raw
    per-cell accumulators (count/total/sumsq/min/max) plus the archive's own
    grid axes. This is the expensive part of a DERIVED build — shared by
    run_derived() and run_derived_all_stats() so requesting multiple stats
    for the same (var, months) never costs more than one archive scan."""
    import xarray as xr

    variables = discover_dynamic(config.data_dir)
    wanted = sorted(set(int(m) for m in months)) if months else None
    count = total = sumsq = low = high = None
    grid_x = grid_y = None
    for path in sorted(variables[var].glob("*.nc")):
        with xr.open_dataset(path) as dataset:
            if wanted is not None:
                file_months = pd.to_datetime(dataset["DATE"].values).month
                pick = np.flatnonzero(np.isin(file_months, wanted))
                if pick.size == 0:
                    continue
            block = (
                dataset[_resolve_var(dataset, var)]
                .transpose("DATE", "y", "x")
                .to_numpy()
                .astype(np.float64)
            )
            if wanted is not None:
                block = block[pick]
            if grid_x is None:
                grid_x, grid_y = _grid_axes(dataset)
        finite = np.isfinite(block)
        filled = np.where(finite, block, 0.0)
        if count is None:
            shape = block.shape[1:]
            count = np.zeros(shape, dtype=np.int64)
            total = np.zeros(shape, dtype=np.float64)
            sumsq = np.zeros(shape, dtype=np.float64)
            low = np.full(shape, np.inf)
            high = np.full(shape, -np.inf)
        count += finite.sum(axis=0)
        total += filled.sum(axis=0)
        sumsq += (filled * filled).sum(axis=0)
        low = np.minimum(low, np.where(finite, block, np.inf).min(axis=0))
        high = np.maximum(high, np.where(finite, block, -np.inf).max(axis=0))
    if count is None or not count.any():
        raise SystemExit(f"Archive '{var}' holds no timesteps for the requested months")
    return count, total, sumsq, low, high, grid_x, grid_y


def _stat_from_accumulators(stat: str, count, total, sumsq, low, high):
    """Reduce the shared accumulators from _scan_dynamic_archive() to one
    per-cell statistic array."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        if stat == "mean":
            result = mean
        elif stat == "std":
            result = np.sqrt(np.maximum(sumsq / count - mean * mean, 0.0))
        elif stat == "min":
            result = low
        else:
            result = high
    return np.where(count > 0, result, np.nan).astype(np.float32)


def _write_derived_raster(config: ProjectConfig, grid_x, grid_y, result, out_path: Path) -> None:
    """Write one per-cell statistic array to out_path as a north-up GeoTIFF
    on the archive's own grid (reference-grid CRS)."""
    # Write north-up: flip when the archive's y axis ascends.
    if grid_y[0] < grid_y[-1]:
        result = result[::-1]
        grid_y = grid_y[::-1]
    res_x = float(np.mean(np.diff(grid_x)))
    row_step = float(np.mean(np.diff(grid_y)))  # negative once north-up
    transform = rasterio.transform.from_origin(
        float(grid_x[0]) - res_x / 2, float(grid_y[0]) - row_step / 2, res_x, -row_step
    )
    with rasterio.open(reference_grid(discover_grids(config.grids_dir))) as ref:
        crs = ref.crs
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "height": result.shape[0], "width": result.shape[1],
        "count": 1, "dtype": "float32", "crs": crs, "transform": transform,
        "nodata": np.nan, "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(result, 1)


def run_derived(
    config: ProjectConfig,
    var: str,
    stat: str,
    months: list[int] | None,
    out_path: Path,
) -> None:
    """Compute one derived_grid() aggregate and write it to out_path. Kept for
    direct single-stat CLI use; derived_grid() itself calls
    run_derived_all_stats() instead so a cache miss never wastes 3 out of 4
    possible archive scans."""
    count, total, sumsq, low, high, grid_x, grid_y = _scan_dynamic_archive(config, var, months)
    result = _stat_from_accumulators(stat, count, total, sumsq, low, high)
    _write_derived_raster(config, grid_x, grid_y, result, out_path)
    log.info("Derived %s -> %s", derived_name(var, stat, months), out_path)


def run_derived_all_stats(
    config: ProjectConfig,
    var: str,
    months: list[int] | None,
    out_dir: Path,
) -> None:
    """Compute every DERIVED_STATS entry for one (var, months) in a SINGLE
    archive scan, writing each to out_dir as <derived_name(var, stat, months)>.tif.
    Called as a fresh `python prepare.py --run-derived-all ...` subprocess by
    derived_grid() so a hung scan can be killed without taking down the
    caller, and so requesting any one stat for a never-built (var, months)
    populates the cache for all four instead of just one."""
    count, total, sumsq, low, high, grid_x, grid_y = _scan_dynamic_archive(config, var, months)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stat in DERIVED_STATS:
        result = _stat_from_accumulators(stat, count, total, sumsq, low, high)
        name = derived_name(var, stat, months)
        out_path = out_dir / f"{name}.tif"
        _write_derived_raster(config, grid_x, grid_y, result, out_path)
        log.info("Derived %s -> %s", name, out_path)


def warm_derived_cache(config: ProjectConfig) -> None:
    """Pre-build every discovered dynamic variable's full-archive DERIVED
    aggregates (mean/min/max/std, no month filter) before an agent session
    starts. Meant to be called from loop.sh in its own plain shell process —
    NOT from inside an agent's own tool-calling harness — so the first,
    potentially multi-minute build of each aggregate is never at risk of an
    agent-side per-command timeout. Safe and cheap to re-run: derived_grid()
    returns instantly for anything already cached (prepare.py:775). Only the
    no-month-filter form is warmed here — `months` is combinatorially
    unbounded (any subset of 1-12), so month-filtered DERIVED entries are
    still built on demand by the agent and still carry the timeout exposure
    documented in program.md's Run step.
    """
    variables = discover_dynamic(config.data_dir)
    if not variables:
        print("No dynamic archives under data/dynamic/ -- nothing to warm.")
        return
    for var in sorted(variables):
        for stat in DERIVED_STATS:
            name = derived_name(var, stat)
            t0 = time.time()
            path = derived_grid(config, var, stat)
            dt = time.time() - t0
            print(f"[warm-derived-cache] {name}: {'built' if dt > 0.5 else 'cached'} "
                  f"({dt:.1f}s) -> {path}")


def describe_dynamic(config: ProjectConfig) -> str:
    """Human-readable inventory of the dynamic layer: variables, units, coverage,
    resolution, value range — read straight from the NetCDF metadata, so it says
    what the data IS and never what to do with it."""
    import xarray as xr

    variables = discover_dynamic(config.data_dir)
    if not variables:
        return "No dynamic variables (data/dynamic/<var>/*.nc absent)."
    lines = [
        f"Time-varying archives under {config.data_dir / DYNAMIC_DIRNAME}:",
        "",
    ]
    for var, var_dir in variables.items():
        files = sorted(var_dir.glob("*.nc"))
        with xr.open_dataset(files[0]) as first, xr.open_dataset(files[-1]) as last:
            grid_x, grid_y = _grid_axes(first)
            res_x = float(np.abs(np.diff(grid_x)).mean())
            res_y = float(np.abs(np.diff(grid_y)).mean())
            name = _resolve_var(first, var)
            units = first[name].attrs.get("units", "unknown")
            long_name = first[name].attrs.get("long_name", "")
            sample = first[name].isel(DATE=0).to_numpy()
            start = pd.to_datetime(first["DATE"].values).min().date()
            end = pd.to_datetime(last["DATE"].values).max().date()
        finite = sample[np.isfinite(sample)]
        rng = (f"{finite.min():.3g} … {finite.max():.3g}" if finite.size else "all-NaN sample")
        lines += [
            f"  {var}"
            + (f"  ({long_name})" if long_name else ""),
            f"    units      : {units}",
            f"    coverage   : {start} … {end}  ({len(files)} files)",
            f"    grid       : {len(grid_x)}×{len(grid_y)} @ ~{res_x:.0f}×{res_y:.0f} map units",
            f"    value range: {rng}  (first date, valid cells)",
            "",
        ]
    lines += [
        "These archives enter the model as static per-cell aggregates (DERIVED, a fixed",
        'declarative form: {"var": <name above>, "stat": "mean"|"min"|"max"|"std",',
        '"months": [1-12] (optional)}), or as raw input to a features.py recipe of',
        "the agent's own design (see --describe-primitives). Either way the resulting",
        "feature joins the same FEATURES include/exclude space as the static grids.",
    ]
    return "\n".join(lines)


# ============================================================
# Class-based exclusion mask — non-event domain restriction
# ============================================================
#
# See module docstring point 6. This framework assumes nothing about WHICH
# categorical raster is relevant -- land cover, geology, demographic zoning,
# soil type, or anything else -- or what its codes mean. The agent discovers
# that itself from --describe-primitives's inventory (file names, CRS, a
# peek) plus its own general knowledge of common classification nomenclatures
# (e.g. Corine's well-known land-cover codes), judges it ONCE per hazard
# branch during setup (program.md), and records both the chosen raster and
# the codes to exclude in config.yml.
# Never revisited by the per-experiment loop afterward.


def _categorical_raster_on_reference(ref_path: Path, src_path: Path) -> np.ndarray:
    """Reproject/resample (nearest — categorical, never averaged) one
    categorical raster onto the reference grid. Source CRS/resolution may
    differ from the reference grid's; this is the one place that
    reprojection happens for this layer, and it is fixed infrastructure, not
    a features.py recipe."""
    if not src_path.exists():
        raise FileNotFoundError(f"{src_path} not found")
    with rasterio.open(ref_path) as ref:
        ref_crs, ref_transform = ref.crs, ref.transform
        ref_height, ref_width = ref.height, ref.width
    with rasterio.open(src_path) as src:
        dest = np.zeros((ref_height, ref_width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1), destination=dest,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_transform, dst_crs=ref_crs,
            resampling=Resampling.nearest,
            src_nodata=src.nodata, dst_nodata=0,
        )
    return dest


def describe_raster_classes(config: ProjectConfig, source: str | None = None) -> str:
    """Class codes present on the reference grid for one categorical raster,
    with pixel counts, so the agent can judge exclusions. No label table is
    bundled here: pick `source` from --describe-primitives's inventory by
    name/content, and rely on your own knowledge of that classification
    scheme's codes (e.g. Corine's well-known level-3 nomenclature) — this
    framework doesn't assume which product, if any, is in use."""
    raster_name = source or config.exclude_raster
    if not raster_name:
        raise SystemExit(
            "no raster given: pass source=... or set config.yml's sampling.exclude_raster "
            "to a categorical raster's path (see --describe-primitives for what's available)"
        )
    grids = discover_grids(config.grids_dir)
    ref_path = reference_grid(grids)
    src_path = _resolve(config.data_dir, raster_name)
    band = _categorical_raster_on_reference(ref_path, src_path)
    values, counts = np.unique(band, return_counts=True)
    total = int(band.size)
    lines = [f"Class codes present in {raster_name} (reprojected onto the reference grid):", ""]
    for value, count in sorted(zip(values, counts), key=lambda vc: -vc[1]):
        code = int(value)
        if code == 0:
            continue
        lines.append(f"  {code:<6d} {int(count):>8d} cells ({100*count/total:.1f}%)")
    lines += [
        "",
        "Judge which classes are physically implausible non-event locations for",
        "THIS branch's specific hazard (see program.md's Setup section), write the",
        f"raster path ({raster_name}) to config.yml's sampling.exclude_raster and the",
        "chosen codes to sampling.exclude_classes, then run",
        "  python prepare.py --build-exclude-mask",
        "This is done ONCE per hazard branch; config.yml is frozen afterward.",
    ]
    return "\n".join(lines)


def build_exclude_mask(config: ProjectConfig) -> Path:
    """Turn config.yml's sampling.exclude_raster + exclude_classes
    into data/grids/mask_exclude.tif, picked up automatically by
    valid_domain(). Run once, during branch setup."""
    if not config.exclude_raster or not config.exclude_classes:
        raise SystemExit(
            "config.yml's sampling.exclude_raster and/or exclude_classes is "
            "empty — judge and set both first (run --describe-raster-classes, then see "
            "program.md's Setup section)"
        )
    grids = discover_grids(config.grids_dir)
    ref_path = reference_grid(grids)
    src_path = _resolve(config.data_dir, config.exclude_raster)
    band = _categorical_raster_on_reference(ref_path, src_path)
    exclude = np.isin(band, sorted(config.exclude_classes)).astype(np.uint8)
    with rasterio.open(ref_path) as ref:
        profile = {
            "driver": "GTiff", "height": ref.height, "width": ref.width, "count": 1,
            "dtype": "uint8", "crs": ref.crs, "transform": ref.transform,
            "nodata": 0, "compress": "deflate",
        }
    out_path = config.grids_dir / EXCLUDE_MASK_NAME
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(exclude, 1)
    print(
        f"Wrote {out_path}: {int(exclude.sum())}/{exclude.size} cells excluded "
        f"(classes {sorted(config.exclude_classes)})"
    )
    return out_path


# ============================================================
# Agent-authored feature recipes (features.py)
# ============================================================
#
# See module docstring point 5. This generalizes DERIVED (a fixed declarative
# aggregate spec) into a surface where the agent writes actual raster-generation
# code. Every guardrail below exists because features.py is otherwise arbitrary
# Python with subprocess access, and ALL of them run automatically regardless
# of what the agent does -- none is something the agent has to remember to
# invoke itself. They're split across the parent/child subprocess boundary
# only because of WHERE each check's information is available: run_recipe()
# (the child -- see build_feature_grid()'s docstring for why it's isolated)
# can see the leakage risk in features.py's own source and the raster it just
# produced; build_feature_grid() (the parent) is the only one that knows the
# CURRENT total cache size. Together they are the "security checks":
#   1. scan_features_module()      -- leakage guard, before import (child)
#   2. validate_feature_raster()   -- alignment, no silent reprojection (child)
#   3. enforce_feature_size_cap()  -- one raster vs 2x dem.tif (child)
#   4. enforce_cache_budget()      -- whole cache vs 15GB (parent)


@dataclass(frozen=True)
class FeatureContext:
    """What a features.py recipe is given. Deliberately narrow: no points path
    (see scan_features_module) — a recipe builds a predictor from raw primitives,
    never from the events/labels it will be scored against. Plain data, no
    methods: to shell out to an external tool (gdaldem, saga_cmd, ...), call
    `subprocess.run(cmd, timeout=N, cwd=ctx.scratch_dir, check=True)` directly
    — see program.md's "Paths available to features.py recipes" reference."""
    dem_path: Path
    dynamic_dir: Path
    vectors_dir: Path
    scratch_dir: Path             # wiped before every recipe run; throwaway intermediates only
    reference_profile: dict[str, Any]  # crs, transform, height, width — what the output MUST match


def features_source_hash(config: ProjectConfig, features_path: Path | None = None) -> str:
    """Cache key: features.py's own full source text plus a coarse fingerprint of
    the raw data it can read (dem.tif, data/vectors/*.gpkg, the dynamic archive
    file listing). Whole-file, not per-recipe: a shared helper used by one recipe
    wouldn't be captured by a narrower hash, and a stale cache HIT is worse than
    over-invalidating everything. Any features.py edit — or a data refresh —
    invalidates every cached name. This is the fix for the fact that derived_grid()
    above caches by output filename alone, with no invalidation on code change."""
    features_path = features_path or (REPO_ROOT / FEATURES_MODULE_NAME)
    parts = [features_path.read_text()]
    dem = reference_grid(discover_grids(config.grids_dir))
    st = dem.stat()
    parts.append(f"dem:{st.st_size}:{st.st_mtime_ns}")
    vectors_dir = config.data_dir / "vectors"
    if vectors_dir.is_dir():
        for p in sorted(vectors_dir.iterdir()):
            st = p.stat()
            parts.append(f"vec:{p.name}:{st.st_size}:{st.st_mtime_ns}")
    dynamic_dir = config.data_dir / DYNAMIC_DIRNAME
    if dynamic_dir.is_dir():
        names = sorted(str(p.relative_to(dynamic_dir)) for p in dynamic_dir.glob("*/*.nc"))
        parts.append("dyn:" + ",".join(names))
    payload = "\n".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def scan_features_module(features_path: Path | None = None) -> None:
    """Security check 1/4 (child-side): leakage guard, layer 1-2 of 3 (see
    program.md for layer 3, the Hard Rule). Refuses to execute features.py if
    it imports prepare's sampling/label machinery, or literally references the
    points/labels files. This is a heuristic, not a proof: it blocks the
    obvious paths, not every conceivable one (see program.md)."""
    features_path = features_path or (REPO_ROOT / FEATURES_MODULE_NAME)
    source = features_path.read_text()
    tree = ast.parse(source, filename=str(features_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "prepare":
            names = {alias.name for alias in node.names}
            bad = names & FEATURES_FORBIDDEN_IMPORTS
            if bad:
                raise SystemExit(
                    f"features.py imports forbidden prepare symbols {sorted(bad)} — "
                    "features.py must never see labels/points; see program.md's leakage Hard Rule"
                )
    # Scan code only, not the module docstring — program.md's own wording (and
    # this file's docstring) legitimately mentions "points_"/".geojson" when
    # documenting the rule; only literal usage in code should trip it.
    docstring = ast.get_docstring(tree)
    code_text = source.replace(docstring, "", 1) if docstring else source
    for substr in FEATURES_FORBIDDEN_SUBSTRINGS:
        if substr in code_text:
            raise SystemExit(
                f"features.py contains the forbidden substring '{substr}' — it must never "
                "reference the points/labels files directly; see program.md's leakage Hard Rule"
            )


def validate_feature_raster(name: str, path: Path, reference_profile: dict[str, Any]) -> None:
    """Security check 2/4 (child-side): a recipe that returns a Path (rather
    than an array prepare.py writes itself) must land EXACTLY on the reference
    grid. No silent reprojection, ever — a mismatch is a crash, forcing the
    agent to fix features.py."""
    with rasterio.open(path) as src:
        if src.count != 1:
            raise SystemExit(f"feature '{name}': raster has {src.count} bands, must be single-band")
        src_epsg = src.crs.to_epsg() if src.crs else None
        if src_epsg != reference_profile["epsg"]:
            raise SystemExit(
                f"feature '{name}': CRS EPSG:{src_epsg} != reference EPSG:{reference_profile['epsg']} "
                "— the recipe must write directly onto the reference grid, no reprojection is done for it"
            )
        ref_t = reference_profile["transform"]
        if any(abs(a - b) > 1e-6 for a, b in zip(tuple(src.transform), tuple(ref_t))):
            raise SystemExit(f"feature '{name}': transform does not match the reference grid within 1e-6")
        if (src.height, src.width) != (reference_profile["height"], reference_profile["width"]):
            raise SystemExit(
                f"feature '{name}': shape {(src.height, src.width)} != reference "
                f"{(reference_profile['height'], reference_profile['width'])}"
            )


def _reference_profile(config: ProjectConfig) -> tuple[Path, dict[str, Any]]:
    ref = reference_grid(discover_grids(config.grids_dir))
    with rasterio.open(ref) as src:
        profile = {
            "epsg": src.crs.to_epsg(), "transform": src.transform,
            "crs": src.crs, "height": src.height, "width": src.width,
        }
    return ref, profile


def run_recipe(config: ProjectConfig, name: str, out_path: Path) -> None:
    """Execute exactly one features.py recipe and write its validated raster to
    out_path. Called as a fresh `python prepare.py --run-recipe ...` subprocess
    by build_feature_grid() — never in-process — so a hang can be killed by
    process group without taking down the caller, and a crash never corrupts
    caller state."""
    features_path = REPO_ROOT / FEATURES_MODULE_NAME
    scan_features_module(features_path)
    spec = importlib.util.spec_from_file_location("features", features_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    recipes = {r["name"]: r["build"] for r in module.RECIPES}
    if name not in recipes:
        raise SystemExit(f"No recipe named '{name}' in features.py RECIPES: {sorted(recipes)}")

    ref, reference_profile = _reference_profile(config)
    scratch_dir = REPO_ROOT / FEATURES_SCRATCH_DIR / name
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    ctx = FeatureContext(
        dem_path=ref,
        dynamic_dir=config.data_dir / DYNAMIC_DIRNAME,
        vectors_dir=config.data_dir / "vectors",
        scratch_dir=scratch_dir,
        reference_profile=reference_profile,
    )
    result = recipes[name](ctx)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(result, np.ndarray):
        expected_shape = (reference_profile["height"], reference_profile["width"])
        if result.shape != expected_shape:
            raise SystemExit(f"feature '{name}': array shape {result.shape} != reference {expected_shape}")
        profile = {
            "driver": "GTiff", "height": reference_profile["height"], "width": reference_profile["width"],
            "count": 1, "dtype": "float32", "crs": reference_profile["crs"],
            "transform": reference_profile["transform"], "nodata": np.nan, "compress": "deflate",
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(result.astype(np.float32), 1)
    elif isinstance(result, Path):
        validate_feature_raster(name, result, reference_profile)
        shutil.copy(result, out_path)
    else:
        raise SystemExit(f"feature '{name}': build() must return an np.ndarray or Path, got {type(result)}")

    enforce_feature_size_cap(name, out_path, ref)
    print(f"[prepare.py] recipe '{name}' -> {out_path} ({out_path.stat().st_size} bytes)")


def enforce_feature_size_cap(name: str, out_path: Path, dem_path: Path) -> None:
    """Security check 3/4 (child-side): refuse a single cached raster over 2x
    dem.tif's file size rather than silently caching a pathologically large one."""
    max_bytes = 2 * dem_path.stat().st_size
    out_bytes = out_path.stat().st_size
    if out_bytes > max_bytes:
        out_path.unlink()
        raise SystemExit(
            f"feature '{name}': cached raster is {out_bytes} bytes, exceeding 2x dem.tif's "
            f"size ({max_bytes} bytes) — refusing to cache a pathologically large raster"
        )


def build_feature_grid(config: ProjectConfig, name: str) -> Path:
    """Build (or reuse) one features.py recipe's cached raster.

    Cache key = features_source_hash() (features.py's own source + a data
    fingerprint), so any edit to features.py — or a refreshed input file —
    invalidates every cached name. Once cached for a given hash, a recipe is
    NEVER recomputed (the user's explicit requirement): delete the cache
    directory, or edit features.py, to force a rebuild. The build itself runs
    in an isolated subprocess under a wall-clock timeout
    (config.feature_build_timeout_s, default 900s) killed by process GROUP
    (not just the direct child) so a hung `gdalwarp` the recipe shelled out to
    is also reclaimed.
    """
    src_hash = features_source_hash(config)
    cache_dir = REPO_ROOT / FEATURES_CACHE_DIR / name
    cache_path = cache_dir / f"{src_hash}.tif"
    if cache_path.exists():
        return cache_path

    scratch_out = REPO_ROOT / FEATURES_SCRATCH_DIR / f"{name}.building.tif"
    scratch_out.parent.mkdir(parents=True, exist_ok=True)
    if scratch_out.exists():
        scratch_out.unlink()

    cmd = [sys.executable, str(REPO_ROOT / "prepare.py"), "--run-recipe", name,
           "--recipe-out", str(scratch_out)]
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, start_new_session=True)
    try:
        ret = proc.wait(timeout=config.feature_build_timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
        raise SystemExit(
            f"feature '{name}' exceeded feature_build_timeout_s="
            f"{config.feature_build_timeout_s}s and was killed (process group)"
        )
    if ret != 0:
        raise SystemExit(f"feature '{name}' recipe failed (exit {ret}); see output above")
    if not scratch_out.exists():
        raise SystemExit(f"feature '{name}' recipe reported success but wrote no output")

    enforce_cache_budget(name, scratch_out)
    cache_dir.mkdir(parents=True, exist_ok=True)
    scratch_out.rename(cache_path)
    return cache_path


def enforce_cache_budget(name: str, scratch_out: Path) -> None:
    """Security check 4/4 (parent-side): refuse a cache addition that would push
    artifacts/features_cache/ over its 15GB total budget."""
    cache_root = REPO_ROOT / FEATURES_CACHE_DIR
    current_bytes = sum(p.stat().st_size for p in cache_root.rglob("*.tif")) if cache_root.exists() else 0
    new_bytes = scratch_out.stat().st_size
    if current_bytes + new_bytes > FEATURES_CACHE_BUDGET_BYTES:
        scratch_out.unlink()
        raise SystemExit(
            f"feature '{name}': caching it would grow {FEATURES_CACHE_DIR}/ past the "
            f"{FEATURES_CACHE_BUDGET_BYTES // (1024**3)}GB budget "
            f"({current_bytes/1e9:.2f}GB used + {new_bytes/1e6:.1f}MB new). Run "
            "`python prepare.py --prune-feature-cache` to reclaim space from recipes "
            "no longer in features.py's RECIPES, or from stale hashes of kept recipes."
        )


def prune_feature_cache(config: ProjectConfig) -> None:
    """Remove cache entries no longer reachable: whole name-directories for
    recipes no longer in features.py's RECIPES, and stale (non-current-hash)
    files within directories that ARE still referenced. Documented escape
    hatch for the 15GB cache budget in build_feature_grid()."""
    features_path = REPO_ROOT / FEATURES_MODULE_NAME
    spec = importlib.util.spec_from_file_location("features", features_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    current_names = {r["name"] for r in module.RECIPES}
    current_hash = features_source_hash(config)

    cache_root = REPO_ROOT / FEATURES_CACHE_DIR
    if not cache_root.is_dir():
        print("Nothing to prune (no features_cache/ yet).")
        return
    removed: list[str] = []
    for name_dir in sorted(cache_root.iterdir()):
        if not name_dir.is_dir():
            continue
        if name_dir.name not in current_names:
            shutil.rmtree(name_dir)
            removed.append(f"{name_dir.name}/ (recipe no longer in RECIPES)")
            continue
        for hash_file in sorted(name_dir.glob("*.tif")):
            if hash_file.stem != current_hash:
                hash_file.unlink()
                removed.append(f"{name_dir.name}/{hash_file.name} (stale hash)")
    print("Pruned:" if removed else "Nothing to prune.")
    for line in removed:
        print(f"  {line}")


def describe_primitives(config: ProjectConfig) -> str:
    """Raw primitives available to features.py: DEM stats, vector/raster layer
    inventory under data/vectors/, and the dynamic archive listing. The
    raw-primitives analogue of --describe-dynamic."""
    lines = ["Raw primitives available to features.py:", ""]
    dem = reference_grid(discover_grids(config.grids_dir))
    with rasterio.open(dem) as src:
        band = src.read(1, masked=True)
        lines += [
            f"DEM ({dem.name}):",
            f"  crs={src.crs}  shape={src.height}x{src.width}  res={src.res}",
            f"  bounds={tuple(src.bounds)}",
            f"  elevation range: {float(band.min()):.1f} .. {float(band.max()):.1f} m",
            "",
        ]
    vectors_dir = config.data_dir / "vectors"
    if vectors_dir.is_dir():
        lines.append("Layers under data/vectors/:")
        seen: set[str] = set()
        for path in sorted(vectors_dir.iterdir()):
            stem = path.stem
            if stem in seen or path.suffix.lower() not in (".gpkg", ".shp", ".tif"):
                continue
            seen.add(stem)
            if path.suffix.lower() == ".tif":
                with rasterio.open(path) as src:
                    lines.append(
                        f"  {path.name}: raster, crs={src.crs}, shape={src.height}x{src.width}"
                    )
            else:
                try:
                    gdf = gpd.read_file(path)
                    lines.append(
                        f"  {path.name}: {len(gdf)} features, "
                        f"geometry={sorted(gdf.geom_type.unique())}, crs={gdf.crs}"
                    )
                except Exception as exc:
                    lines.append(f"  {path.name}: (unreadable: {exc})")
        lines.append("")
    dynamic = discover_dynamic(config.data_dir)
    lines.append(
        "Dynamic archives under data/dynamic/: " + ", ".join(sorted(dynamic))
        if dynamic else "No dynamic archives."
    )
    if dynamic:
        lines.append("Run --describe-dynamic for units/coverage/value-range detail.")
    return "\n".join(lines)


# ============================================================
# EBM explanation — feature importances + TalkToEBM text
# ============================================================


def _t2ebm_graphs():
    # t2ebm instantiates an OpenAI client at import time; the offline
    # graph-to-text API works fine with a placeholder key.
    os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder-t2ebm-offline")
    from t2ebm import graphs

    return graphs


def term_importance_table(model, top_n: int = 20) -> list[tuple[str, float]]:
    names = list(model.term_names_)
    scores = [float(v) for v in model.term_importances()]
    ranked = sorted(zip(names, scores), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_n]


def graph_texts(model, top_n: int = 6, max_tokens: int = 1500) -> list[tuple[str, str]]:
    """TalkToEBM text renderings of the highest-importance univariate shape functions."""
    graphs = _t2ebm_graphs()
    scores = model.term_importances()
    # Zero-importance terms have constant/degenerate graphs that can hang the
    # t2ebm simplifier (e.g. the null-baseline constant feature) — skip them.
    univariate = [
        i for i, term in enumerate(model.term_features_)
        if len(term) == 1 and float(scores[i]) > 1e-12
    ]
    ranked = sorted(univariate, key=lambda i: float(scores[i]), reverse=True)[:top_n]
    out: list[tuple[str, str]] = []
    for index in ranked:
        name = str(model.term_names_[index])
        try:
            graph = graphs.extract_graph(model, index)
            graph = graphs.simplify_graph(graph, min_variation_per_cent=0.04)
            out.append((name, graphs.graph_to_text(graph, max_tokens=max_tokens)))
        except Exception as exc:
            out.append((name, f"(graph-to-text failed: {exc})"))
    return out


def write_explanations(bundle: dict[str, Any], out_path: Path, *, top_terms: int = 20, top_graphs: int = 6) -> None:
    """Write explain.md for a train.py model bundle."""
    lines = ["# Model Explanation", ""]
    for process_key, item in bundle["processes"].items():
        model = item["model"]
        lines += [f"## {process_key}", "", "### Term importances (mean absolute score)", ""]
        for name, score in term_importance_table(model, top_terms):
            lines.append(f"- {name}: {score:.5f}")
        lines.append("")
        try:
            texts = graph_texts(model, top_graphs)
        except Exception as exc:
            texts = []
            lines += [f"(TalkToEBM unavailable: {exc})", ""]
        for name, text in texts:
            lines += [f"### Shape of `{name}`", "", "```", text.strip(), "```", ""]
    out_path.write_text("\n".join(lines))
    log.info("Wrote %s", out_path)


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="config.yml", help="Optional override; defaults apply when absent.")
    parser.add_argument("--process", default=None, help="Comma-separated process keys or 'all'.")
    parser.add_argument("--out", default="artifacts/features")
    parser.add_argument("--explain-bundle", default=None, help="Write explain.md for a model bundle and exit.")
    parser.add_argument("--describe-dynamic", action="store_true",
                        help="Print the time-varying archives (name, units, coverage, "
                             "resolution, value range) from data/dynamic/ and exit.")
    parser.add_argument("--describe-primitives", action="store_true",
                        help="Print DEM stats, vector/raster layer inventory, and the "
                             "dynamic archive listing available to features.py, and exit.")
    parser.add_argument("--describe-raster-classes", action="store_true",
                        help="Print class codes present (reprojected onto the reference grid, "
                             "with pixel counts) for a categorical raster -- pick one from "
                             "--describe-primitives's inventory via --source, or fall back to "
                             "config.yml's sampling.exclude_raster if already set. Exit.")
    parser.add_argument("--source", default=None,
                        help="Path (relative to data/) to a categorical raster, for "
                             "--describe-raster-classes, e.g. a land-cover, geology, or "
                             "demographic-zoning layer such as vectors/CorineLandCover_polygon.tif.")
    parser.add_argument("--build-exclude-mask", action="store_true",
                        help="Build data/grids/mask_exclude.tif from config.yml's "
                             "sampling.exclude_raster + exclude_classes, and exit. "
                             "Setup-only.")
    parser.add_argument("--run-recipe", default=None,
                        help="Internal: execute one features.py recipe in isolation "
                             "(used by build_feature_grid() via subprocess). Requires --recipe-out.")
    parser.add_argument("--recipe-out", default=None, help="Output path for --run-recipe.")
    parser.add_argument("--run-derived", action="store_true",
                        help="Internal: compute one derived_grid() aggregate in isolation "
                             "(used by derived_grid() via subprocess). Requires --var, --stat, --derived-out.")
    parser.add_argument("--stat", default=None, help="mean|min|max|std for --run-derived.")
    parser.add_argument("--months", default=None,
                        help="Comma-separated calendar months 1-12 for --run-derived, e.g. '6,7,8'.")
    parser.add_argument("--derived-out", default=None, help="Output path for --run-derived.")
    parser.add_argument("--run-derived-all", action="store_true",
                        help="Internal: compute every DERIVED_STATS entry for one (var, months) "
                             "in a single archive scan (used by derived_grid() via subprocess). "
                             "Requires --var, --derived-out-dir.")
    parser.add_argument("--derived-out-dir", default=None, help="Output directory for --run-derived-all.")
    parser.add_argument("--warm-derived-cache", action="store_true",
                        help="Pre-build every discovered dynamic variable's full-archive "
                             "mean/min/max/std aggregates and exit. Meant to be run from loop.sh "
                             "before the agent session starts, not from inside the agent's own "
                             "tool calls -- see warm_derived_cache()'s docstring.")
    parser.add_argument("--prune-feature-cache", action="store_true",
                        help="Remove artifacts/features_cache/ entries for recipes no longer "
                             "in features.py's RECIPES, and stale hashes of kept recipes, and exit.")
    parser.add_argument("--var", default=None, help="Canonical variable name for --run-derived/--run-derived-all.")
    parser.add_argument("--log-experiment", action="store_true",
                        help="Append one row to the experiment ledger and exit "
                             "(requires --commit, --score, --status, --description, --spec-hash).")
    parser.add_argument("--commit", default=None, help="Candidate commit hash for --log-experiment.")
    parser.add_argument("--score", type=float, default=None, help="val_pr_auc (decision metric) for --log-experiment.")
    parser.add_argument("--roc-auc", type=float, default=float("nan"), help="val_roc_auc (secondary, informational) for --log-experiment.")
    parser.add_argument("--pr-gain-auc", type=float, default=float("nan"), help="val_pr_gain_auc / AUPRG (secondary, informational) for --log-experiment.")
    parser.add_argument("--status", default=None, help="baseline|keep|discard|crash for --log-experiment.")
    parser.add_argument("--description", default=None, help="One-line experiment summary for --log-experiment.")
    parser.add_argument("--spec-hash", default=None, help="12-char spec_hash from run.log for --log-experiment.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.log_experiment:
        missing = [name for name in ("commit", "score", "status", "description", "spec_hash")
                   if getattr(args, name) is None]
        if missing:
            raise SystemExit("--log-experiment requires --" + ", --".join(m.replace("_", "-") for m in missing))
        config = load_config(args.config)
        if len(config.processes) != 1:
            raise SystemExit("--log-experiment requires the branch config to pin exactly one process")
        row = log_experiment(config.processes[0], args.commit, args.score,
                             args.status, args.description, args.spec_hash,
                             roc_auc=args.roc_auc, pr_gain_auc=args.pr_gain_auc)
        print(row)
        return

    if args.explain_bundle:
        import pickle

        bundle_path = Path(args.explain_bundle)
        with bundle_path.open("rb") as handle:
            bundle = pickle.load(handle)
        write_explanations(bundle, bundle_path.parent / "explain.md")
        return
    if args.describe_dynamic:
        print(describe_dynamic(load_config(args.config)))
        return
    if args.describe_primitives:
        print(describe_primitives(load_config(args.config)))
        return
    if args.describe_raster_classes:
        print(describe_raster_classes(load_config(args.config), source=args.source))
        return
    if args.build_exclude_mask:
        build_exclude_mask(load_config(args.config))
        return
    if args.run_recipe:
        if not args.recipe_out:
            raise SystemExit("--run-recipe requires --recipe-out")
        run_recipe(load_config(args.config), args.run_recipe, Path(args.recipe_out))
        return
    if args.run_derived:
        missing = [n for n in ("var", "stat", "derived_out") if getattr(args, n) is None]
        if missing:
            raise SystemExit("--run-derived requires --" + ", --".join(m.replace("_", "-") for m in missing))
        months = [int(m) for m in args.months.split(",") if m.strip()] if args.months else None
        run_derived(load_config(args.config), args.var, args.stat, months, Path(args.derived_out))
        return
    if args.run_derived_all:
        missing = [n for n in ("var", "derived_out_dir") if getattr(args, n) is None]
        if missing:
            raise SystemExit("--run-derived-all requires --" + ", --".join(m.replace("_", "-") for m in missing))
        months = [int(m) for m in args.months.split(",") if m.strip()] if args.months else None
        run_derived_all_stats(load_config(args.config), args.var, months, Path(args.derived_out_dir))
        return
    if args.warm_derived_cache:
        warm_derived_cache(load_config(args.config))
        return
    if args.prune_feature_cache:
        prune_feature_cache(load_config(args.config))
        return

    config = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = [prepare_one(config, key, out) for key in parse_processes(config, args.process)]
    summary_path = out / "summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    log.info("Wrote %s", summary_path)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
