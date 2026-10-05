#!/usr/bin/env python3
"""Fixed study infrastructure for a static susceptibility model. The agent never edits it.

  python prepare.py [--force]           build the presence/absence sample once
  python prepare.py log STATUS "WHY"    append the last run to results.tsv (keep|discard|crash)
  python prepare.py usage               size of artifacts/ + tmp/ against limits.artifacts_gb
  python prepare.py prune               delete artifacts/features/<name>/ no kept run uses

Spatial sampling after Moreno et al. (2025, JGR-ES, doi:10.1029/2024JF008219) and
Moreno et al. (2024, STOTEN 912:169166). Every rule applies to both classes and the map:
 1. Domain = valid DEM cells minus trivial terrain from Corine Land Cover (rock, glacier,
    water), optionally minus flat land and cells outside a supplied survey mask.
 2. Presences: inventoried events inside the domain, one per DEM cell.
 3. Absences: random domain cells >= buffer_m from ANY inventoried event (also events
    outside the domain), absence_ratio per presence.
 4. Folds: k-means clusters of the coordinates (spatial CV, as sperrorest partition_kmeans).
Predictors are rasters: raw inputs in data/, or artifacts/features/<name>/ outputs saved with
the recipe script that made them. evaluate() scores them on the frozen folds, maps the domain
and records each recipe's SAGA lines, so results.tsv lists exactly what every run computed.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
ART, RESULTS = ROOT / "artifacts", ROOT / "results.tsv"
PREPARED, FEATURES, MAPS = ART / "prepared", ART / "features", ART / "maps"
LAST_RUN, LAST_MAP = ART / "last_run.json", ART / "last_map.tif"
LAST_EXPLANATION, LAST_MODEL = ART / "last_explanation.md", ART / "last_model.json"
BUDGET_DIRS = (ART, ROOT / "tmp")
FIELDS = ["commit", "val_auroc", "val_pr_auc", "status", "description", "features", "saga_commands",
          "study", "experiment"]

CFG = yaml.safe_load((ROOT / "config.yml").read_text())
LIMITS, SAMPLING = CFG["limits"], CFG["sampling"]
SEED, CPUS = int(SAMPLING["seed"]), int(LIMITS["cpus"])
DEM = ROOT / CFG["data"]["dem"]

# Long computations are fine, within limits. Any process importing this module (train.py,
# tmp/ scripts) ends after limits.timeout_min (default SIGALRM action, even in native code)
# and gets MemoryError beyond limits.memory_gb of allocated data (RLIMIT_DATA, Linux >= 4.7).
signal.alarm(max(1, int(LIMITS["timeout_min"] * 60)))
_hard = resource.getrlimit(resource.RLIMIT_DATA)[1]
_cap = int(LIMITS["memory_gb"] * 1e9)
_cap = _cap if _hard == resource.RLIM_INFINITY else min(_cap, _hard)
resource.setrlimit(resource.RLIMIT_DATA, (_cap, _cap))

# Python's GDAL skips conda plugins that clash with pip rasterio (saga_cmd is unaffected).
os.environ["GDAL_DRIVER_PATH"] = "disable"
_spec = importlib.util.find_spec("rasterio")
if _spec and _spec.origin and (Path(_spec.origin).parent / "proj_data").is_dir():
    os.environ["PROJ_DATA"] = str(Path(_spec.origin).parent / "proj_data")
    os.environ.pop("PROJ_LIB", None)


# ---------------------------------------------------------------- disk budget
def usage_bytes(path: Path | None = None) -> int:
    if path is None:
        return sum(usage_bytes(d) for d in BUDGET_DIRS)
    total = 0
    for directory, _, files in os.walk(path):
        for name in files:
            try:
                st = os.lstat(os.path.join(directory, name))
            except FileNotFoundError:
                continue
            total += max(st.st_size, st.st_blocks * 512)
    return total


def check_budget() -> None:
    used, limit = usage_bytes(), LIMITS["artifacts_gb"] * 1e9
    if used > limit:
        raise RuntimeError(f"artifacts/ + tmp/ use {used / 1e9:.2f} GB > {LIMITS['artifacts_gb']} GB. "
                           "Run `python prepare.py prune` or delete stale outputs.")


# ---------------------------------------------------------------- grids
def read_dem() -> tuple[np.ndarray, dict]:
    import rasterio
    with rasterio.open(DEM) as src:
        return src.read(1, masked=True).astype("float64").filled(np.nan), src.profile


def on_dem(path: str | Path, categorical: bool = False, src_crs: str | None = None) -> np.ndarray:
    """Any single-band raster warped onto the DEM grid (nearest if categorical, else bilinear)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    _, p = read_dem()
    with rasterio.open(path) as src, WarpedVRT(
            src, src_crs=src_crs or src.crs, crs=p["crs"], transform=p["transform"], width=p["width"],
            height=p["height"], resampling=Resampling.nearest if categorical else Resampling.bilinear) as vrt:
        return vrt.read(1, masked=True).astype("float64").filled(np.nan)


def write_grid(path: Path, values: np.ndarray) -> Path:
    import rasterio
    _, p = read_dem()
    out = p | {"driver": "GTiff", "count": 1, "dtype": "float32", "nodata": np.nan, "compress": "deflate"}
    with rasterio.open(path, "w", **out) as dst:
        dst.write(values.astype("float32"), 1)
    return path


def cells(xy: np.ndarray, profile: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row, column and inside-grid flag of coordinates."""
    t = profile["transform"]
    col = np.floor((xy[:, 0] - t.c) / t.a).astype(int)
    row = np.floor((xy[:, 1] - t.f) / t.e).astype(int)
    inside = (row >= 0) & (row < profile["height"]) & (col >= 0) & (col < profile["width"])
    return row, col, inside


# ---------------------------------------------------------------- sampling
def build_domain(z: np.ndarray, profile: dict) -> np.ndarray:
    lc = CFG["data"]["landcover"]
    landcover = on_dem(ROOT / lc["path"], categorical=True, src_crs=lc.get("crs"))
    domain = np.isfinite(z) & np.isfinite(landcover) & ~np.isin(landcover, SAMPLING["mask_landcover"])
    if SAMPLING.get("flat_slope_deg") is not None:
        gy, gx = np.gradient(z, abs(profile["transform"].e), profile["transform"].a)
        domain &= np.degrees(np.arctan(np.hypot(gx, gy))) >= SAMPLING["flat_slope_deg"]
    if SAMPLING.get("survey_mask"):
        domain &= on_dem(ROOT / SAMPLING["survey_mask"], categorical=True) == 1
    return domain


def prepare(force: bool = False) -> None:
    import geopandas as gpd
    from scipy.spatial import cKDTree
    from sklearn.cluster import KMeans
    if PREPARED.exists() and not force:
        print((PREPARED / "report.json").read_text())
        print("Already prepared. `--force` rebuilds it; results.tsv rows of the old study stay separate.")
        return
    ART.mkdir(exist_ok=True)
    check_budget()
    rng = np.random.default_rng(SEED)
    z, profile = read_dem()
    domain = build_domain(z, profile)
    width = z.shape[1]

    ev = gpd.read_file(ROOT / CFG["data"]["events"]).to_crs(profile["crs"])
    ev = ev[ev.geometry.notna() & ~ev.geometry.is_empty & (ev.geom_type == "Point")]
    xy = np.c_[ev.geometry.x, ev.geometry.y]
    row, col, inside = cells(xy, profile)
    in_domain = inside.copy()
    in_domain[inside] = domain[row[inside], col[inside]]
    presence_cells = np.unique(row[in_domain] * width + col[in_domain])

    t = profile["transform"]
    cr, cc = np.nonzero(domain)
    cand = np.c_[t.c + (cc + 0.5) * t.a, t.f + (cr + 0.5) * t.e]
    free = cKDTree(xy).query(cand)[0] >= SAMPLING["buffer_m"]
    free &= ~np.isin(cr * width + cc, row[inside] * width + col[inside])
    n_abs = int(SAMPLING["absence_ratio"]) * len(presence_cells)
    if n_abs > free.sum():
        raise ValueError(f"Only {free.sum()} candidate absence cells for {n_abs}; lower absence_ratio")
    absence_cells = (cr * width + cc)[free][rng.choice(free.sum(), n_abs, replace=False)]

    frame = pd.DataFrame({"cell": np.r_[presence_cells, absence_cells],
                          "label": np.r_[np.ones(len(presence_cells), int), np.zeros(n_abs, int)]})
    frame["row"], frame["col"] = frame.cell // width, frame.cell % width
    frame["x"], frame["y"] = t.c + (frame.col + 0.5) * t.a, t.f + (frame.row + 0.5) * t.e
    frame["fold"] = KMeans(CFG["evaluation"]["folds"], n_init=10, random_state=SEED).fit_predict(frame[["x", "y"]])
    if (frame.groupby("fold").label.nunique() < 2).any():
        raise ValueError("A spatial fold lacks one class; reduce evaluation.folds")
    frame = frame.drop(columns="cell")
    frame.insert(0, "sample_id", np.arange(len(frame)))

    report = {"events": len(ev), "outside_grid": int((~inside).sum()),
              "outside_domain": int((inside & ~in_domain).sum()),
              "presence_cells": len(presence_cells), "absence_cells": n_abs,
              "domain_cells": int(domain.sum()), "fold_events": frame.groupby("fold").label.sum().to_dict()}
    tmp = ART / ".prepared-tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    frame.to_csv(tmp / "samples.csv", index=False)
    frame[["sample_id", "x", "y", "row", "col"]].to_csv(tmp / "points.csv", index=False)
    write_grid(tmp / "domain.tif", np.where(np.isfinite(z), domain, np.nan))
    report["study"] = hashlib.sha256((tmp / "samples.csv").read_bytes()).hexdigest()[:10]
    report["sampling"] = SAMPLING
    (tmp / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    shutil.rmtree(PREPARED, ignore_errors=True)
    tmp.rename(PREPARED)
    print(json.dumps(report, indent=2, default=str))


# ---------------------------------------------------------------- evaluation
def study_id() -> str:
    return json.loads((PREPARED / "report.json").read_text())["study"]


def _rel(path: str | Path) -> str:
    p = Path(os.path.abspath(ROOT / path))
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def recipe(path: str) -> str:
    """Provenance of a feature raster: raw input, or the recipe script saved beside it."""
    if path.startswith("data/"):
        return "raw input"
    parts = Path(path).parts
    if len(parts) < 4 or parts[:2] != ("artifacts", "features"):
        raise ValueError(f"{path}: features are raw data/ rasters or artifacts/features/<name>/ outputs")
    scripts = sorted((ROOT / Path(*parts[:3])).glob("recipe.*"))
    if not scripts:
        raise ValueError(f"{path}: save the script that made it as {Path(*parts[:3])}/recipe.sh or recipe.py")
    return "\n".join(f"# {_rel(s)}\n{s.read_text()}" for s in scripts)


def experiment_hash(recipes: dict[str, str]) -> str:
    """train.py (comments and formatting ignored) plus the recipes of every feature it uses."""
    code = ast.dump(ast.parse((ROOT / "train.py").read_text()))
    return hashlib.sha256(json.dumps([code, recipes], sort_keys=True).encode()).hexdigest()[:12]


def explain(ebm) -> str:
    """Glass-box report via TalkToEBM: importances and every shape function as text.

    The coding agent reads it directly. With T2EBM_LLM (e.g. gpt-4o) and OPENAI_API_KEY set,
    t2ebm.describe_ebm adds a narrative summary as well.
    """
    placeholder = "OPENAI_API_KEY" not in os.environ
    os.environ.setdefault("OPENAI_API_KEY", "offline")  # t2ebm builds an OpenAI client on import
    import t2ebm
    from t2ebm.graphs import extract_graph, graph_to_text, simplify_graph
    if placeholder:
        os.environ.pop("OPENAI_API_KEY")
    parts = ["# EBM explanation (TalkToEBM)", "", "## Term importances", "", t2ebm.feature_importances_to_text(ebm)]
    for i, name in enumerate(ebm.term_names_):
        if len(ebm.term_features_[i]) > 1:
            parts += [f"## {name}", "", "Pairwise interaction: see the model JSON for its score table.", ""]
            continue
        graph, precision = extract_graph(ebm, i), None
        if np.ptp(graph.scores) == 0:  # t2ebm 0.1.2 loops forever on flat graphs
            parts += [f"## {name}", "", "Flat shape function: no effect on the prediction.", ""]
            continue
        if graph.feature_type == "continuous":
            graph, precision = simplify_graph(graph, min_variation_per_cent=0.05), 3
        try:
            text = graph_to_text(graph, x_axis_precision=precision, max_tokens=1500)
        except Exception as e:  # noqa: BLE001  t2ebm raises bare Exception for oversized graphs
            text = f"Not rendered as text ({e}); see the model JSON."
        parts += [f"## {name}", "", text, ""]
    if os.environ.get("T2EBM_LLM") and not placeholder:
        parts += ["## LLM summary", "", t2ebm.describe_ebm(os.environ["T2EBM_LLM"], ebm)]
    return "\n".join(parts) + "\n"


def _positive(model, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(X))[:, list(model.classes_).index(1)]


def evaluate(features: dict[str, str], make_model: Callable[[], Any], categorical: tuple[str, ...] = ()) -> dict:
    """Fixed metric: mean AUROC over k-means spatial folds; also maps the full-data model.

    features: name -> raster path relative to the repository (any CRS/resolution; it is
    warped onto the DEM grid, nearest-neighbour for names listed in `categorical`).
    """
    from interpret.glassbox import ExplainableBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    started = time.time()
    if not isinstance(make_model(), ExplainableBoostingClassifier):
        sys.exit("make_model() must return an interpret ExplainableBoostingClassifier: EBM is compulsory.")
    if not (PREPARED / "samples.csv").exists():
        sys.exit("No prepared sample: run `python prepare.py` first.")
    check_budget()
    paths = {name: _rel(p) for name, p in features.items()}
    recipes = {name: recipe(p) for name, p in paths.items()}
    h, study = experiment_hash(recipes), study_id()
    for row in ledger():
        if row["experiment"] == h and row["study"] == study and os.environ.get("RETRY") != "1":
            sys.exit(f"DUPLICATE of {row['commit']} ({row['status']}, val_auroc {row['val_auroc']}). "
                     "Change the experiment, or RETRY=1 to repeat it.")
    for path in (LAST_RUN, LAST_MAP, LAST_EXPLANATION, LAST_MODEL):
        path.unlink(missing_ok=True)
    z, _ = read_dem()
    domain = on_dem(PREPARED / "domain.tif", categorical=True) == 1
    landcover = CFG["data"]["landcover"]
    grids = {}
    if not 1 <= len(paths) <= 200:
        raise ValueError("Need 1-200 features")
    for name, p in paths.items():
        crs = landcover.get("crs") if p == _rel(landcover["path"]) else None
        g = on_dem(ROOT / p, name in categorical, crs)
        if np.isinf(g[domain]).any() or np.isnan(g[domain]).all():
            raise ValueError(f"Feature {name}: infinite values or empty on the domain")
        grids[name] = g
    s = pd.read_csv(PREPARED / "samples.csv")
    X = pd.DataFrame({n: g[s.row, s.col] for n, g in grids.items()})
    y, folds = s.label.to_numpy(), []
    for fold in sorted(s.fold.unique()):
        te = s.fold.to_numpy() == fold
        prob = _positive(make_model().fit(X[~te], y[~te]), X[te])
        folds.append({"fold": int(fold), "auroc": roc_auc_score(y[te], prob),
                      "pr_auc": average_precision_score(y[te], prob), "events": int(y[te].sum()), "n": int(te.sum())})
    result: dict[str, Any] = {"val_auroc": float(np.mean([f["auroc"] for f in folds])),
              "val_auroc_sd": float(np.std([f["auroc"] for f in folds])),
              "val_pr_auc": float(np.mean([f["pr_auc"] for f in folds])), "folds": folds}
    model = make_model().fit(X, y)
    surface, cells_x = np.full(z.shape, np.nan), pd.DataFrame({n: g[domain] for n, g in grids.items()})
    surface[domain] = np.concatenate([_positive(model, cells_x.iloc[i:i + 200_000])
                                      for i in range(0, len(cells_x), 200_000)])
    write_grid(LAST_MAP, surface)
    model.to_json(LAST_MODEL)
    LAST_EXPLANATION.write_text(explain(model))
    result["importance"] = dict(sorted(zip(model.term_names_, map(float, model.term_importances())),
                                       key=lambda kv: -kv[1]))
    saga_lines = [m.group(0).strip() for r in recipes.values() for line in r.splitlines()
                  if not line.lstrip().startswith("#") and (m := re.search(r"saga_cmd\s[^;&|)>\n]*", line))]
    result |= {"commit": git_commit(), "experiment": h, "study": study, "features": paths,
               "recipes": recipes, "saga_commands": list(dict.fromkeys(saga_lines)),
               "model": repr(make_model())[:2000], "seconds": round(time.time() - started, 1)}
    LAST_RUN.write_text(json.dumps(result, indent=2) + "\n")
    best = max((float(r["val_auroc"]) for r in ledger() if r["status"] == "keep" and r["study"] == study),
               default=None)
    print("---")
    for key in ("val_auroc", "val_auroc_sd", "val_pr_auc"):
        print(f"{key}: {result[key]:.6f}")
    print(f"best_kept_val_auroc: {'none' if best is None else f'{best:.6f}'}")
    print(f"n_features: {X.shape[1]}\nn_samples: {len(y)} ({int(y.sum())} presences)")
    print(f"eval_seconds: {result['seconds']}\nmap: {_rel(LAST_MAP)}\nexplanation: {_rel(LAST_EXPLANATION)}")
    for name, value in list(result.get("importance", {}).items())[:15]:
        print(f"importance: {value:.4f} {name}")
    return result


# ---------------------------------------------------------------- ledger
def git_commit() -> str:
    proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=False)
    return proc.stdout.strip() or "nogit"


def ledger() -> list[dict]:
    if not RESULTS.exists():
        return []
    with RESULTS.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"))


def log(status: str, description: str) -> None:
    if status not in {"keep", "discard", "crash"}:
        sys.exit("status must be keep, discard or crash")
    run = json.loads(LAST_RUN.read_text()) if LAST_RUN.exists() else {}
    fresh = run.get("commit") == git_commit() and run.get("experiment") == experiment_hash(run.get("recipes", {}))
    if status != "crash" and not fresh:
        sys.exit("No finished run for the current commit and train.py: rerun it, or log it as crash.")
    commit = git_commit()
    row = {"commit": commit, "status": status, "study": study_id(),
           "experiment": run["experiment"] if fresh else "",
           "val_auroc": f"{run['val_auroc']:.6f}" if fresh else "0.000000",
           "val_pr_auc": f"{run['val_pr_auc']:.6f}" if fresh else "0.000000",
           "description": " ".join(description.replace('"', "'").split()),
           "features": ",".join(f"{k}={v}" for k, v in run["features"].items()) if fresh else "",
           "saga_commands": " && ".join(run["saga_commands"]) if fresh else ""}
    new = not RESULTS.exists()
    with RESULTS.open("a", newline="") as f:
        writer = csv.DictWriter(f, FIELDS, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        if new:
            writer.writeheader()
        writer.writerow(row)
    if status == "keep":
        MAPS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LAST_MAP, MAPS / f"susceptibility_{commit}.tif")
        shutil.copy2(LAST_MAP, MAPS / "susceptibility_best.tif")
        for src, name in ((LAST_RUN, "run"), (LAST_MODEL, "ebm"), (LAST_EXPLANATION, "explanation")):
            shutil.copy2(src, MAPS / f"{name}_{commit}{src.suffix}")
            shutil.copy2(src, MAPS / f"{name}_best{src.suffix}")
    print("\t".join(row[k] for k in FIELDS[:5]))


def usage() -> None:
    print(f"artifacts/ + tmp/: {usage_bytes() / 1e9:.3f} GB of {LIMITS['artifacts_gb']} GB")
    for child in sorted(p for d in BUDGET_DIRS if d.exists() for p in d.iterdir() if p.is_dir()):
        print(f"  {_rel(child):<40} {usage_bytes(child) / 1e9:8.3f} GB")


def prune() -> None:
    """Delete artifacts/features/<name>/ directories that no `keep` row (or the last run) uses."""
    text = "".join(r["features"] for r in ledger() if r["status"] == "keep")
    text += LAST_RUN.read_text() if LAST_RUN.exists() else ""
    used, freed = set(re.findall(r"artifacts/features/([^/,\"]+)/", text)), 0
    for entry in FEATURES.iterdir() if FEATURES.exists() else []:
        if entry.is_dir() and entry.name not in used:
            freed += usage_bytes(entry)
            shutil.rmtree(entry)
    print(f"Freed {freed / 1e9:.3f} GB")
    usage()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="prepare", choices=["prepare", "log", "usage", "prune"])
    parser.add_argument("args", nargs="*")
    parser.add_argument("--force", action="store_true", help="rebuild the prepared sample")
    a = parser.parse_args()
    if a.command == "log":
        if len(a.args) != 2:
            sys.exit('Usage: python prepare.py log keep|discard|crash "description"')
        log(*a.args)
    elif a.command == "usage":
        usage()
    elif a.command == "prune":
        prune()
    else:
        prepare(a.force)
