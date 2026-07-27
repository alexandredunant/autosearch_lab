#!/usr/bin/env python3
"""THE experiment file. This and features.py are the only files the
autoresearch agent edits.

Each research run targets exactly ONE hazard process — the single key in
`project.processes` in this branch's config.yml. Different hazards are fully
independent runs on their own git branches. Train an Explainable Boosting
Machine (EBM) on the prepared feature matrix and report the primary metric.
The model family is fixed: interpret.glassbox.ExplainableBoostingClassifier,
nothing else. The evaluation protocol comes from prepare.py and must be used
unchanged: val_pr_auc = mean PR-AUC across the spatial K-folds.

Output contract (the loop depends on it):
  - artifacts under --out (metrics.json, model_bundle.pkl, summary.md, explain.md)
  - stdout carries an early line:  spec_hash: <hash>  (the experiment identity;
    duplicates of ledger rows are rejected before training)
  - the final stdout line is exactly:  val_pr_auc: <float>
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import threading
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier

from features import FEATURE_RATIONALE, RECIPES
from prepare import (
    assert_new_experiment,
    build_feature_grid,
    derived_grid,
    derived_name,
    experiment_hash,
    load_artifact,
    load_config,
    parse_processes,
    sample_grids,
    scalar_metrics,
    select_feature_names,
    selected_matrix,
    spatial_folds,
    write_explanations,
)

# ============================================================
# EXPERIMENT — the hypothesis under test lives in this block
# ============================================================

# Feature selection by fnmatch name patterns against whatever is available —
# the static grids prepare.py sampled, plus any derived aggregates (DERIVED,
# below) and any features.py recipes (imported as RECIPES above). Discover the
# static names in artifacts/features/<process>/. An empty "include" list is
# the null baseline: the model trains on a single constant feature.
FEATURES = {
    "include": ["precipitation_sum_1d", "precipitation_sum_7d", "precipitation_sum_30d"],
    "exclude": [],
}

# Static aggregates of the time-varying archives (if any exist — run
# `python prepare.py --describe-dynamic` to see them). Each entry is
# {"var": <archive name>, "stat": "mean"|"min"|"max"|"std", "months": [1-12]}
# ("months" optional: restrict the aggregate to those calendar months).
# prepare.py reduces the archive to one cached static raster per entry; the
# resulting feature, named derived_name(var, stat, months), joins the same
# FEATURES include/exclude space as the grids and the RECIPES below. DERIVED
# ships empty. This declarative form still exists for simple aggregates;
# features.py's RECIPES generalize it to arbitrary raster-generation code —
# prefer RECIPES for anything DERIVED's {var, stat, months} can't express.
DERIVED: list[dict[str, Any]] = []

# Starting configuration, not a tuned one. The feature phase leaves this fixed;
# the hyperparameter phase is where it changes.
MODEL_CONFIG = {
    "outer_bags": 8,
    "learning_rate": 0.01,
    "validation_size": 0.15,
    "early_stopping_rounds": 50,
}

EXPERIMENT_RATIONALE = """
Add a 30-day antecedent precipitation total to the retained short-window
rainfall signals, testing whether monthly antecedent wetness adds stable
landslide discrimination before terrain or other static predictors.
"""

# ============================================================
# Pipeline — editable too, but the goalposts are fixed:
# EBM only, prepare.py spatial K-fold and metric, output contract above.
# ============================================================

FIT_HEARTBEAT_SECONDS = 30.0


def fit_ebm(X: np.ndarray, y: np.ndarray, feature_names: list[str], seed: int, label: str):
    model = ExplainableBoostingClassifier(
        feature_names=feature_names,
        random_state=seed,
        n_jobs=-1,
        **MODEL_CONFIG,
    )
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(FIT_HEARTBEAT_SECONDS):
            print(
                f"[train.py] still fitting {label}: rows={len(y)} features={len(feature_names)}",
                file=sys.stderr,
                flush=True,
            )

    print(
        f"[train.py] fitting {label}: rows={len(y)} features={len(feature_names)}",
        file=sys.stderr,
        flush=True,
    )
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        model.fit(X, y)
    finally:
        stop.set()
        thread.join(timeout=1.0)
    print(f"[train.py] finished {label}", file=sys.stderr, flush=True)
    return model


def importances(model) -> dict[str, float]:
    try:
        data = model.explain_global().data()
        return {str(name): float(score) for name, score in zip(data["names"], data["scores"])}
    except Exception:
        return {}


def derived_columns(config, metadata, crs: str) -> dict[str, np.ndarray]:
    """Sample each DERIVED aggregate raster at the observations.

    prepare.py computes (and caches) the raster; here it is sampled at every
    observation's x/y like any static grid. Observations outside an archive's
    footprint come back NaN, which the EBM reads natively.
    """
    out: dict[str, np.ndarray] = {}
    for entry in DERIVED:
        name = derived_name(entry["var"], entry["stat"], entry.get("months"))
        path = derived_grid(config, entry["var"], entry["stat"], entry.get("months"))
        points = gpd.GeoSeries(gpd.points_from_xy(metadata["x"], metadata["y"]), crs=crs)
        out[name] = sample_grids(points, {name: path})[name].to_numpy(dtype=np.float32)
    return out


def recipe_columns(config, metadata, crs: str) -> dict[str, np.ndarray]:
    """Sample each features.py RECIPES entry's raster at the observations.

    build_feature_grid() builds (or reuses the cache for) the recipe; here it
    is sampled exactly like a DERIVED aggregate or a static grid — the same
    code path postprocess.py's predict_maps() uses later, so there is no
    train/serve skew.
    """
    out: dict[str, np.ndarray] = {}
    for entry in RECIPES:
        name = entry["name"]
        path = build_feature_grid(config, name)
        points = gpd.GeoSeries(gpd.points_from_xy(metadata["x"], metadata["y"]), crs=crs)
        out[name] = sample_grids(points, {name: path})[name].to_numpy(dtype=np.float32)
    return out


def _assert_no_name_collisions(static_names: list[str]) -> None:
    """DERIVED and RECIPES names must not collide with each other or with a
    static grid stem — silently shadowing one is worse than a clear error."""
    derived_names = [derived_name(e["var"], e["stat"], e.get("months")) for e in DERIVED]
    recipe_names = [e["name"] for e in RECIPES]
    seen: dict[str, str] = {n: "static grid" for n in static_names}
    for label, names in (("DERIVED", derived_names), ("features.py RECIPES", recipe_names)):
        for name in names:
            if name in seen:
                raise ValueError(
                    f"name collision: '{name}' from {label} already exists as a {seen[name]} — "
                    "rename the recipe/aggregate or drop the duplicate"
                )
            seen[name] = label


def train_one(process: str, config, args) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = config.random_state
    features_dir = Path(args.features) / process
    X_full, y, metadata, feature_names, schema = load_artifact(features_dir)

    _assert_no_name_collisions(feature_names)

    for name, column in {
        **derived_columns(config, metadata, schema["crs"]),
        **recipe_columns(config, metadata, schema["crs"]),
    }.items():
        column = np.asarray(column, dtype=np.float32).reshape(-1)
        if column.shape[0] != X_full.shape[0]:
            raise ValueError(
                f"derived/recipe feature '{name}' has length {column.shape[0]}, "
                f"expected {X_full.shape[0]} (one value per observation)"
            )
        X_full = np.hstack([X_full, column.reshape(-1, 1)]).astype(np.float32)
        feature_names = feature_names + [name]

    selected = select_feature_names(feature_names, FEATURES)
    X = selected_matrix(X_full, feature_names, selected)

    # One CV pass on the fixed spatial K-folds; the mean PR-AUC over them is the
    # decision metric (val_pr_auc). Validation blocks are spatially disjoint
    # from training, so the score measures transfer to unseen terrain.
    folds = spatial_folds(y, metadata, config.k_folds, seed)
    fold_metrics: list[dict[str, float]] = []
    for fold_id, (train_idx, val_idx) in enumerate(folds):
        model = fit_ebm(
            X[train_idx], y[train_idx], selected, seed,
            f"{process} fold {fold_id + 1}/{len(folds)}",
        )
        m = scalar_metrics(y[val_idx], model.predict_proba(X[val_idx])[:, 1])
        m.update({"fold": fold_id, "n_train": int(len(train_idx)), "n_val": int(len(val_idx))})
        fold_metrics.append(m)
    pr_aucs = [m["pr_auc"] for m in fold_metrics if np.isfinite(m["pr_auc"])]
    roc_aucs = [m["roc_auc"] for m in fold_metrics if np.isfinite(m["roc_auc"])]
    pr_gain_aucs = [m["pr_gain_auc"] for m in fold_metrics if np.isfinite(m["pr_gain_auc"])]

    final_model = fit_ebm(X, y, selected, seed, f"{process} full fit")
    result = {
        "process": process,
        "pr_auc_mean": float(np.mean(pr_aucs)),
        "pr_auc_std": float(np.std(pr_aucs)),
        "roc_auc_mean": float(np.mean(roc_aucs)) if roc_aucs else float("nan"),
        "roc_auc_std": float(np.std(roc_aucs)) if roc_aucs else float("nan"),
        # Informational only (see prg_metric.py) -- not the decision metric.
        "pr_gain_auc_mean": float(np.mean(pr_gain_aucs)) if pr_gain_aucs else float("nan"),
        "pr_gain_auc_std": float(np.std(pr_gain_aucs)) if pr_gain_aucs else float("nan"),
        "folds": fold_metrics,
        "n_rows": int(len(y)),
        "n_positive": int(y.sum()),
        "n_features": int(len(selected)),
        "n_derived": len(DERIVED),
        "n_recipes": len(RECIPES),
        "features": selected,
        "importances": importances(final_model),
    }
    bundle_item = {
        "model": final_model,
        "features": selected,
        "schema": schema,
        "all_feature_names": feature_names,
        "metrics": result,
    }
    return result, bundle_item


def write_report(path: Path, output: dict[str, Any]) -> None:
    item = output["process_result"]
    lines = [
        "# Current Experiment",
        "",
        f"Process: {item['process']}",
        f"Decision val_pr_auc (spatial K-fold): {output['val_pr_auc']:.6f}",
        "",
        "## Spatial K-fold (decision metric)",
        f"- PR-AUC={item['pr_auc_mean']:.6f} "
        f"(+/- {item['pr_auc_std']:.6f} over {len(item['folds'])} folds), "
        f"ROC-AUC={item['roc_auc_mean']:.6f}, "
        f"AUPRG={item['pr_gain_auc_mean']:.6f} (informational, see prg_metric.py), "
        f"features={item['n_features']} (derived={item['n_derived']}, recipes={item['n_recipes']})",
        "",
        "## Rationale",
        output["rationale"].strip(),
    ]
    path.write_text("\n".join(lines))


def validate() -> None:
    if MODEL_CONFIG.get("outer_bags", 0) < 4:
        raise ValueError("outer_bags must be >= 4")
    if MODEL_CONFIG.get("learning_rate", 1.0) > 0.5:
        raise ValueError("learning_rate must be <= 0.5")
    if MODEL_CONFIG.get("validation_size", 0.0) <= 0:
        raise ValueError("validation_size must be positive")
    if not EXPERIMENT_RATIONALE.strip():
        raise ValueError("EXPERIMENT_RATIONALE must explain the hypothesis")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml", help="Optional override; defaults apply when absent.")
    parser.add_argument("--features", default="artifacts/features")
    parser.add_argument("--out", default="artifacts/run_current")
    args = parser.parse_args()

    config = load_config(args.config)
    processes = parse_processes(config, None)
    if len(processes) != 1:
        raise SystemExit(
            "Each research run targets exactly one hazard: set project.processes "
            "to a single key in this branch's config.yml "
            f"(found {processes})."
        )
    process = processes[0]

    _, _, _, feature_names, _ = load_artifact(Path(args.features) / process)
    validate()

    # Experiment identity: config + pipeline (rationale excluded) + features.py's
    # OWN source (rationale excluded) + the available feature space, so a
    # raster-generation code change alone — even with unchanged FEATURES/
    # DERIVED/MODEL_CONFIG and unchanged recipe names — is still a new
    # experiment, and regenerating data is a new experiment, but re-running a
    # logged one is mechanically rejected.
    features_py_text = Path("features.py").read_text().replace(FEATURE_RATIONALE, "")
    spec_hash = experiment_hash({
        "process": process,
        "features": FEATURES,
        "derived": DERIVED,
        "recipe_names": sorted(e["name"] for e in RECIPES),
        "model_config": MODEL_CONFIG,
        "pipeline": Path(__file__).read_text().replace(EXPERIMENT_RATIONALE, ""),
        "features_pipeline": features_py_text,
        "feature_space": sorted(feature_names),
    })
    assert_new_experiment(spec_hash)
    print(f"spec_hash: {spec_hash}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result, bundle_item = train_one(process, config, args)
    bundle = {
        "config_path": args.config,
        "feature_selection": FEATURES,
        "derived": DERIVED,
        "recipes": [e["name"] for e in RECIPES],
        "model_config": MODEL_CONFIG,
        "rationale": EXPERIMENT_RATIONALE,
        "processes": {process: bundle_item},
    }

    output = {
        "val_pr_auc": round(result["pr_auc_mean"], 6),
        "val_roc_auc": round(result["roc_auc_mean"], 6),
        "val_pr_gain_auc": round(result["pr_gain_auc_mean"], 6),
        "spec_hash": spec_hash,
        "process_result": result,
        "model_config": MODEL_CONFIG,
        "feature_selection": FEATURES,
        "derived": DERIVED,
        "rationale": EXPERIMENT_RATIONALE,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    with (out_dir / "model_bundle.pkl").open("wb") as handle:
        pickle.dump(bundle, handle)
    write_report(out_dir / "summary.md", output)

    try:
        write_explanations(bundle, out_dir / "explain.md")
    except Exception as exc:  # explanation must never take the run down
        print(f"[train.py] explain.md skipped: {exc}", file=sys.stderr)

    # val_roc_auc and val_pr_gain_auc are secondary/informational; the decision
    # metric val_pr_auc is the final line (program.md and loop.sh grep "^val_pr_auc:").
    print(f"val_roc_auc: {output['val_roc_auc']:.6f}")
    print(f"val_pr_gain_auc: {output['val_pr_gain_auc']:.6f}")
    print(f"val_pr_auc: {output['val_pr_auc']:.6f}")


if __name__ == "__main__":
    main()
