#!/usr/bin/env python3
"""Freeze dated events, generated non-events and spatial evaluation splits (run once per study).

Protocol after Moreno et al. (2025, JGR-ES, doi:10.1029/2024JF008219) and Steger et al.
(2026, NHESS 26:3637): non-events at random cells of the surveyed, non-trivial domain, at least
event_buffer_m from ANY inventoried event, one random date per location with year-month quotas
proportional to days, then one wet-day rule applied to BOTH classes. Non-events are
unrecorded events, not confirmed absences.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

from _runtime.guard import ROOT, artifacts, config, digest, file_hash, run_job, source, write_json


def study_identity(cfg):
    paths = [source(cfg, cfg["project"][k]) for k in ("dem", "events")]
    paths += [source(cfg, cfg["sampling"][k]) for k in ("surveyed_mask", "terrain_mask", "exclude_raster")
              if cfg["sampling"].get(k)]
    for spec in cfg["dynamic"]["sources"].values():
        found = sorted(glob.glob(str(source(cfg, spec["glob"]))))
        if not found:
            raise ValueError(f"No archives match {spec['glob']}")
        paths += map(Path, found)
    inputs = [{"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in paths]
    design = {k: cfg[k] for k in ("project", "sampling", "dynamic", "evaluation")}
    code = {n: file_hash(ROOT / n) for n in ("prepare.py", "_runtime/geo.py")}
    return digest({"design": design, "inputs": inputs, "code": code})


def study_dir(cfg):
    return artifacts(cfg) / cfg["project"]["process"] / study_identity(cfg)


def validate_design(cfg):
    s, d, e = cfg["sampling"], cfg["dynamic"], cfg["evaluation"]
    for value in (s["background_ratio"], s["event_buffer_m"], e["block_size_m"]):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Sampling ratio, spatial buffer and block size must be positive and finite")
    if type(d["max_history_days"]) is not int or not 1 <= d["max_history_days"] <= 365:
        raise ValueError("max_history_days must be an integer in 1..365")
    if type(e["folds"]) is not int or not 2 <= e["folds"] <= 10 or not 0 < e["holdout_fraction"] < 0.5:
        raise ValueError("Choose 2..10 folds and a holdout fraction between 0 and 0.5")
    if s.get("wet_days"):
        wet = s["wet_days"]
        if wet["variable"] not in d["sources"] or wet["stat"] not in {"sum", "max"}:
            raise ValueError("wet_days needs a configured precipitation source and sum/max statistic")
        if type(wet["days"]) is not int or type(wet["lag"]) is not int or wet["days"] < 1 or wet["lag"] < 0 or wet["days"] + wet["lag"] - 1 > d["max_history_days"]:
            raise ValueError("Invalid wet-day window: must fit inside max_history_days")
        if not math.isfinite(wet["threshold"]) or wet["threshold"] < 0:
            raise ValueError("Wet-day threshold must be finite and nonnegative, in the source's units")


def assign_splits(frame, cfg):
    """Square map blocks; all dates at a location stay together. Holdout blocks are never tuned on."""
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
    e, seed = cfg["evaluation"], cfg["project"]["seed"]
    frame = frame.reset_index(drop=True)
    frame["block"] = (np.floor(frame.x / e["block_size_m"]).astype(int).astype(str) + ":"
                      + np.floor(frame.y / e["block_size_m"]).astype(int).astype(str))
    if frame.block.nunique() < e["folds"] + 2:
        raise ValueError("Too few spatial blocks for CV plus holdout; review block size/extent")
    dev, hold = next(GroupShuffleSplit(1, test_size=e["holdout_fraction"], random_state=seed)
                     .split(frame, frame.label, frame.block))
    frame["fold"] = -1
    cv = StratifiedGroupKFold(e["folds"], shuffle=True, random_state=seed)
    for fold, (_, va) in enumerate(cv.split(frame.iloc[dev], frame.label.iloc[dev], frame.block.iloc[dev])):
        frame.loc[dev[va], "fold"] = fold
    for part in [frame.fold == -1, *(frame.fold == f for f in range(e["folds"]))]:
        if frame.label[part].nunique() != 2:
            raise ValueError("A fold or the holdout lacks one class; review block_size_m/folds/extent")
    return frame


def build_samples(cfg):
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import rasterio
    from scipy.spatial import cKDTree
    from _runtime.geo import archive_index, daily_history, mask_on_dem, reference

    validate_design(cfg)
    s, profile = cfg["sampling"], reference(cfg)
    rng = np.random.default_rng(cfg["project"]["seed"])
    h, w = profile["height"], profile["width"]
    events = gpd.read_file(source(cfg, cfg["project"]["events"]))
    if events.empty or events.crs is None or not events.geom_type.eq("Point").all() or events.geometry.is_empty.any():
        raise ValueError("Events must be Points with a declared CRS")
    events = events.to_crs(profile["crs"])
    dates = pd.to_datetime(events[cfg["project"]["date_column"]], errors="coerce", utc=True)
    dates = dates.dt.tz_convert(None).dt.normalize()
    xy = np.c_[events.geometry.x, events.geometry.y]
    if not np.isfinite(xy).all() or dates.notna().sum() < 10:
        raise ValueError("Need finite event coordinates and at least 10 valid event dates")
    rows, cols = map(np.asarray, rasterio.transform.rowcol(profile["transform"], xy[:, 0], xy[:, 1]))
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)

    # Sampling domain, applied identically to events and non-events.
    with rasterio.open(source(cfg, cfg["project"]["dem"])) as dem:
        values = dem.read(1, masked=True).astype("float32")
        domain = ~np.ma.getmaskarray(values) & np.isfinite(values.filled(np.nan))
    warnings = []
    for key in ("surveyed_mask", "terrain_mask"):
        if s.get(key):
            mask = mask_on_dem(source(cfg, s[key]), profile)
            domain &= ~np.ma.getmaskarray(mask) & (mask.filled(0) == 1)
        else:
            warnings.append(f"No {key}: exploratory domain, not a reproduction of the papers")
    if s.get("exclude_raster"):
        mask = mask_on_dem(source(cfg, s["exclude_raster"]), profile, s.get("exclude_crs"))
        domain &= ~np.ma.getmaskarray(mask) & ~np.isin(mask.filled(0), s.get("exclude_classes", []))

    # Calendar: days whose full history exists in every declared daily archive.
    history, wet = cfg["dynamic"]["max_history_days"], s.get("wet_days")
    same_day = cfg["dynamic"]["same_day"] or bool(wet and wet["lag"] == 0)
    start, end = pd.Timestamp(s.get("start_date") or dates.min()), pd.Timestamp(s.get("end_date") or dates.max())
    if end < start or (end - start).days > 365 * 150:
        raise ValueError("Invalid or excessively long observation period")
    calendar = pd.date_range(start, end, freq="D")
    if not len(calendar):
        raise ValueError("No days in the configured observation period")
    indexes = {name: archive_index(cfg, name) for name in cfg["dynamic"]["sources"]}
    for entries in indexes.values():
        have = pd.DatetimeIndex(sorted(d for _, _, days in entries for d in days))
        present = pd.Series(1, index=have).reindex(
            pd.date_range(calendar[0] - pd.Timedelta(days=history), calendar[-1]), fill_value=0)
        full = present.shift(1, fill_value=0).rolling(history).sum().eq(history)
        if same_day:
            full &= present.eq(1)
        calendar = calendar[full.reindex(calendar).to_numpy()]
        if not len(calendar):
            raise ValueError("No dates with complete dynamic histories in the inventory period")
    if not len(calendar):
        raise ValueError("No dates with complete dynamic histories in the inventory period")

    ok = inside.copy()
    ok[inside] &= domain[rows[inside], cols[inside]]
    ok &= dates.isin(calendar).to_numpy()
    cases = pd.DataFrame({"x": xy[ok, 0], "y": xy[ok, 1], "date": dates[ok].to_numpy(), "label": 1,
                          "cell": rows[ok] * w + cols[ok]}).drop_duplicates(["cell", "date"])
    if len(cases) < 10:
        raise ValueError("Fewer than 10 dated events remain inside the domain and period")
    target = int(np.ceil(len(cases) * s["background_ratio"]))
    if len(cases) + target > cfg["limits"].get("max_samples", 100_000):
        raise ValueError("Requested samples exceed limits.max_samples")

    # Non-event cells: in domain, >= buffer from ALL events (also undated/out-of-period), unique.
    rr, cc = np.where(domain)
    cand = np.c_[rasterio.transform.xy(profile["transform"], rr, cc)]
    keep = cKDTree(xy[np.isfinite(xy).all(1)]).query(cand)[0] >= s["event_buffer_m"]
    keep &= ~np.isin(rr * w + cc, rows[inside] * w + cols[inside])
    cand, cells = cand[keep], (rr * w + cc)[keep]
    if target > len(cells):
        raise ValueError("Fewer candidate cells than requested non-events; reduce background_ratio")
    pools = [g.to_numpy() for _, g in pd.Series(calendar).groupby(calendar.to_period("M"))]
    quota = target * np.array([len(p) for p in pools]) / len(calendar)
    counts = np.floor(quota).astype(int)
    counts[np.lexsort((rng.random(len(pools)), counts - quota))[:target - counts.sum()]] += 1
    days = np.concatenate([rng.choice(p, n) for p, n in zip(pools, counts) if n])
    rng.shuffle(days)
    pick = rng.choice(len(cells), target, replace=False)
    controls = pd.DataFrame({"x": cand[pick, 0], "y": cand[pick, 1], "date": pd.to_datetime(days),
                             "label": 0, "cell": cells[pick]})
    frame = pd.concat([cases, controls], ignore_index=True)
    before = frame.label.value_counts().to_dict()

    # Same coverage and wet-day rule for both classes (Moreno: >1 mm in 72 h; Steger: >=10 mm).
    for name in indexes:
        print(f"Checking {name} coverage for {len(frame)} observations", flush=True)
        hist = daily_history(cfg, frame, name, history, indexes)
        frame = frame[np.isfinite(hist[:, 0 if same_day else 1:]).all(1)].reset_index(drop=True)
    if wet:
        hist = daily_history(cfg, frame, wet["variable"], wet["days"] + wet["lag"] - 1, indexes)
        window = hist[:, wet["lag"]:wet["lag"] + wet["days"]]
        frame = frame[getattr(np, wet["stat"])(window, axis=1) >= wet["threshold"]].reset_index(drop=True)
        warnings.append("Wet-day rule: results are conditional on wet antecedent conditions")
    frame = assign_splits(frame.iloc[rng.permutation(len(frame))].reset_index(drop=True), cfg)
    frame.insert(0, "sample_id", np.arange(len(frame)))
    report = {"input_events": len(events), "undated_events": int(dates.isna().sum()),
              "counts_sampled": before, "counts_final": frame.label.value_counts().to_dict(),
              "eligible_days": len(calendar), "start": str(calendar[0].date()), "end": str(calendar[-1].date()),
              "control_month_quota": {str(p[0])[:7]: int(n) for p, n in zip(pools, counts)},
              "control_month_counts_final": frame.loc[frame.label == 0, "date"].dt.to_period("M").astype(str).value_counts().sort_index().to_dict(),
              "warnings": warnings}
    return frame, report


def verify_prepared(cfg):
    path = study_dir(cfg) / "prepared"
    if not path.exists():
        raise FileNotFoundError("Run python prepare.py first")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["study_id"] != study_identity(cfg) or file_hash(path / "samples.csv") != manifest["samples_sha256"] or file_hash(path / "points.csv") != manifest["points_sha256"]:
        raise ValueError("Frozen samples were modified")
    return path, manifest


def main(default_run=False):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "config.yml"))
    parser.add_argument("--run", action="store_true", default=default_run)
    parser.add_argument("--train", default=str(ROOT / "train.py"), help="editable experiment source")
    parser.add_argument("--description", default="baseline", help="hypothesis / reason for this experiment")
    parser.add_argument("--retry", action="store_true", help="explicitly repeat an identical experiment")
    parser.add_argument("--list", action="store_true", help="show experiment and command history")
    parser.add_argument("--usage", action="store_true", help="show shared generated-data usage")
    parser.add_argument("--holdout", action="store_true", help="human-only final evaluation; closes the study")
    parser.add_argument("--experiment-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = config(args.config)
    validate_design(cfg)
    if args.usage:
        from _runtime.guard import tree_bytes
        print(f"artifacts/: {tree_bytes(artifacts(cfg)) / 1e9:.3f} / {cfg['limits']['disk_gb']} GB")
        return
    if args.run or args.list or args.holdout or args.experiment_worker:
        from _runtime.engine import dispatch
        dispatch(cfg, args)
        return
    if args.worker:
        identity = study_identity(cfg)
        frame, report = build_samples(cfg)
        frame.to_csv("samples.csv", index=False, date_format="%Y-%m-%d")
        # What the agent sees: no labels, folds or blocks.
        frame[["sample_id", "x", "y", "date"]].to_csv("points.csv", index=False, date_format="%Y-%m-%d")
        if study_identity(cfg) != identity:
            raise RuntimeError("Inputs or preparation code changed during sampling; retry")
        write_json("manifest.json", {"study_id": identity, "samples_sha256": file_hash("samples.csv"),
                                     "points_sha256": file_hash("points.csv"),
                                     "design": {k: cfg[k] for k in ("project", "sampling", "dynamic", "evaluation")}})
        write_json("sampling_report.json", report)
        return
    path = study_dir(cfg) / "prepared"
    if path.exists():
        verify_prepared(cfg)
    else:
        run_job(cfg, "prepare.py", path)
    report = json.loads((path / "sampling_report.json").read_text())
    print(f"Samples: {path}\nFinal counts {report['counts_final']} ({report['start']} to {report['end']})")
    print("\n".join(f"Warning: {w}" for w in report["warnings"]))


if __name__ == "__main__":
    main()
