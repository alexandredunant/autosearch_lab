#!/usr/bin/env python3
"""Manual, post-loop reporting: susceptibility maps + the search-progress plot.

Not part of the fixed-infra "never edit" contract in program.md — this is a
separate tool you run yourself, after (or during) a loop.sh run, to look at
results. It imports prepare.py's internals rather than duplicating them.

Usage:
  python postprocess.py --predict-map [--model PATH] [--maps-out DIR]
  python postprocess.py --plot-progress
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy as transform_xy

from prepare import (
    LEDGER_PATH,
    build_feature_grid,
    derived_grid,
    derived_name,
    discover_grids,
    encode_features,
    load_config,
    reference_grid,
    selected_matrix,
    valid_domain,
)


def predict_grid(
    model,
    selected_features: list[str],
    grids: dict[str, Path],
    grids_dir: Path,
    out_path: Path,
    *,
    block_rows: int = 256,
) -> None:
    """Predict susceptibility over the reference grid and write a GeoTIFF."""
    mask, transform, crs = valid_domain(grids, grids_dir)
    ref = reference_grid(grids)
    with rasterio.open(ref) as src:
        height, width = src.height, src.width
    result = np.full((height, width), np.nan, dtype=np.float32)

    readers = {name: rasterio.open(path) for name, path in grids.items()}
    try:
        for row0 in range(0, height, block_rows):
            rows = slice(row0, min(row0 + block_rows, height))
            block_mask = mask[rows]
            if not block_mask.any():
                continue
            r_idx, c_idx = np.nonzero(block_mask)
            xs, ys = transform_xy(transform, r_idx + row0, c_idx, offset="center")
            sampled: dict[str, np.ndarray] = {}
            for name, reader in readers.items():
                pts = gpd.GeoSeries(gpd.points_from_xy(xs, ys), crs=crs).to_crs(reader.crs)
                values = np.array(
                    [v[0] for v in reader.sample([(g.x, g.y) for g in pts])], dtype=np.float64
                )
                if reader.nodata is not None:
                    values[values == reader.nodata] = np.nan
                sampled[name] = values
            encoded = encode_features(pd.DataFrame(sampled))
            # strict=False: one-hot levels absent from this block become zero
            # columns, keeping the column order the model was trained on.
            X = selected_matrix(
                np.nan_to_num(encoded.to_numpy(dtype=np.float32)),
                encoded.columns.tolist(),
                selected_features,
                strict=False,
            )
            prob = model.predict_proba(X)[:, 1].astype(np.float32)
            block = result[rows]
            block[r_idx, c_idx] = prob
            result[rows] = block
    finally:
        for reader in readers.values():
            reader.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": 1,
        "dtype": "float32", "crs": crs, "transform": transform,
        "nodata": np.nan, "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(result, 1)


def predict_maps(config, model_path: Path, out_dir: Path) -> None:
    """Export susceptibility_<key>.tif for every process in the model bundle."""
    grids = discover_grids(config.grids_dir)
    with model_path.open("rb") as handle:
        bundle = pickle.load(handle)
    # The bundle's derived aggregates and features.py recipes are ordinary
    # grids at prediction time: recomputed (or read from cache) and sampled
    # by the same code path used at training time — no train/serve skew.
    for entry in bundle.get("derived", []):
        name = derived_name(entry["var"], entry["stat"], entry.get("months"))
        grids[name] = derived_grid(config, entry["var"], entry["stat"], entry.get("months"))
    for name in bundle.get("recipes", []):
        grids[name] = build_feature_grid(config, name)
    for process, item in bundle["processes"].items():
        out_path = out_dir / f"susceptibility_{process}.tif"
        predict_grid(item["model"], item["features"], grids, config.grids_dir, out_path)
        print(f"Wrote {out_path}")


def plot_progress(tsv: Path, out: Path) -> None:
    """Search trajectory from the experiment ledger (karpathy-style progress.png)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = pd.read_csv(tsv, sep="\t")
    if rows.empty:
        raise SystemExit(f"No experiments logged yet in {tsv}")
    rows["iteration"] = rows["iteration"].astype(int)
    rows = rows.sort_values("iteration")

    best_iter, best_val = [], []
    best = float("-inf")
    for _, row in rows.iterrows():
        if row["status"] in ("keep", "baseline") and row["val_pr_auc"] > best:
            best = float(row["val_pr_auc"])
            best_iter.append(int(row["iteration"]))
            best_val.append(best)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step([*best_iter, rows["iteration"].max()], [*best_val, best_val[-1]],
            where="post", color="#0072B2", lw=2, label="best so far", zorder=2)
    for status, marker, color, label in [
        ("keep", "o", "#009E73", "kept"),
        ("discard", "x", "#D55E00", "discarded"),
        ("crash", "x", "#999999", "crashed"),
        ("baseline", "s", "#999999", "null baseline"),
    ]:
        sub = rows[rows["status"] == status]
        if len(sub):
            ax.scatter(sub["iteration"], sub["val_pr_auc"], marker=marker,
                       color=color, s=45, label=label, zorder=3)
    ax.set_xlabel("experiment iteration")
    ax.set_ylabel("val_pr_auc")
    ax.set_title(f"autoresearch progress — best {best_val[-1]:.4f} "
                 f"({int((rows['status'] == 'keep').sum())} kept / {len(rows)} logged)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="config.yml", help="Optional override; defaults apply when absent.")
    parser.add_argument("--predict-map", action="store_true",
                        help="Export susceptibility GeoTIFFs from --model and exit.")
    parser.add_argument("--model", default="artifacts/models/best_model.pkl")
    parser.add_argument("--maps-out", default="artifacts/maps")
    parser.add_argument("--plot-progress", action="store_true",
                        help="Draw progress.png from the experiment ledger and exit.")
    args = parser.parse_args()

    if args.predict_map:
        predict_maps(load_config(args.config), Path(args.model), Path(args.maps_out))
        return
    if args.plot_progress:
        plot_progress(LEDGER_PATH, Path("progress.png"))
        return
    parser.print_help()


if __name__ == "__main__":
    main()
