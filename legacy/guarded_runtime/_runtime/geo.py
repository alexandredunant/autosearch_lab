"""Optional bounded geospatial helpers; the agent owns feature engineering in train.py."""
from __future__ import annotations

import glob
from pathlib import Path

from .guard import source

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
import xarray as xr
from pyproj import CRS, Transformer



def reference(cfg):
    """Profile of the reference DEM (projected, metres, north-up, within limits.max_cells)."""
    with rasterio.open(source(cfg, cfg["project"]["dem"])) as src:
        crs = CRS(src.crs)
        if not crs.is_projected or any(abs(a.unit_conversion_factor - 1) > 1e-9 for a in crs.axis_info):
            raise ValueError("DEM must use a projected CRS in metres")
        if src.count != 1 or src.width * src.height > cfg["limits"]["max_cells"]:
            raise ValueError("DEM must be single-band and within limits.max_cells")
        if src.transform.b or src.transform.d or src.transform.e >= 0 or src.transform.a <= 0:
            raise ValueError("DEM must be a north-up regular grid")
        return src.profile


def sample_raster(path, frame, crs, categorical=False):
    """Values of any georeferenced single-band raster (GeoTIFF, SAGA .sdat, ...) at frame.x/y."""
    with rasterio.open(path) as src:
        if src.crs is None or src.count != 1:
            raise ValueError("Predictor raster needs one band and a declared CRS")
        with WarpedVRT(src, crs=crs, resampling=Resampling.nearest if categorical else Resampling.bilinear) as vrt:
            values = list(vrt.sample(zip(frame.x, frame.y), masked=True))
    return np.ma.asarray(values, dtype="float32").filled(np.nan).reshape(-1)


def mask_on_dem(path, profile, src_crs=None):
    with rasterio.open(path) as src, WarpedVRT(src, src_crs=src_crs or src.crs, crs=profile["crs"],
                                               transform=profile["transform"], width=profile["width"],
                                               height=profile["height"], resampling=Resampling.nearest) as vrt:
        return vrt.read(1, masked=True)


def _same_grid_crs(ds, a, declared):
    """Older CF files omit datum names or use a zero-offset BoundCRS; accept equal parameters."""
    mapping = a.attrs.get("grid_mapping")
    if not (mapping and mapping in ds and "grid_mapping_name" in ds[mapping].attrs):
        return True
    actual = CRS.from_cf(ds[mapping].attrs)
    if actual.is_bound and all(p.value == 0 for p in actual.coordinate_operation.params):
        actual = actual.source_crs
    declared = CRS(declared)
    if actual.equals(declared, ignore_axis_order=True):
        return True
    ops = [c.coordinate_operation for c in (actual, declared)]
    return (None not in ops and ops[0].method_code == ops[1].method_code
            and {p.code: p.value for p in ops[0].params} == {p.code: p.value for p in ops[1].params}
            and np.isclose(actual.ellipsoid.semi_major_metre, declared.ellipsoid.semi_major_metre)
            and np.isclose(actual.ellipsoid.inverse_flattening, declared.ellipsoid.inverse_flattening))


def archive_index(cfg, name):
    """[(file, time dimension, days)] for one daily (time, y, x) source declared in config.yml."""
    spec = cfg["dynamic"]["sources"][name]
    files = sorted(Path(p) for p in glob.glob(str(source(cfg, spec["glob"]))))
    if not files:
        raise ValueError(f"No NetCDF files for {name}")
    entries, seen = [], set()
    for path in files:
        with xr.open_dataset(path, cache=False) as ds:
            a = ds[spec.get("variable", name)]
            td = [d for d in a.dims if d not in {"x", "y"}]
            if len(td) != 1 or set(a.dims) != {td[0], "y", "x"}:
                raise ValueError(f"Expected (time, y, x) daily data: {path}")
            days = pd.DatetimeIndex(ds[td[0]].values).normalize()
            if not len(days) or days.isna().any() or days.duplicated().any() or not days.is_monotonic_increasing or seen & set(days):
                raise ValueError(f"Duplicate, unordered or overlapping days in {name}: {path}")
            seen.update(days)
            if not _same_grid_crs(ds, a, spec["crs"]):
                raise ValueError(f"Configured CRS disagrees with NetCDF CRS: {path}")
            for name_axis in ("x", "y"):
                axis = ds[name_axis].values
                delta = np.diff(axis)
                if len(axis) < 2 or not np.isfinite(axis).all() or not (np.all(delta > 0) or np.all(delta < 0)) or not np.allclose(delta, delta[0]):
                    raise ValueError(f"Expected regular monotone {name_axis} coordinates: {path}")
            entries.append((path, td[0], days))
    return entries


def daily_history(cfg, frame, name, days, indexes=None):
    """Array (n_points, days + 1): column k = value k days BEFORE each point's date (0 = same day).

    Missing days, nodata and points outside the archive stay NaN (missing rain is not 0).
    Use columns >= 1 for predictors unless the study deliberately allows same-day data.
    """
    spec = cfg["dynamic"]["sources"][name]
    if type(days) is not int or not 0 <= days <= cfg["dynamic"]["max_history_days"]:
        raise ValueError("History length must be an integer within dynamic.max_history_days")
    if len(frame) > cfg["limits"].get("max_samples", 100_000):
        raise ValueError("Point histories exceed limits.max_samples")
    entries = indexes[name] if indexes else archive_index(cfg, name)
    dates = pd.DatetimeIndex(frame.date).normalize().values.astype("datetime64[D]")
    px, py = Transformer.from_crs(reference(cfg)["crs"], spec["crs"], always_xy=True).transform(
        frame.x.to_numpy(), frame.y.to_numpy())
    hist = np.full((len(frame), days + 1), np.nan, dtype="float32")
    for path, td, stamps in entries:
        day = stamps.values.astype("datetime64[D]")
        rows = np.flatnonzero((dates >= day.min()) & (dates - np.timedelta64(days, "D") <= day.max()))
        if not len(rows):
            continue
        with xr.open_dataset(path, cache=False) as ds:
            ix = np.rint((px - ds.x.values[0]) / (ds.x.values[1] - ds.x.values[0])).astype(int)
            iy = np.rint((py - ds.y.values[0]) / (ds.y.values[1] - ds.y.values[0])).astype(int)
            rows = rows[(ix[rows] >= 0) & (ix[rows] < ds.sizes["x"]) & (iy[rows] >= 0) & (iy[rows] < ds.sizes["y"])]
            for start in range(0, len(rows), 128):  # bounded in both space AND time (even for one large file)
                r = rows[start:start + 128]
                for t in range(0, len(day), 31):
                    offset = (dates[r, None] - day[None, t:t + 31]).astype(int)
                    ri, ti = np.where((offset >= 0) & (offset <= days))
                    if not len(ri):
                        continue
                    vals = ds[spec.get("variable", name)].isel({td: slice(t, t + 31),
                        "x": xr.DataArray(ix[r], dims="point"), "y": xr.DataArray(iy[r], dims="point")}).transpose("point", td).values
                    hist[r[ri], offset[ri, ti]] = vals[ri, ti]
    return hist



