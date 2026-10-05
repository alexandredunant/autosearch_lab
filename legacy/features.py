#!/usr/bin/env python3
# Archived workflow; use ../features.py for new studies.
"""THE feature-engineering file. This and train.py are the only files the
autoresearch agent edits.

Each entry in RECIPES proposes ONE candidate predictor raster, built from the
raw primitives in data/ (dem.tif, dynamic/<var>/*.nc, vectors/*.gpkg|*.shp|*.tif)
using whatever operation the experiment calls for — a GDAL CLI tool
(gdaldem, gdal_proximity.py, gdalwarp, gdal_rasterize, ...), saga_cmd (SAGA
GIS — fast and robust for terrain analysis (`ta_compound 0`) and distance
transforms (`grid_tools 26`); prefer it over reimplementing one by hand — a
whole-DEM terrain-compound pass takes only a few seconds, well inside this
recipe's own timeout — via plain `subprocess.run(cmd, timeout=N,
cwd=ctx.scratch_dir, check=True)`), rasterio, numpy, geopandas, scipy.ndimage,
xarray, or any combination. There is no fixed menu: unlike the old
DERIVED-only design, this file is real code.

Contract
--------
def build_<name>(ctx: FeatureContext) -> np.ndarray | Path:
    ...

RECIPES: list[dict] = [
    {"name": "<feature_name>", "build": build_<name>},
]

`ctx` (see prepare.FeatureContext, and program.md's "Paths available to
build_<name>(ctx) recipes") is plain data, no methods — exposes exactly:
  ctx.dem_path            Path to the reference grid (dem.tif)
  ctx.dynamic_dir         Path to data/dynamic/ (one subdirectory per variable)
  ctx.vectors_dir         Path to data/vectors/ (whatever layers actually exist
                          — run --describe-primitives, don't assume names)
  ctx.scratch_dir         Path, wiped before every run — throwaway intermediates only
  ctx.reference_profile   dict: epsg, transform, crs, height, width — what the
                          returned raster MUST match

Return either:
  - an np.ndarray of shape (reference_profile["height"], reference_profile["width"]):
    prepare.py writes the GeoTIFF itself from the reference profile, so
    misalignment is structurally impossible. Use NaN for nodata, never a
    sentinel like -9999.
  - a Path to a GeoTIFF the recipe already wrote (for GDAL-CLI chains that
    naturally emit files): prepare.py validates it strictly against the
    reference grid (single-band, same EPSG, same transform within 1e-6, same
    shape) and refuses — no silent reprojection — if it doesn't match.

What NOT to do
--------------
- Never import or reference data/points_<hazard>.geojson, or prepare's
  load_points/build_sample/sample_absences/spatial_folds — prepare.py's
  leakage guard (scan_features_module) refuses to execute a features.py that
  does, and it is a Hard Rule in program.md regardless.
- Never touch the sampling domain, spatial folds, metric, or ledger — those
  stay in prepare.py, exactly as train.py already can't touch them.
- A cached raster over 2x dem.tif's file size, or one that would push
  artifacts/features_cache/ over 15GB total, is refused (crash) rather than
  silently written — see build_feature_grid() in prepare.py.

Caching
-------
Every recipe's output is cached under artifacts/features_cache/<name>/<hash>.tif,
keyed on this file's own full source text plus a fingerprint of the raw data.
Any edit here — even to one recipe — invalidates every cached name. Once
cached, a recipe is NEVER recomputed: delete the cache directory, or edit this
file, to force a rebuild. `python prepare.py --prune-feature-cache` reclaims
space from recipes no longer listed below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

FEATURE_RATIONALE = """
Null baseline: no recipes. Establishes the reference val_pr_auc alongside
train.py's empty FEATURES/DERIVED (see program.md). Logged with --status
baseline. The build_* functions below are inert reference implementations
(not registered in RECIPES) kept from prior experiments -- copy from them
freely, but this run tests the new antecedent dynamic-archive features and
search-order rule from a clean slate.
"""


def build_slope_degrees(ctx) -> np.ndarray:
    """Terrain slope angle in degrees, with DEM nodata propagated as NaN."""
    import rasterio

    with rasterio.open(ctx.dem_path) as src:
        dem = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        transform = src.transform
    dy, dx = np.gradient(dem, abs(transform.e), abs(transform.a))
    slope = np.degrees(np.arctan(np.hypot(dx, dy))).astype(np.float32)
    slope[~np.isfinite(dem)] = np.nan
    return slope


def build_northness(ctx) -> np.ndarray:
    """Cosine aspect: +1 north-facing, -1 south-facing, NaN at DEM nodata."""
    import rasterio

    with rasterio.open(ctx.dem_path) as src:
        dem = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        transform = src.transform
    dy, dx = np.gradient(dem, abs(transform.e), abs(transform.a))
    aspect = np.arctan2(-dx, dy)
    northness = np.cos(aspect).astype(np.float32)
    northness[~np.isfinite(dem)] = np.nan
    return northness


def build_eastness(ctx) -> np.ndarray:
    """Sine aspect: +1 east-facing, -1 west-facing, NaN at DEM nodata."""
    import rasterio

    with rasterio.open(ctx.dem_path) as src:
        dem = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        transform = src.transform
    dy, dx = np.gradient(dem, abs(transform.e), abs(transform.a))
    aspect = np.arctan2(-dx, dy)
    eastness = np.sin(aspect).astype(np.float32)
    eastness[~np.isfinite(dem)] = np.nan
    return eastness


def build_distance_to_roads_m(ctx) -> np.ndarray:
    """Euclidean distance in metres to the nearest mapped road segment."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from scipy.ndimage import distance_transform_edt

    with rasterio.open(ctx.dem_path) as src:
        profile = src.profile
        valid = ~src.read(1, masked=True).mask
    roads = gpd.read_file(ctx.vectors_dir / "roads.gpkg").to_crs(profile["crs"])
    road_mask = rasterize(
        ((geom, 1) for geom in roads.geometry if geom is not None and not geom.is_empty),
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"], fill=0, dtype="uint8", all_touched=True,
    ).astype(bool)
    distance = distance_transform_edt(~road_mask, sampling=(abs(profile["transform"].e), abs(profile["transform"].a)))
    distance = distance.astype(np.float32)
    distance[~valid] = np.nan
    return distance


def build_distance_to_buildings_m(ctx) -> np.ndarray:
    """Euclidean distance in metres to the nearest mapped building point."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from scipy.ndimage import distance_transform_edt

    with rasterio.open(ctx.dem_path) as src:
        profile = src.profile
        valid = ~src.read(1, masked=True).mask
    buildings = gpd.read_file(ctx.vectors_dir / "buildings.gpkg").to_crs(profile["crs"])
    building_mask = rasterize(
        ((geom, 1) for geom in buildings.geometry if geom is not None and not geom.is_empty),
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"], fill=0, dtype="uint8", all_touched=True,
    ).astype(bool)
    distance = distance_transform_edt(
        ~building_mask, sampling=(abs(profile["transform"].e), abs(profile["transform"].a))
    ).astype(np.float32)
    distance[~valid] = np.nan
    return distance


def build_distance_to_electrical_infrastructure_m(ctx) -> np.ndarray:
    """Euclidean distance in metres to mapped electrical line infrastructure."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from scipy.ndimage import distance_transform_edt

    with rasterio.open(ctx.dem_path) as src:
        profile = src.profile
        valid = ~src.read(1, masked=True).mask
    lines = gpd.read_file(ctx.vectors_dir / "electrical_infrastructure.shp")
    # The source lacks a .prj, but its coordinates lie in the reference grid's
    # projected extent; assign that documented grid CRS before rasterization.
    if lines.crs is None:
        lines = lines.set_crs(profile["crs"])
    else:
        lines = lines.to_crs(profile["crs"])
    line_mask = rasterize(
        ((geom, 1) for geom in lines.geometry if geom is not None and not geom.is_empty),
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"], fill=0, dtype="uint8", all_touched=True,
    ).astype(bool)
    distance = distance_transform_edt(
        ~line_mask, sampling=(abs(profile["transform"].e), abs(profile["transform"].a))
    ).astype(np.float32)
    distance[~valid] = np.nan
    return distance


def build_forest_cover(ctx) -> np.ndarray:
    """Binary broadleaf/conifer/mixed forest cover, nearest-reprojected to DEM."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(ctx.dem_path) as dem_src:
        profile = dem_src.profile
        valid = ~dem_src.read(1, masked=True).mask
    with rasterio.open(ctx.vectors_dir / "CorineLandCover_polygon.tif") as land_src:
        codes = np.zeros((profile["height"], profile["width"]), dtype=np.int16)
        reproject(
            source=rasterio.band(land_src, 1), destination=codes,
            src_transform=land_src.transform, src_crs=land_src.crs,
            dst_transform=profile["transform"], dst_crs=profile["crs"],
            resampling=Resampling.nearest,
        )
    forest = np.isin(codes, (311, 312, 313)).astype(np.float32)
    forest[~valid] = np.nan
    return forest


def build_open_natural_cover(ctx) -> np.ndarray:
    """Binary natural grassland, heath, shrub, or sparse-vegetation cover."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(ctx.dem_path) as dem_src:
        profile = dem_src.profile
        valid = ~dem_src.read(1, masked=True).mask
    with rasterio.open(ctx.vectors_dir / "CorineLandCover_polygon.tif") as land_src:
        codes = np.zeros((profile["height"], profile["width"]), dtype=np.int16)
        reproject(
            source=rasterio.band(land_src, 1), destination=codes,
            src_transform=land_src.transform, src_crs=land_src.crs,
            dst_transform=profile["transform"], dst_crs=profile["crs"],
            resampling=Resampling.nearest,
        )
    open_cover = np.isin(codes, (321, 322, 324, 333)).astype(np.float32)
    open_cover[~valid] = np.nan
    return open_cover


def build_managed_agriculture_cover(ctx) -> np.ndarray:
    """Binary Corine cropland, pasture, plantation, or mosaic-agriculture cover."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(ctx.dem_path) as dem_src:
        profile = dem_src.profile
        valid = ~dem_src.read(1, masked=True).mask
    with rasterio.open(ctx.vectors_dir / "CorineLandCover_polygon.tif") as land_src:
        codes = np.zeros((profile["height"], profile["width"]), dtype=np.int16)
        reproject(
            source=rasterio.band(land_src, 1), destination=codes,
            src_transform=land_src.transform, src_crs=land_src.crs,
            dst_transform=profile["transform"], dst_crs=profile["crs"],
            resampling=Resampling.nearest,
        )
    managed = np.isin(codes, (211, 221, 222, 242, 243)).astype(np.float32)
    managed[~valid] = np.nan
    return managed


def build_wetland_cover(ctx) -> np.ndarray:
    """Binary Corine inland/coastal wetland cover, nearest-reprojected to DEM."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(ctx.dem_path) as dem_src:
        profile = dem_src.profile
        valid = ~dem_src.read(1, masked=True).mask
    with rasterio.open(ctx.vectors_dir / "CorineLandCover_polygon.tif") as land_src:
        codes = np.zeros((profile["height"], profile["width"]), dtype=np.int16)
        reproject(
            source=rasterio.band(land_src, 1), destination=codes,
            src_transform=land_src.transform, src_crs=land_src.crs,
            dst_transform=profile["transform"], dst_crs=profile["crs"],
            resampling=Resampling.nearest,
        )
    wetland = np.isin(codes, (411, 412, 421, 422, 423)).astype(np.float32)
    wetland[~valid] = np.nan
    return wetland


def build_artificial_cover(ctx) -> np.ndarray:
    """Binary Corine artificial-surface cover, including urban and extractive land."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(ctx.dem_path) as dem_src:
        profile = dem_src.profile
        valid = ~dem_src.read(1, masked=True).mask
    with rasterio.open(ctx.vectors_dir / "CorineLandCover_polygon.tif") as land_src:
        codes = np.zeros((profile["height"], profile["width"]), dtype=np.int16)
        reproject(source=rasterio.band(land_src, 1), destination=codes,
                  src_transform=land_src.transform, src_crs=land_src.crs,
                  dst_transform=profile["transform"], dst_crs=profile["crs"],
                  resampling=Resampling.nearest)
    artificial = np.isin(codes, (111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142)).astype(np.float32)
    artificial[~valid] = np.nan
    return artificial


def build_pasture_cover(ctx) -> np.ndarray:
    """Binary Corine permanent-pasture cover, separate from cropland mosaics."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(ctx.dem_path) as dem_src:
        profile = dem_src.profile
        valid = ~dem_src.read(1, masked=True).mask
    with rasterio.open(ctx.vectors_dir / "CorineLandCover_polygon.tif") as land_src:
        codes = np.zeros((profile["height"], profile["width"]), dtype=np.int16)
        reproject(source=rasterio.band(land_src, 1), destination=codes,
                  src_transform=land_src.transform, src_crs=land_src.crs,
                  dst_transform=profile["transform"], dst_crs=profile["crs"],
                  resampling=Resampling.nearest)
    pasture = (codes == 231).astype(np.float32)
    pasture[~valid] = np.nan
    return pasture


def build_slope_variability_1000m(ctx) -> np.ndarray:
    """Local standard deviation of slope over an approximately 1-km window."""
    from scipy.ndimage import uniform_filter
    import rasterio

    with rasterio.open(ctx.dem_path) as src:
        dem = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        transform = src.transform
    dy, dx = np.gradient(dem, abs(transform.e), abs(transform.a))
    slope = np.degrees(np.arctan(np.hypot(dx, dy))).astype(np.float32)
    finite = np.isfinite(slope)
    size = max(3, int(round(1000.0 / abs(transform.a))))
    weight = uniform_filter(finite.astype(np.float32), size=size, mode="nearest")
    mean = uniform_filter(np.where(finite, slope, 0.0), size=size, mode="nearest") / np.maximum(weight, 1e-6)
    mean_sq = uniform_filter(np.where(finite, slope * slope, 0.0), size=size, mode="nearest") / np.maximum(weight, 1e-6)
    variability = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)).astype(np.float32)
    variability[~finite] = np.nan
    return variability


def build_slope_variability_500m(ctx) -> np.ndarray:
    """Local standard deviation of slope over an approximately 500-m window."""
    from scipy.ndimage import uniform_filter
    import rasterio

    with rasterio.open(ctx.dem_path) as src:
        dem = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        transform = src.transform
    dy, dx = np.gradient(dem, abs(transform.e), abs(transform.a))
    slope = np.degrees(np.arctan(np.hypot(dx, dy))).astype(np.float32)
    finite = np.isfinite(slope)
    size = max(3, int(round(500.0 / abs(transform.a))))
    weight = uniform_filter(finite.astype(np.float32), size=size, mode="nearest")
    mean = uniform_filter(np.where(finite, slope, 0.0), size=size, mode="nearest") / np.maximum(weight, 1e-6)
    mean_sq = uniform_filter(np.where(finite, slope * slope, 0.0), size=size, mode="nearest") / np.maximum(weight, 1e-6)
    variability = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)).astype(np.float32)
    variability[~finite] = np.nan
    return variability


def build_slope_variability_250m(ctx) -> np.ndarray:
    """Local standard deviation of slope over an approximately 250-m window."""
    from scipy.ndimage import uniform_filter
    import rasterio

    with rasterio.open(ctx.dem_path) as src:
        dem = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        transform = src.transform
    dy, dx = np.gradient(dem, abs(transform.e), abs(transform.a))
    slope = np.degrees(np.arctan(np.hypot(dx, dy))).astype(np.float32)
    finite = np.isfinite(slope)
    size = max(3, int(round(250.0 / abs(transform.a))))
    weight = uniform_filter(finite.astype(np.float32), size=size, mode="nearest")
    mean = uniform_filter(np.where(finite, slope, 0.0), size=size, mode="nearest") / np.maximum(weight, 1e-6)
    mean_sq = uniform_filter(np.where(finite, slope * slope, 0.0), size=size, mode="nearest") / np.maximum(weight, 1e-6)
    variability = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)).astype(np.float32)
    variability[~finite] = np.nan
    return variability


def build_slope_variability_2000m(ctx) -> np.ndarray:
    """Local standard deviation of slope over an approximately 2-km window."""
    from scipy.ndimage import uniform_filter
    import rasterio

    with rasterio.open(ctx.dem_path) as src:
        dem = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        transform = src.transform
    dy, dx = np.gradient(dem, abs(transform.e), abs(transform.a))
    slope = np.degrees(np.arctan(np.hypot(dx, dy))).astype(np.float32)
    finite = np.isfinite(slope)
    size = max(3, int(round(2000.0 / abs(transform.a))))
    weight = uniform_filter(finite.astype(np.float32), size=size, mode="nearest")
    mean = uniform_filter(np.where(finite, slope, 0.0), size=size, mode="nearest") / np.maximum(weight, 1e-6)
    mean_sq = uniform_filter(np.where(finite, slope * slope, 0.0), size=size, mode="nearest") / np.maximum(weight, 1e-6)
    variability = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)).astype(np.float32)
    variability[~finite] = np.nan
    return variability


def build_plan_curvature(ctx) -> np.ndarray:
    """Planform curvature from second DEM derivatives, with nodata propagated."""
    import rasterio

    with rasterio.open(ctx.dem_path) as src:
        dem = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        transform = src.transform
    cell_y, cell_x = abs(transform.e), abs(transform.a)
    dz_dy, dz_dx = np.gradient(dem, cell_y, cell_x)
    d2z_dyy, d2z_dyx = np.gradient(dz_dy, cell_y, cell_x)
    d2z_dxy, d2z_dxx = np.gradient(dz_dx, cell_y, cell_x)
    grad_sq = dz_dx * dz_dx + dz_dy * dz_dy
    numerator = d2z_dxx * dz_dy * dz_dy - 2.0 * d2z_dxy * dz_dx * dz_dy + d2z_dyy * dz_dx * dz_dx
    curvature = (numerator / np.maximum(grad_sq, 1e-12) ** 1.5).astype(np.float32)
    curvature[~np.isfinite(dem)] = np.nan
    return curvature


RECIPES: list[dict[str, Any]] = [
    {"name": "slope_degrees", "build": build_slope_degrees},
    {"name": "northness", "build": build_northness},
    {"name": "eastness", "build": build_eastness},
    {"name": "slope_variability_500m", "build": build_slope_variability_500m},
    {"name": "slope_variability_1000m", "build": build_slope_variability_1000m},
    {"name": "slope_variability_2000m", "build": build_slope_variability_2000m},
    {"name": "plan_curvature", "build": build_plan_curvature},
    {"name": "forest_cover", "build": build_forest_cover},
    {"name": "distance_to_buildings_m", "build": build_distance_to_buildings_m},
    {"name": "distance_to_electrical_infrastructure_m", "build": build_distance_to_electrical_infrastructure_m},
    {"name": "open_natural_cover", "build": build_open_natural_cover},
    {"name": "artificial_cover", "build": build_artificial_cover},
    {"name": "pasture_cover", "build": build_pasture_cover},
]
