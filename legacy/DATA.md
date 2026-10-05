# Archived data notes

The input data lives in the repo's `data/` directory (kept out of git by
`.gitignore` — the dynamic archives alone are ~4.4 GB). This copy is
self-contained: it was physically copied from `autosearch_susceptibility` and
`autosearch_spacetime`, not symlinked, so it survives deletion of those repos.

## Contents

```
data/
  points_landslide.geojson    landslide event points
  points_wildfire.geojson     wildfire event points
  grids/
    dem.tif                   the ONLY seed static grid (the reference grid)
  dynamic/                    time-varying archives (one directory per variable)
    temperature/*.nc          daily mean temperature, °C, dims (DATE, y, x)
    precipitation/*.nc        daily total precipitation, mm, dims (DATE, y, x)
    lightning/*.nc            daily lightning flash density, flashes/day, dims (DATE, y, x),
                               one file per year (2012-2024); grid matches dem.tif exactly
                               (250m, 642x398) — unlike precipitation/temperature (see below)
  vectors/
    roads.gpkg                 LineString, EPSG:32632
    buildings.gpkg              Point (centroids, not footprints), EPSG:32632
    CorineLandCover_polygon.tif Corine Land Cover level-3 codes, EPSG:25832 @ 30m
    electrical_infrastructure.* electrical infrastructure vectors
```

This is deliberately minimal compared to `autosearch_susceptibility`'s bundle
of ~14 precomputed static grids (slope, aspect, TRI, northness/eastness,
distroads, flammability, lightning, treecoverdensity, walking-time-to-building,
climate climatologies, ...) — **none of those are carried over as pre-baked
grids**. The agent derives whatever it wants from these raw primitives via
`features.py`; that is the point of this repo. `SAGA GIS` (`saga_cmd`) is the
preferred tool for this: fast (a full terrain-compound pass over the DEM
takes ~2-3s; a distance-to-feature layer a few seconds more) and robust
enough to run live inside one `features.py` recipe's own subprocess-isolated,
`feature_build_timeout_s`-guarded call — there is no need to precompute SAGA
outputs outside the agent's timeout the way the (genuinely slow, multi-minute,
hundreds-of-files) climate `DERIVED` aggregates are. See `features.py`'s
module docstring for the exact tools (`ta_compound 0` for terrain, `grid_tools
26` for distance-to-feature) and naming guidance. `data/grids/mask_exclude.tif`
is also NOT bundled — it is generated once per hazard branch from the Corine
layer during setup (see `program.md`'s Setup section), never committed.

## Raw primitives, in brief

- **`dem.tif`** — EPSG:32632, 642×398, 250m resolution. Defines the sampling
  domain and prediction resolution for everything else.
- **`dynamic/{precipitation,temperature}/*.nc`** — same CRS and resolution as
  the DEM (EPSG:32632, 250m), but **not** the same shape/bounds: 641x643
  cells covering a taller extent than `dem.tif`'s 642x398 (they extend
  further south). Point-sampling still works correctly since it's
  coordinate-based, not pixel-index-based, but do not assume these rasters
  are pixel-aligned with the DEM or with each other's cache outputs.
- **`dynamic/lightning/*.nc`** — same CRS, resolution, AND shape/bounds as
  the DEM (642x398, verified identical), unlike precipitation/temperature
  above. Fill value is a genuine `NaN` (precipitation/temperature convention),
  not a sentinel.
  `prepare.py --describe-dynamic` gives units/coverage/value-range detail;
  `--describe-primitives` gives a quick inventory of everything below too.
- **`vectors/roads.gpkg`** — `LineString` road network, EPSG:32632.
- **`vectors/buildings.gpkg`** — `Point` building **centroids** (not
  footprints), EPSG:32632. A "distance to building" recipe is a
  centroid-distance, not a footprint-distance.
- **`vectors/CorineLandCover_polygon.tif`** — Corine Land Cover, level-3 codes,
  EPSG:25832 @ 30m — a **different CRS/resolution than the reference grid**;
  any use of it (as a features.py recipe input, or via
  `--describe-raster-classes`/`--build-exclude-mask`) reprojects it first
  (nearest-neighbor — categorical data, never averaged). This is this
  branch's own choice, not a framework assumption — `prepare.py` bundles no
  Corine-specific (or any other domain's) code or label table; the exclusion
  mechanism works on any categorical raster (land cover, geology, demographic
  zoning, ...) configured via `config.yml`'s `sampling.exclude_raster` (see
  program.md's Setup section). A future branch in a different region, or
  using a different classification entirely, can point that setting anywhere.
- **`vectors/electrical_infrastructure.shp`** — electrical infrastructure
  vectors. **Missing its `.shx`/`.prj` sidecars** (a pre-existing issue in the
  source data) — `prepare.py` sets `SHAPE_RESTORE_SHX=YES` so GDAL rebuilds
  the index on the fly, but the CRS comes back as `None`; assume EPSG:32632
  (consistent with every other vector layer here) if a recipe needs to set it
  explicitly.

## Antecedent dynamic-archive features
`prepare.py`'s `build_antecedent_features()` computes, once per `prepare.py`
run, sum and mean of every dynamic variable over 1/7/30/60-day windows (each
ending on and including an observation's own date), sampled at that
observation's own location — not a single climatology raster shared by every
point, but a genuinely per-observation, date-linked feature. This is what
lets a short-lived triggering signal (e.g. the rainfall that actually
preceded a landslide) show up, which a long-term spatial average of the same
variable structurally cannot capture. Columns are named
`<var>_sum_<N>d`/`<var>_mean_<N>d` (e.g. `precipitation_sum_1d`) and are
ordinary base features in `artifacts/features/<process>/`, selectable via
`FEATURES.include` exactly like `dem`.

Presence points get their real recorded `date`. Background (absence) points
have no real date, so `assign_absence_dates()` resamples one with
replacement from the real event dates (preserving their actual seasonal/
yearly distribution) — rejecting and redrawing any candidate within
`sampling.antecedent_buffer_days` of a real event within
`sampling.antecedent_buffer_radius_m`, so a background point can't be
assigned a date when a nearby real event's conditions (e.g. a regional
storm) were actually present. Both are `config.yml` settings, not agent-
editable (same convention as `event_buffer_m`).

Dynamic archive timestamps are noon-anchored (`12:00:00`), not midnight, so
window boundaries are computed on calendar-day resolution, not exact-instant
equality — see the comment in `build_antecedent_features()` if extending
this.

## Adding a new dynamic archive
`prepare.py` has no data-ingestion code (deliberately — that's a rare, one-off
task, not part of the repeated pipeline). To add a new time-varying variable,
write a one-off script that produces `data/dynamic/<var>/<var>_<YYYY>.nc`
(or any file split by year/period) matching the format the `DERIVED`/
`derived_grid()` machinery already reads:
- dims `(DATE, y, x)`, data variable named `<var>` (the directory name).
- Reprojected onto the reference grid's CRS (nearest/bilinear as appropriate
  for the variable) — shape/bounds need not match `dem.tif` exactly (see
  precipitation/temperature below), but the CRS must.
- `NaN` fill value, not a sentinel.
ERA5-Land (global, free via Copernicus CDS, `cdsapi` + `rioxarray`) is one
easy source if a new climate variable is needed; there's nothing dynamic-data
-specific about the ingestion step, so any well-specified script works.

## Grid specification (the reference grid)

- **CRS:** WGS 84 / UTM zone 32N (EPSG:32632)
- **Resolution / extent:** 250 m, 642 × 398 (defined by `dem.tif`)
- **Format:** single-band GeoTIFF

## Provenance and licensing — _to complete before public release_

| Layer / inventory | Source & citation | License |
|---|---|---|
| `points_wildfire.geojson` | _[FILL IN: inventory author, year, DOI]_ | _[FILL IN]_ |
| `points_landslide.geojson` | _[FILL IN]_ | _[FILL IN]_ |
| `dem.tif` | _[FILL IN: e.g. Copernicus DEM]_ | _[FILL IN]_ |
| dynamic (temperature, precipitation daily grids) | _[FILL IN: e.g. local met service; or ERA5-Land, Muñoz-Sabater 2021]_ | _[FILL IN]_ |
| `dynamic/lightning/*.nc` | _[FILL IN: e.g. lightning detection network, provider, year range]_ | _[FILL IN]_ |
| `roads.gpkg`, `buildings.gpkg` | _[FILL IN: e.g. OpenStreetMap]_ | _[FILL IN]_ |
| `CorineLandCover_polygon.tif` | Copernicus Land Monitoring Service, Corine Land Cover | _[FILL IN: CLC license terms]_ |
| `electrical_infrastructure.*` | _[FILL IN]_ | _[FILL IN]_ |

Non-events (absences) are **generated**, not observed: random background
locations in the valid domain, buffered away from every event, optionally
restricted by the land-cover exclusion mask — see `prepare.py` and
`program.md`.
