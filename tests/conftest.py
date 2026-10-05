"""Synthetic study fixture: a throwaway git repo running the real prepare.py as subprocesses."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

for _var in ("GDAL_DRIVER_PATH", "GDAL_DATA", "PROJ_DATA", "PROJ_LIB"):  # as .venv/bin/activate does
    os.environ.pop(_var, None)

REPO = Path(__file__).resolve().parent.parent
CRS, RES, X0, Y0, ROWS, COLS = "EPSG:32632", 250.0, 600_000.0, 5_200_000.0, 40, 60
WATER, GLACIER = (slice(0, 10), slice(0, 12)), (slice(30, 40), slice(50, 60))

TRAIN = '''from interpret.glassbox import ExplainableBoostingClassifier
from prepare import SEED, evaluate

FEATURES = {FEATURES}
CATEGORICAL = ("landcover",)


def make_model():
    return ExplainableBoostingClassifier(
        interactions=0, outer_bags=2, random_state=SEED, n_jobs=1,
        feature_types=["nominal" if n in CATEGORICAL else "continuous" for n in FEATURES])


if __name__ == "__main__":
    evaluate(FEATURES, make_model, CATEGORICAL)
'''
RAW = {"elevation": "data/dem.tif", "landcover": "data/landcover.tif"}

CONFIG = {
    "data": {"dem": "data/dem.tif", "events": "data/events.geojson",
             "landcover": {"path": "data/landcover.tif", "crs": CRS}},
    "sampling": {"seed": 7, "mask_landcover": [335, 512], "flat_slope_deg": None, "survey_mask": None,
                 "absence_ratio": 5, "buffer_m": 150},
    "evaluation": {"folds": 3},
    "limits": {"artifacts_gb": 1, "file_gb": 0.1, "timeout_min": 30, "memory_gb": 20, "cpus": 1},
}


def write_raster(path: Path, values: np.ndarray, dtype: str = "float32") -> None:
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", height=ROWS, width=COLS, count=1, dtype=dtype, crs=CRS,
                       transform=from_origin(X0, Y0, RES, RES), nodata=np.nan if "float" in dtype else 0) as dst:
        dst.write(values.astype(dtype), 1)


def cell_center(row: int, col: int) -> tuple[float, float]:
    return X0 + (col + 0.5) * RES, Y0 - (row + 0.5) * RES


@dataclass
class Lab:
    root: Path

    def run(self, *args: str, check: bool = True, env: dict | None = None,
            timeout: float = 600) -> subprocess.CompletedProcess:
        proc = subprocess.run([sys.executable, *args], cwd=self.root, capture_output=True, text=True,
                              env=env, timeout=timeout, check=False)
        if check and proc.returncode:
            raise AssertionError(f"{args} failed:\n{proc.stdout}\n{proc.stderr}")
        return proc

    def git(self, *args: str) -> None:
        subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=t@t", *args], cwd=self.root,
                       check=True, capture_output=True)

    def set_features(self, features: dict[str, str], message: str) -> None:
        (self.root / "train.py").write_text(TRAIN.replace("{FEATURES}", json.dumps(features)))
        self.git("commit", "-qam", message)

    def train(self, **kw) -> subprocess.CompletedProcess:
        return self.run("train.py", **kw)

    def metric(self, proc: subprocess.CompletedProcess, key: str = "val_auroc") -> float:
        return float(next(line.split()[1] for line in proc.stdout.splitlines() if line.startswith(f"{key}:")))

    def add_feature(self, name: str, values: np.ndarray, recipe: str | None) -> str:
        folder = self.root / "artifacts" / "features" / name
        folder.mkdir(parents=True, exist_ok=True)
        write_raster(folder / "grid.tif", values)
        if recipe is not None:
            (folder / "recipe.sh").write_text(recipe)
        return f"artifacts/features/{name}/grid.tif"


@pytest.fixture
def lab(tmp_path: Path) -> Lab:
    import geopandas as gpd
    import yaml
    from shapely.geometry import Point
    rng = np.random.default_rng(0)
    root = tmp_path / "study"
    (root / "data").mkdir(parents=True)
    shutil.copy(REPO / "prepare.py", root)
    (root / "config.yml").write_text(yaml.safe_dump(CONFIG))
    rr, cc = np.mgrid[0:ROWS, 0:COLS]
    write_raster(root / "data" / "dem.tif", 500 + 30 * rr + 10 * cc + rng.normal(0, 5, (ROWS, COLS)))
    landcover = np.full((ROWS, COLS), 311, "int16")
    landcover[:, 30:] = 231                       # pasture east, forest west: two domain classes
    landcover[WATER], landcover[GLACIER] = 512, 335
    write_raster(root / "data" / "landcover.tif", landcover, "int16")
    domain_cells = [(r, c) for r in range(12, 28) for c in range(15, 45)]
    picks = [domain_cells[i] for i in rng.choice(len(domain_cells), 36, replace=False)]
    points = [Point(cell_center(r, c)) for r, c in picks]
    points.append(Point(cell_center(*picks[0])[0] + 20, cell_center(*picks[0])[1]))  # same cell as another
    points += [Point(cell_center(2, 3)), Point(cell_center(5, 8)), Point(cell_center(35, 55))]  # masked
    points.append(Point(X0 - 5_000, Y0 + 5_000))  # outside the grid
    gpd.GeoDataFrame({"date": ["2010-01-01"] * len(points)}, geometry=points, crs=CRS).to_file(
        root / "data" / "events.geojson", driver="GeoJSON")
    (root / ".gitignore").write_text("/data/\n/artifacts/\n/tmp/\nresults.tsv\nrun.log\n__pycache__/\n")
    (root / "train.py").write_text(TRAIN.replace("{FEATURES}", json.dumps(RAW)))
    lab = Lab(root)
    lab.git("init", "-q")
    lab.git("add", ".")
    lab.git("commit", "-qm", "baseline")
    lab.run("prepare.py")
    return lab
