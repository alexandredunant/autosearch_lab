"""Sampling, leakage and resource-guard checks on tiny synthetic data (python -m unittest)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import xarray as xr
import yaml

import guard
from features import archive_index, daily_history, validate_experiment
from prepare import build_samples, study_dir


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="massmovement-test-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        c = self.cfg = copy.deepcopy(guard.config(guard.ROOT / "config.yml"))
        c["_root"], c["_path"] = str(self.root), str(self.root / "config.yml")
        c["project"].update(dem="dem.tif", events="events.geojson")
        c["sampling"].update(background_ratio=3, exclude_raster=None, wet_days=None)
        c["dynamic"] = {"max_history_days": 7, "same_day": False,
                        "sources": {"rain": {"glob": "rain.nc", "variable": "rain", "crs": "EPSG:32632"}}}
        c["evaluation"].update(block_size_m=500, folds=3, holdout_fraction=0.2)
        c["limits"].update(disk_gb=0.01, job_gb=0.004, file_gb=0.001, memory_gb=4, cpus=1, job_timeout_s=120)
        yy, xx = np.mgrid[:40, :40]
        with rasterio.open(self.root / "dem.tif", "w", driver="GTiff", height=40, width=40, count=1,
                           dtype="float32", crs="EPSG:32632", transform=from_origin(600000, 5200000, 100, 100),
                           nodata=-9999) as f:
            f.write((100 + xx * 5 + yy * 2 + np.sin(xx) * 8).astype("float32"), 1)
        points = [(600050 + c_ * 100, 5199950 - r * 100) for r in range(3, 38, 7) for c_ in range(3, 38, 7)]
        dates = pd.date_range("2020-01-15", "2020-03-15", periods=len(points)).normalize()
        gpd.GeoDataFrame({"date": dates.strftime("%Y-%m-%d")}, geometry=gpd.points_from_xy(*zip(*points)),
                         crs="EPSG:32632").to_file(self.root / "events.geojson", driver="GeoJSON")
        # Rain on day i (from 2020-01-01) equals i; shifted grid with ascending y, unlike the DEM.
        values = np.broadcast_to(np.arange(90, dtype="float32")[:, None, None], (90, 42, 42)).copy()
        self.ds = xr.Dataset({"rain": (("DATE", "y", "x"), values)},
                             coords={"DATE": pd.date_range("2020-01-01 12:00", periods=90),
                                     "x": 599950 + np.arange(42) * 100, "y": 5195950 + np.arange(42) * 100})
        self.ds.to_netcdf(self.root / "rain.nc")
        self.save()

    def save(self):
        (self.root / "config.yml").write_text(yaml.safe_dump({k: v for k, v in self.cfg.items() if k[0] != "_"}))

    def test_history_offsets_footprint_and_missing_days(self):
        frame = pd.DataFrame({"x": [600350, 900000], "y": [5199650] * 2, "date": pd.to_datetime(["2020-01-11"] * 2)})
        hist = daily_history(self.cfg, frame, "rain", 3)
        np.testing.assert_array_equal(hist[0], [10, 9, 8, 7])     # offset k = k days before
        self.assertTrue(np.isnan(hist[1]).all())                   # outside the archive
        self.ds.isel(DATE=[i for i in range(90) if i != 8]).to_netcdf(self.root / "rain.nc", mode="w")
        self.assertTrue(np.isnan(daily_history(self.cfg, frame.iloc[:1], "rain", 3)[0, 2]))  # 2020-01-09 removed
        self.ds.to_netcdf(self.root / "rain_copy.nc")
        self.cfg["dynamic"]["sources"]["rain"]["glob"] = "rain*.nc"
        with self.assertRaisesRegex(ValueError, "overlapping"):
            archive_index(self.cfg, "rain")

    def test_sampling_protocol(self):
        a, report = build_samples(self.cfg)
        pd.testing.assert_frame_equal(a, build_samples(self.cfg)[0])  # seeded
        cases, controls = a[a.label == 1], a[a.label == 0]
        from scipy.spatial import cKDTree
        self.assertGreaterEqual(cKDTree(cases[["x", "y"]]).query(controls[["x", "y"]])[0].min(), 150)
        self.assertFalse(controls.cell.duplicated().any())          # one date per location
        self.assertFalse(set(cases.cell) & set(controls.cell))
        self.assertEqual(a.groupby("block").fold.nunique().max(), 1)  # blocks never split
        quota = report["control_month_quota"]
        for month, days in {"2020-01": 17, "2020-02": 29, "2020-03": 15}.items():
            self.assertLess(abs(quota[month] - sum(quota.values()) * days / 61), 1)

    def test_wet_day_rule_applies_to_both_classes(self):
        self.cfg["sampling"]["wet_days"] = {"variable": "rain", "days": 1, "lag": 1, "stat": "sum", "threshold": 40}
        a, _ = build_samples(self.cfg)
        self.assertTrue((a.date >= "2020-02-10").all())
        self.assertEqual(set(a.label), {0, 1})

    def test_guard_kills_oversized_and_runaway_jobs(self):
        root = guard.artifacts(self.cfg)
        root.mkdir()
        for name, code in [("big file", "open('a', 'wb').write(b'x' * 2_000_000)"),
                           ("timeout", "import time; time.sleep(30)"),
                           ("total", "import time\nfor i in range(40):\n    open(str(i), 'wb').write(b'x' * 900_000); time.sleep(0.1)\ntime.sleep(30)")]:
            with self.subTest(name):
                (self.root / "job.py").write_text(code)
                work = root / name.replace(" ", "_")
                work.mkdir()
                try:
                    code = guard.supervise(self.cfg, [sys.executable, str(self.root / "job.py")], work, 5)
                    self.assertNotEqual(code, 0)                    # SIGXFSZ on the per-file limit
                except RuntimeError as e:
                    self.assertIn("Stopped", str(e))
        (root / "huge").write_bytes(b"x" * 11_000_000)
        with self.assertRaisesRegex(RuntimeError, "already uses"):
            guard.supervise(self.cfg, ["true"], root, 5)
        with guard.locked(self.cfg), self.assertRaisesRegex(RuntimeError, "Another"):
            with guard.locked(self.cfg):
                pass

    def test_prepare_and_train_end_to_end(self):
        self.cfg["limits"].update(disk_gb=1, job_gb=0.8, file_gb=0.5)
        self.save()
        run = lambda *a: subprocess.run([sys.executable, *a], cwd=self.root, capture_output=True, text=True, timeout=300)
        done = run(str(guard.ROOT / "prepare.py"), "--config", "config.yml")
        self.assertEqual(done.returncode, 0, done.stderr)
        prepared = study_dir(self.cfg) / "prepared"
        points = pd.read_csv(prepared / "points.csv")
        self.assertEqual(list(points), ["sample_id", "x", "y", "date"])  # no labels for the agent
        spec = {"description": "test", "terrain": [], "dynamic": [{"variable": "rain", "days": 3, "lag": 1, "stat": "sum"}], "model": {"max_rounds": 4}}
        path = self.root / "experiment.yml"
        path.write_text(yaml.safe_dump(spec))
        command = (str(guard.ROOT / "train.py"), "--config", "config.yml", "--experiment", str(path))
        done = run(*command)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(len(list((study_dir(self.cfg) / "runs").glob("*/metrics.json"))), 1)
        done = run(*command)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(len(list((study_dir(self.cfg) / "runs").glob("*/metrics.json"))), 1)

    def test_fast_total_write_is_caught_after_exit(self):
        root = guard.artifacts(self.cfg)
        root.mkdir()
        self.cfg["limits"].update(disk_gb=0.0015, job_gb=0.001, file_gb=0.001)
        code = "from pathlib import Path\nPath('a').write_bytes(b'x'*900000)\nPath('b').write_bytes(b'x'*900000)"
        with self.assertRaisesRegex(RuntimeError, "Stopped"):
            guard.supervise(self.cfg, [sys.executable, "-c", code], root, 5)

    def test_empty_calendar_and_irregular_coordinates_fail_clearly(self):
        self.cfg["sampling"].update(start_date="2010-01-01", end_date="2010-02-01")
        self.cfg["dynamic"]["sources"]["other"] = self.cfg["dynamic"]["sources"]["rain"].copy()
        with self.assertRaisesRegex(ValueError, "No dates"):
            build_samples(self.cfg)
        ds = self.ds.assign_coords(x=self.ds.x.values + np.r_[0, np.ones(41)])
        ds.to_netcdf(self.root / "rain.nc", mode="w")
        with self.assertRaisesRegex(ValueError, "regular monotone"):
            archive_index(self.cfg, "rain")

    def test_undated_events_and_same_cells_remain_excluded(self):
        events = gpd.read_file(self.root / "events.geojson")
        events.loc[0, "date"] = None
        events.to_file(self.root / "events.geojson", driver="GeoJSON")
        self.cfg["sampling"]["event_buffer_m"] = 1
        frame, report = build_samples(self.cfg)
        self.assertEqual(report["undated_events"], 1)
        self.assertFalse(set(frame.loc[frame.label == 0, "cell"]) & {r * 40 + c for r in range(3, 38, 7) for c in range(3, 38, 7)})

    def test_saga_baseline_and_holdout_are_supervised(self):
        import shutil
        if shutil.which("saga_cmd") is None:
            self.skipTest("SAGA GIS is unavailable")
        self.cfg["limits"].update(disk_gb=1, job_gb=0.8, file_gb=0.5)
        from shapely.geometry import LineString
        gpd.GeoDataFrame(geometry=[LineString([(600000, 5199000), (604000, 5199000)])], crs="EPSG:32632").to_file(self.root / "roads.gpkg", driver="GPKG")
        self.cfg["predictors"] = [{"name": "distance_to_roads", "path": "roads.gpkg", "kind": "vector_distance"}]
        self.save()
        spec = {"description": "SAGA integration", "terrain": ["slope", "plan_curvature"],
                "vectors": [{"name": "distance_to_roads", "path": "roads.gpkg"}], "model": {"max_rounds": 4}}
        path = self.root / "experiment.yml"
        path.write_text(yaml.safe_dump(spec))
        for script, args in (("prepare.py", []), ("train.py", ["--experiment", str(path)]), ("train.py", ["--evaluate"])):
            done = subprocess.run([sys.executable, str(guard.ROOT / script), "--config", self.cfg["_path"], *args],
                                  cwd=self.root, capture_output=True, text=True, timeout=90)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        root = study_dir(self.cfg)
        self.assertTrue(list((root / "runs").glob("*/slope.sdat")))
        self.assertTrue(list((root / "runs").glob("*/distance_to_roads.sdat")))
        result = json.loads((root / "holdout" / "metrics.json").read_text())
        self.assertTrue(0 <= result["average_precision"] <= 1)
        done = subprocess.run([sys.executable, str(guard.ROOT / "train.py"), "--config", self.cfg["_path"], "--experiment", str(path)], capture_output=True, text=True)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("closed", done.stderr)

    def test_executable_proposals_unapproved_inputs_and_future_windows_rejected(self):
        base = {"description": "valid", "terrain": [], "model": {}}
        for change in ({"command": "touch outside"}, {"terrain": ["upsample"]},
                       {"model": {"n_jobs": 100}}, {"model": {"max_rounds": 1000000}},
                       {"dynamic": [{"variable": "rain", "days": 2, "lag": 0, "stat": "sum"}]},
                       {"vectors": [{"name": "events", "path": "events.geojson"}]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_experiment(base | change, self.cfg)

    def test_failed_job_cleanup_and_frozen_points_integrity(self):
        with self.assertRaises(RuntimeError):
            guard.run_job(self.cfg, "train.py", guard.artifacts(self.cfg) / "failed")
        self.assertFalse(list(guard.artifacts(self.cfg).glob(".job-*")))
        self.assertFalse((guard.artifacts(self.cfg) / "failed").exists())
        from prepare import verify_prepared
        done = subprocess.run([sys.executable, str(guard.ROOT / "prepare.py"), "--config", self.cfg["_path"]], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        path = study_dir(self.cfg) / "prepared" / "points.csv"
        with path.open("a") as f:
            f.write("\n")
        with self.assertRaisesRegex(ValueError, "modified"):
            verify_prepared(self.cfg)

    def test_llm_proposals_are_read_only_and_validated_without_execution(self):
        import os
        from unittest.mock import patch
        from train import suggest
        valid = {"description": "Test rain", "terrain": [],
                 "dynamic": [{"variable": "rain", "days": 3, "lag": 1, "stat": "sum"}]}
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            for candidate, accepted in ((valid, True), ({"description": "bad", "command": "touch outside"}, False)):
                def fake_model(command, **kwargs):
                    self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
                    self.assertIn("--ignore-user-config", command)
                    self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", command)
                    Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps(candidate))
                with patch("train.shutil.which", return_value="/fake/codex"), patch("train.subprocess.run", side_effect=fake_model):
                    if accepted:
                        suggest(self.cfg)
                    else:
                        with self.assertRaises(ValueError):
                            suggest(self.cfg)
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
