"""The frozen sample obeys the Corine mask, buffer, ratio and spatial-fold rules."""
from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
from conftest import GLACIER, WATER
from scipy.spatial import cKDTree


def masked(rows: pd.Series, cols: pd.Series) -> np.ndarray:
    out = np.zeros(len(rows), bool)
    for rs, cs in (WATER, GLACIER):
        out |= rows.between(rs.start, rs.stop - 1).to_numpy() & cols.between(cs.start, cs.stop - 1).to_numpy()
    return out


def test_report_counts(lab):
    report = json.loads((lab.root / "artifacts/prepared/report.json").read_text())
    assert report["events"] == 41
    assert report["outside_grid"] == 1
    assert report["outside_domain"] == 3          # two in water, one on the glacier
    assert report["presence_cells"] == 36         # duplicate cell counted once
    assert report["absence_cells"] == 5 * 36


def test_classes_respect_mask_and_buffer(lab):
    s = pd.read_csv(lab.root / "artifacts/prepared/samples.csv")
    assert not masked(s.row, s.col).any()
    assert s.groupby(["row", "col"]).size().max() == 1
    events = gpd.read_file(lab.root / "data/events.geojson")
    tree = cKDTree(np.c_[events.geometry.x, events.geometry.y])
    absences = s[s.label == 0]
    assert (tree.query(absences[["x", "y"]].to_numpy())[0] >= 150).all()   # incl. masked events


def test_folds_and_label_free_points(lab):
    s = pd.read_csv(lab.root / "artifacts/prepared/samples.csv")
    assert sorted(s.fold.unique()) == [0, 1, 2]
    assert (s.groupby("fold").label.nunique() == 2).all()
    points = pd.read_csv(lab.root / "artifacts/prepared/points.csv")
    assert not {"label", "fold"} & set(points.columns)


def test_deterministic_and_protected(lab):
    report = lab.root / "artifacts/prepared/report.json"
    study = json.loads(report.read_text())["study"]
    out = lab.run("prepare.py")
    assert "Already prepared" in out.stdout
    lab.run("prepare.py", "--force")
    assert json.loads(report.read_text())["study"] == study
