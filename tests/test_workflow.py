"""The experiment loop as the agent runs it: evaluate, log, duplicates, recipes, limits."""
from __future__ import annotations

import csv
import json
import os
import signal
import time

import numpy as np
import rasterio
import yaml
from conftest import COLS, RAW, ROWS

SLOPE_RECIPE = """#!/usr/bin/env bash
set -euo pipefail
ulimit -f 2000000
mkdir -p artifacts/features/slope
timeout 30m saga_cmd --cores=4 ta_morphometry 0 -ELEVATION data/dem.tif -SLOPE artifacts/features/slope/slope.sdat -UNIT_SLOPE 1 >/dev/null
# saga_cmd in a comment is not a command
(cd . && timeout 30m saga_cmd grid_filter 0 -INPUT a.sdat -RESULT b.sdat) && echo done
"""


def ledger(lab) -> list[dict]:
    with (lab.root / "results.tsv").open() as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"))


def test_baseline_run_outputs(lab):
    out = lab.train()
    assert 0.5 < lab.metric(out) <= 1
    assert "best_kept_val_auroc: none" in out.stdout
    art = lab.root / "artifacts"
    with rasterio.open(art / "last_map.tif") as src:
        surface = src.read(1)
    assert surface.shape == (ROWS, COLS)
    assert np.isnan(surface[0:10, 0:12]).all()              # water is not mapped
    assert np.nanmin(surface) >= 0 and np.nanmax(surface) <= 1
    text = (art / "last_explanation.md").read_text()
    assert "## elevation" in text and "Feature Type: categorical" in text
    assert json.loads((art / "last_model.json").read_text())
    run = json.loads((art / "last_run.json").read_text())
    assert run["features"] == RAW and len(run["folds"]) == 3


def test_keep_discard_and_duplicates(lab):
    lab.train()
    lab.run("prepare.py", "log", "keep", 'raw "baseline"')
    maps = lab.root / "artifacts" / "maps"
    assert {"susceptibility_best.tif", "explanation_best.md", "ebm_best.json"} <= {p.name for p in maps.iterdir()}
    again = lab.train(check=False)
    assert again.returncode and "DUPLICATE" in again.stderr + again.stdout
    env = os.environ | {"RETRY": "1"}
    assert lab.metric(lab.train(env=env)) > 0.5

    lab.set_features({"elevation": RAW["elevation"]}, "drop landcover")
    out = lab.train()
    assert "best_kept_val_auroc: " in out.stdout and "none" not in out.stdout
    lab.run("prepare.py", "log", "discard", "elevation only")
    rows = ledger(lab)
    assert [r["status"] for r in rows] == ["keep", "discard"]
    assert rows[0]["description"] == "raw 'baseline'"
    assert rows[1]["features"] == "elevation=data/dem.tif"


def test_log_requires_fresh_run(lab):
    refused = lab.run("prepare.py", "log", "keep", "never ran", check=False)
    assert refused.returncode and "No finished run" in refused.stderr
    lab.run("prepare.py", "log", "crash", "it crashed")
    assert ledger(lab)[0]["val_auroc"] == "0.000000"


def test_recipes_are_required_and_recorded(lab):
    rr = np.mgrid[0:ROWS, 0:COLS][0].astype(float)
    bare = lab.add_feature("norecipe", rr, recipe=None)
    lab.set_features(RAW | {"rows": bare}, "feature without recipe")
    refused = lab.train(check=False)
    assert refused.returncode and "recipe" in refused.stderr

    path = lab.add_feature("slope", rr, recipe=SLOPE_RECIPE)
    lab.set_features(RAW | {"slope": path}, "slope feature")
    lab.train()
    lab.run("prepare.py", "log", "keep", "slope")
    commands = ledger(lab)[0]["saga_commands"].split(" && ")
    assert commands == [
        "saga_cmd --cores=4 ta_morphometry 0 -ELEVATION data/dem.tif -SLOPE artifacts/features/slope/slope.sdat -UNIT_SLOPE 1",
        "saga_cmd grid_filter 0 -INPUT a.sdat -RESULT b.sdat",
    ]
    assert "DUPLICATE" in lab.train(check=False).stderr
    (lab.root / "artifacts/features/slope/recipe.sh").write_text(SLOPE_RECIPE.replace("-UNIT_SLOPE 1", "-UNIT_SLOPE 0"))
    assert lab.metric(lab.train()) > 0.5                     # a changed recipe is a new experiment


def test_ebm_is_compulsory(lab):
    (lab.root / "train.py").write_text(
        "from sklearn.linear_model import LogisticRegression\nfrom prepare import evaluate\n"
        "evaluate({'elevation': 'data/dem.tif'}, LogisticRegression)\n")
    out = lab.train(check=False)
    assert out.returncode and "EBM is compulsory" in out.stderr


def test_disk_budget_blocks_runs(lab):
    config = yaml.safe_load((lab.root / "config.yml").read_text())
    config["limits"]["artifacts_gb"] = 1e-6
    (lab.root / "config.yml").write_text(yaml.safe_dump(config))
    out = lab.train(check=False)
    assert out.returncode and "prune" in out.stderr


def test_prune_keeps_features_of_kept_runs(lab):
    rr = np.mgrid[0:ROWS, 0:COLS][0].astype(float)
    kept = lab.add_feature("kept", rr, recipe="python tmp/rows.py\n")
    lab.add_feature("stale", rr, recipe="python tmp/stale.py\n")
    lab.set_features(RAW | {"kept": kept}, "kept feature")
    lab.train()
    lab.run("prepare.py", "log", "keep", "kept")
    lab.run("prepare.py", "prune")
    features = lab.root / "artifacts" / "features"
    assert sorted(p.name for p in features.iterdir()) == ["kept"]


def test_flat_terms_do_not_hang(lab):
    flat = lab.add_feature("constant", np.ones((ROWS, COLS)), recipe="python tmp/constant.py\n")
    lab.set_features(RAW | {"constant": flat}, "constant feature")
    lab.run("train.py", timeout=120)
    text = (lab.root / "artifacts/last_explanation.md").read_text()
    assert "## constant" in text and "Flat shape function" in text


def test_time_limit_ends_long_runs(lab):
    config = yaml.safe_load((lab.root / "config.yml").read_text())
    config["limits"]["timeout_min"] = 0.02                  # about one second
    (lab.root / "config.yml").write_text(yaml.safe_dump(config))
    (lab.root / "tmp").mkdir()
    (lab.root / "tmp" / "slow.py").write_text("import time\nimport prepare\ntime.sleep(30)\n")
    started = time.monotonic()
    out = lab.run("tmp/slow.py", check=False, timeout=60, env=os.environ | {"PYTHONPATH": str(lab.root)})
    assert out.returncode == -signal.SIGALRM and time.monotonic() - started < 15


def test_memory_limit_raises(lab):
    config = yaml.safe_load((lab.root / "config.yml").read_text())
    config["limits"]["memory_gb"] = 2
    (lab.root / "config.yml").write_text(yaml.safe_dump(config))
    (lab.root / "tmp").mkdir()
    (lab.root / "tmp" / "big.py").write_text(
        "import numpy as np\nimport prepare\ntry:\n    np.ones(3 * 10**9 // 8)\nexcept MemoryError:\n"
        "    print('refused')\nelse:\n    print('allocated')\nnp.ones(10**9 // 8)\nprint('small ok')\n")
    out = lab.run("tmp/big.py", env=os.environ | {"PYTHONPATH": str(lab.root)})
    assert out.stdout.splitlines() == ["refused", "small ok"]
