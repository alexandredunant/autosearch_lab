"""Integration on the real inputs and the system SAGA (skipped when either is missing)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest
from conftest import REPO

needs_data = pytest.mark.skipif(not (REPO / "data" / "grids" / "dem.tif").exists(), reason="no data/")
needs_saga = pytest.mark.skipif(shutil.which("saga_cmd") is None, reason="no saga_cmd")


@needs_data
@needs_saga
def test_real_study_with_saga_recipe(tmp_path):
    root = tmp_path / "lab"
    root.mkdir()
    for name in ("prepare.py", "train.py", "config.yml", "program.md"):
        shutil.copy(REPO / name, root)
    (root / "data").symlink_to(REPO / "data")

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=900, check=True)

    run(sys.executable, "prepare.py")
    report = json.loads((root / "artifacts/prepared/report.json").read_text())
    assert report["study"] == json.loads((REPO / "artifacts/prepared/report.json").read_text())["study"]

    # The SAGA template from program.md, verbatim, must run and yield a usable feature.
    text = (root / "program.md").read_text()
    recipe = text.split("```bash\n", 1)[1].split("```", 1)[0]
    (root / "tmp").mkdir()
    (root / "tmp/slope.sh").write_text(recipe)
    env_run = subprocess.run(["bash", "tmp/slope.sh"], cwd=root, capture_output=True, text=True,
                             timeout=900, env=os.environ, check=False)
    assert env_run.returncode == 0, env_run.stderr
    assert "ERROR" not in env_run.stderr                     # no conda GDAL plugin clash
    shutil.copy(root / "tmp/slope.sh", root / "artifacts/features/slope/recipe.sh")
    train = (root / "train.py").read_text().replace(
        '"elevation": "data/grids/dem.tif",',
        '"elevation": "data/grids/dem.tif",\n    "slope": "artifacts/features/slope/slope.sdat",')
    (root / "train.py").write_text(train)
    out = run(sys.executable, "train.py").stdout
    auroc = float(next(line.split()[1] for line in out.splitlines() if line.startswith("val_auroc:")))
    assert auroc > 0.75
    run_info = json.loads((root / "artifacts/last_run.json").read_text())
    assert run_info["saga_commands"][0].startswith("saga_cmd --cores=4 ta_morphometry 0")
