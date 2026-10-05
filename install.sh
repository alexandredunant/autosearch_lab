#!/usr/bin/env bash
# Installer: Python environment in .venv (uv, or venv + pip), SAGA GIS check, sample preparation.
#   bash install.sh            install, check, prepare
#   PYTHON_VERSION=3.11 bash install.sh
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

echo "== Python environment (.venv)"
if command -v uv >/dev/null 2>&1; then
  uv venv --allow-existing --python "$PYTHON_VERSION" .venv
  uv pip install --python .venv/bin/python -r requirements.txt
else
  "python$PYTHON_VERSION" -m venv .venv || python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"
import numpy, pandas, scipy, sklearn, yaml, geopandas, rasterio, xarray, netCDF4, interpret
print("Python", sys.version.split()[0], "packages OK")'
# TalkToEBM: import needs a key (placeholder is fine); fetch its tokenizer once so runs stay offline.
OPENAI_API_KEY="${OPENAI_API_KEY:-offline}" .venv/bin/python -c 'from t2ebm.utils import num_tokens_from_string_ as n; n("warm up", "gpt-4"); print("TalkToEBM OK")'

echo "== SAGA GIS"
if ! command -v saga_cmd >/dev/null 2>&1; then
  echo "saga_cmd not found. Install SAGA GIS (Debian/Ubuntu: sudo apt install saga;"
  echo "other systems: https://saga-gis.sourceforge.io), then rerun this script."
  exit 1
fi
# Run SAGA outside any conda GDAL, exactly as program.md's recipes do.
env -u GDAL_DRIVER_PATH -u GDAL_DATA -u PROJ_DATA saga_cmd --version | head -1

echo "== Data and sample"
.venv/bin/python - <<'PY'
import sys, yaml
from pathlib import Path
d = yaml.safe_load(Path("config.yml").read_text())["data"]
missing = [p for p in (d["dem"], d["events"], d["landcover"]["path"]) if not Path(p).exists()]
sys.exit(f"Missing inputs from config.yml: {missing}" if missing else 0)
PY
mkdir -p tmp
.venv/bin/python prepare.py
echo
echo "Ready. Activate with: source .venv/bin/activate"
echo "Then point your coding agent at program.md, e.g.: 'Read program.md and start a new run.'"
