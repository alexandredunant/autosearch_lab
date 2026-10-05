#!/usr/bin/env bash
# Python environment installer. SAGA remains a system dependency.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
case "${1:-}" in
  --check) PYTHON_BIN="${PYTHON_BIN:-python}" ;;
  "")
    PYTHON_BIN="${PYTHON_BIN:-python3}"
    "$PYTHON_BIN" -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    PYTHON_BIN=.venv/bin/python
    ;;
  *) echo "Usage: bash install.sh [--check]"; exit 2 ;;
esac
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"; from _runtime import guard; import numpy, pandas, scipy, sklearn, yaml, geopandas, rasterio, xarray, netCDF4, pyproj, interpret, psutil; print("Python dependencies OK")'
if ! command -v saga_cmd >/dev/null 2>&1; then
  echo "Install SAGA GIS separately (e.g. sudo apt install saga on Debian/Ubuntu), then rerun --check."
  exit 1
fi
saga_cmd --version
echo "Ready. Review config.yml, then: python prepare.py && python train.py --description baseline"
echo "If you installed .venv, first run: source .venv/bin/activate"
