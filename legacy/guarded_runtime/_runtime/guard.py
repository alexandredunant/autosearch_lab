#!/usr/bin/env python3
"""Resource guard: everything generated lives in artifacts/ and shares one disk budget.

python guard.py usage          show artifact usage against limits.disk_gb

Supervision = per-file size limit (RLIMIT_FSIZE, inherited by saga_cmd/python children),
a watchdog on the total size of artifacts/, memory and wall-clock limits. On violation
the whole process group is killed. This is a watchdog, not a filesystem quota.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import psutil
import yaml

ROOT = Path(__file__).resolve().parent.parent
PROCESSES = {"landslide", "debris_flow", "rockfall"}

# pip rasterio bundles its own PROJ database; conda's PROJ/GDAL plugins clash with it and with
# the system SAGA. This environment (inherited by the agent and saga_cmd) works for both.
_spec = importlib.util.find_spec("rasterio")
if _spec and _spec.origin and (Path(_spec.origin).parent / "proj_data").is_dir():
    os.environ["PROJ_DATA"] = str(Path(_spec.origin).parent / "proj_data")
    os.environ.pop("PROJ_LIB", None)
    os.environ["GDAL_DRIVER_PATH"] = "disable"


def config(path=ROOT / "config.yml"):
    path = Path(path).resolve()
    if path.stat().st_size > 64_000:
        raise ValueError("Configuration exceeds 64 KB")
    cfg = yaml.safe_load(path.read_text())
    cfg["_path"], cfg["_root"] = str(path), str(path.parent)
    if cfg["project"]["process"] not in PROCESSES:
        raise ValueError(f"process must be one of {sorted(PROCESSES)}")
    for key, value in cfg["limits"].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not value > 0:
            raise ValueError(f"limits.{key} must be a positive number")
    if not cfg["limits"]["file_gb"] <= cfg["limits"]["disk_gb"]:
        raise ValueError("file_gb must not exceed disk_gb")
    if "job_gb" in cfg["limits"] and not cfg["limits"]["file_gb"] <= cfg["limits"]["job_gb"] <= cfg["limits"]["disk_gb"]:
        raise ValueError("Budgets must satisfy file_gb <= job_gb <= disk_gb")
    for key in ("cpus", "max_cells", "max_samples"):
        if key in cfg["limits"] and type(cfg["limits"][key]) is not int:
            raise ValueError(f"limits.{key} must be an integer")
    return cfg


def source(cfg, value):
    return (Path(cfg["_root"]) / value).resolve()


def artifacts(cfg):
    return Path(cfg["_root"]) / "artifacts"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:20]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")



def tree_bytes(root):
    """Allocated or logical size, whichever is larger; symlinks are not followed or allowed."""
    total = 0
    if Path(root).is_symlink():
        raise ValueError(f"Artifact root must not be a symlink: {root}")
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            p = Path(directory) / name
            if p.is_symlink():
                raise ValueError(f"Symlinks are not allowed in artifacts: {p}")
            with contextlib.suppress(FileNotFoundError):
                st = p.stat()
                if p.is_file():
                    total += max(st.st_size, st.st_blocks * 512)
                elif not p.is_dir():
                    raise ValueError(f"Special file in artifacts: {p}")
    return total


@contextlib.contextmanager
def locked(cfg):
    root = artifacts(cfg)
    tree_bytes(root)
    root.mkdir(exist_ok=True)
    with (root / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another job/agent session owns the artifact budget") from None
        yield


def supervise(cfg, cmd, cwd, timeout_s, stdin=None, log=None):
    """Run cmd in its own process group under the shared limits; kill the group on violation."""
    limits, root = cfg["limits"], artifacts(cfg)
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Timeout must be positive and finite")
    if not Path(cwd).resolve().is_relative_to(root.resolve()):
        raise ValueError("Supervised working directories must be inside artifacts/")
    disk = int(limits["disk_gb"] * 1e9)
    if tree_bytes(root) >= disk:
        raise RuntimeError(f"artifacts/ already uses >= {limits['disk_gb']} GB; delete outputs first")
    if shutil.disk_usage(root).free < 2e9:
        raise RuntimeError("Less than 2 GB free on the artifact filesystem")
    file_limit = int(limits["file_gb"] * 1e9)
    used = tree_bytes(root)
    reservation = int(limits.get("job_gb", 0) * 1e9)
    if reservation and used + reservation > disk:
        raise RuntimeError("Cannot reserve job_gb within the shared generated-data budget")
    if reservation and shutil.disk_usage(root).free < reservation + 1e9:
        raise RuntimeError("Insufficient free disk for job reservation plus 1 GB headroom")

    def child_limits():
        for kind, value in ((resource.RLIMIT_FSIZE, file_limit), (resource.RLIMIT_CORE, 0)):
            _, hard = resource.getrlimit(kind)
            value = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(kind, (value, value))
        if hasattr(os, "sched_getaffinity"):
            os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:int(limits["cpus"])])

    env = os.environ.copy()
    tmp = Path(tempfile.mkdtemp(prefix=".tmp-", dir=cwd))
    for key in ("TMPDIR", "TMP", "TEMP", "CPL_TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
        env[key] = str(tmp)  # temporary files count against the budget too
    env["PYTHONPATH"] = str(ROOT)
    env["AUTOSEARCH_GUARD_CONFIG"] = cfg["_path"]
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = str(int(limits["cpus"]))
    # A file avoids blocking the supervisor on a full stdin pipe during CLI startup.
    prompt_file = None
    try:
        if stdin is not None:
            (tmp / "stdin.txt").write_text(stdin)
            prompt_file = (tmp / "stdin.txt").open()
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=prompt_file,
                                stdout=log, stderr=subprocess.STDOUT if log else None,
                                preexec_fn=child_limits, start_new_session=True, text=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    finally:
        if prompt_file is not None:
            prompt_file.close()
    started, reason = time.monotonic(), None
    try:
        while proc.poll() is None and reason is None:
            if tree_bytes(root) > disk:
                reason = f"artifacts/ exceeded {limits['disk_gb']} GB"
            elif reservation and tree_bytes(cwd) > reservation:
                reason = "job exceeded its reserved scratch/output budget"
            elif time.monotonic() - started > timeout_s:
                reason = f"wall-clock limit {timeout_s:.0f} s"
            else:
                with contextlib.suppress(psutil.Error):
                    tree = [psutil.Process(proc.pid), *psutil.Process(proc.pid).children(recursive=True)]
                    if sum(p.memory_info().rss for p in tree) > limits["memory_gb"] * 1e9:
                        reason = f"memory above {limits['memory_gb']} GB"
                    elif len(tree) > 64:
                        reason = "more than 64 worker processes"
            time.sleep(0.2)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)  # also stops leftover children (e.g. saga_cmd)
        proc.wait()
        # Charge short-lived writes too, before temporary-file cleanup.
        if tree_bytes(root) > disk:
            reason = f"artifacts/ exceeded {limits['disk_gb']} GB"
        elif reservation and tree_bytes(cwd) > reservation:
            reason = "job exceeded its reserved scratch/output budget"
        shutil.rmtree(tmp, ignore_errors=True)
    if reason:
        raise RuntimeError(f"Stopped: {reason}")
    return proc.returncode


def run_job(cfg, script, destination, args=(), payload=None):
    """Run a trusted script in a fresh directory; publish it only on success."""
    destination = Path(destination)
    with locked(cfg):
        if destination.exists():
            raise FileExistsError(f"Existing result preserved: {destination}")
        job = Path(tempfile.mkdtemp(prefix=".job-", dir=artifacts(cfg)))
        try:
            for name, value in (payload or {}).items():
                if Path(name).name != name:
                    raise ValueError("Job payload names must be simple filenames")
                write_json(job / name, value)
            with (job / "worker.log").open("w") as log:
                code = supervise(cfg, [sys.executable, str(ROOT / script), "--worker", "--config", cfg["_path"], *args],
                                 job, cfg["limits"]["job_timeout_s"], log=log)
            if code:
                with (job / "worker.log").open("rb") as f:
                    f.seek(max(0, (job / "worker.log").stat().st_size - 4000))
                    raise RuntimeError(f"{script} failed:\n" + f.read().decode(errors="replace"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            job.rename(destination)
        finally:
            shutil.rmtree(job, ignore_errors=True)


