"""Fixed scoring and append-only experiment history for agent-editable train.py."""
from __future__ import annotations

import ast
import contextlib
import csv
import importlib.util
import json
from pathlib import Path
import pickle
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

from .guard import ROOT, artifacts, digest, file_hash, locked, run_job, source, supervise, write_json
from prepare import study_dir, verify_prepared


def load_experiment(path):
    spec = importlib.util.spec_from_file_location("agent_experiment", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def positive_probability(model, matrix):
    import numpy as np
    classes = list(model.classes_)
    if set(classes) != {0, 1}:
        raise ValueError("Model must expose binary classes_ and predict_proba")
    p = np.asarray(model.predict_proba(matrix))[:, classes.index(1)]
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probability predictions")
    return p


class Context:
    """Optional helpers, not a sandbox. All heavy work runs in the supervised child."""
    def __init__(self, cfg, points):
        from .geo import reference
        self.cfg, self.points = cfg, points
        self.work = Path.cwd().resolve()
        self.dem = source(cfg, cfg["project"]["dem"])
        self.profile = reference(cfg)
        self.cpus = cfg["limits"]["cpus"]
        self._commands = {}
        self.command_log = self.work / "commands.jsonl"

    def input(self, relative_path):
        return source(self.cfg, relative_path)

    def output(self, name):
        path = (self.work / name).resolve()
        if not path.is_relative_to(self.work) or path == self.work:
            raise ValueError("Generated files must stay inside the experiment directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def raster(self, path, categorical=False):
        import rasterio
        from .geo import sample_raster
        path = Path(path).resolve()
        if path.is_relative_to(self.work):
            with rasterio.open(path) as ds:
                p = self.profile
                if ds.count != 1 or ds.shape != (p["height"], p["width"]) or ds.crs != p["crs"] or not ds.transform.almost_equals(p["transform"]):
                    raise ValueError("Generated predictor raster must match the DEM grid")
        return sample_raster(path, self.points, self.profile["crs"], categorical)

    def history(self, variable, days):
        import numpy as np
        from .geo import daily_history
        values = daily_history(self.cfg, self.points, variable, days)
        if not self.cfg["dynamic"].get("same_day"):
            values[:, 0] = np.nan
        return values

    def run(self, command, *, inputs=(), outputs=()):
        """Execute argv (or shlex-split command); log exact command and declared input hashes."""
        argv = shlex.split(command) if isinstance(command, str) else list(map(str, command))
        if not argv:
            raise ValueError("Empty command")
        inputs = [Path(p).resolve() for p in inputs]
        outputs = [self.output(p) for p in outputs]
        fingerprints = {str(p).replace(str(self.work), "{work}"): file_hash(p) for p in inputs}
        normalized = [a.replace(str(self.work), "{work}") for a in argv]
        identity = digest({"argv": normalized, "inputs": fingerprints})
        record = {"command_id": identity, "argv": argv, "bash": shlex.join(argv),
                  "normalized_argv": normalized, "inputs": fingerprints,
                  "outputs": [str(p) for p in outputs], "time": time.time()}
        def log(status, **extra):
            with self.command_log.open("a") as f:
                f.write(json.dumps(record | {"status": status} | extra) + "\n")
                f.flush()
        prior = self._commands.get(identity)
        if prior and outputs and all(p.is_file() and file_hash(p) == prior.get(str(p)) for p in outputs):
            log("reused")
            return
        log("started")
        try:
            subprocess.run(argv, cwd=self.work, check=True, timeout=self.cfg["limits"]["job_timeout_s"])
            if any(not p.is_file() for p in outputs):
                raise ValueError("Command did not create all declared outputs")
        except BaseException as e:
            log("failed", error=str(e), returncode=getattr(e, "returncode", None))
            raise
        self._commands[identity] = {str(p): file_hash(p) for p in outputs}
        log("completed", returncode=0, output_hashes=self._commands[identity])


def best_run(cfg):
    paths = sorted((study_dir(cfg) / "runs").glob("*/metrics.json"))
    return max(paths, key=lambda p: json.loads(p.read_text())["cv_average_precision"]).parent if paths else None


def experiment_identity(cfg, code, manifest):
    # Comments/formatting are not new experiments. Extra raw predictors are fingerprinted too.
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
    inputs = []
    for p in sorted((Path(cfg["_root"]) / "data").rglob("*")):
        if p.is_file():
            st = p.stat()
            inputs.append((str(p), st.st_size, st.st_mtime_ns))
    runtime = {n: file_hash(ROOT / "_runtime" / n) for n in ("engine.py", "geo.py", "guard.py")}
    return digest({"ast": ast.dump(tree), "study": manifest, "inputs": inputs, "runtime": runtime,
                   "cpus": cfg["limits"]["cpus"]})


FIELDS = ["attempt", "experiment_id", "status", "cv_average_precision", "description", "saga_commands", "path", "error"]


def append_ledger(root, record):
    path = root / "experiments.tsv"
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(record)
        f.flush()


def ledger(root):
    path = root / "experiments.tsv"
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def command_records(path):
    # Bounded failure audit, even when a runaway child wrote a huge command log.
    records = []
    if path.exists():
        with path.open() as f:
            for _ in range(2000):
                line = f.readline(65537)
                if not line:
                    break
                if len(line) > 65536:
                    break
                with contextlib.suppress(ValueError):
                    records.append(json.loads(line))
    return records


def run_experiment(cfg, args, prepared, manifest):
    code_path = Path(args.train).resolve()
    if code_path.stat().st_size > 1_000_000:
        raise ValueError("train.py exceeds 1 MB")
    code = code_path.read_text()
    identity = experiment_identity(cfg, code, manifest)
    root = study_dir(cfg)
    with locked(cfg):
        if (root / "holdout").exists():
            raise ValueError("Holdout evaluated: study closed to further search")
        matches = [r for r in ledger(root) if r["experiment_id"] == identity]
        if matches and not args.retry:
            print(f"Duplicate experiment {identity}; already attempted. Inspect --list or explicitly --retry.")
            return
        attempt = identity + "-" + uuid.uuid4().hex[:8]
        destination = root / "runs" / attempt
        record = {"attempt": attempt, "experiment_id": identity, "status": "running",
                  "description": args.description, "path": str(destination)}
        append_ledger(root, record)
        job = Path(tempfile.mkdtemp(prefix=".job-", dir=artifacts(cfg)))
        error = None
        try:
            (job / "train.py").write_text(code)
            write_json(job / "request.json", {"description": args.description, "experiment_id": identity})
            with (job / "worker.log").open("w") as log:
                result = supervise(cfg, [sys.executable, str(ROOT / "prepare.py"), "--config", cfg["_path"],
                                        "--experiment-worker"], job, cfg["limits"]["job_timeout_s"], log=log)
            if result:
                raise RuntimeError(f"Experiment worker exited with code {result}")
            verify_prepared(cfg)
            if experiment_identity(cfg, code, manifest) != identity:
                raise RuntimeError("Inputs or fixed runtime changed during experiment")
            metrics = json.loads((job / "metrics.json").read_text())
            record.update(status=metrics["status"], cv_average_precision=metrics["cv_average_precision"])
        except BaseException as e:
            error = e
            record.update(status="crash", error=str(e))
        finally:
            commands = command_records(job / "commands.jsonl")
            record["saga_commands"] = json.dumps([r["bash"] for r in commands
                if r.get("status") == "started" and Path(r["argv"][0]).name == "saga_cmd"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            if error is None:
                job.rename(destination)
            else:
                tail = ""
                if (job / "worker.log").exists():
                    with (job / "worker.log").open("rb") as f:
                        f.seek(max(0, (job / "worker.log").stat().st_size - 16000))
                        tail = f.read().decode(errors="replace")
                shutil.rmtree(job, ignore_errors=True)
                destination.mkdir()
                (destination / "train.py").write_text(code)
                (destination / "worker.log").write_text(tail)
                write_json(destination / "failed_commands.json", commands)
            write_json(destination / "attempt.json", record)
            append_ledger(root, record)
            shutil.rmtree(job, ignore_errors=True)
        print(json.dumps(record, indent=2))
        if error:
            raise RuntimeError(f"{error}\nSee {destination / 'worker.log'}") from error


def dispatch(cfg, args):
    root = study_dir(cfg)
    if args.list:
        latest = {r["attempt"]: r for r in ledger(root)}
        for r in latest.values():
            print("\t".join(r.get(k, "") for k in FIELDS))
        return
    prepared, manifest = verify_prepared(cfg)
    if (root / "holdout").exists():
        raise ValueError("Holdout evaluated: study closed to further search")
    if args.holdout:
        if args.worker:
            evaluate_holdout(cfg, prepared)
        else:
            run_job(cfg, "prepare.py", root / "holdout", args=["--holdout"])
            print((root / "holdout" / "metrics.json").read_text())
    elif args.experiment_worker:
        train(cfg, prepared)
    else:
        run_experiment(cfg, args, prepared, manifest)


def train(cfg, prepared):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score
    from importlib.metadata import version

    frame = pd.read_csv(prepared / "samples.csv", parse_dates=["date"])
    if len(frame) > cfg["limits"]["max_samples"]:
        raise ValueError("Prepared sample exceeds limits.max_samples")
    module = load_experiment(Path("train.py"))
    points = pd.read_csv(prepared / "points.csv", parse_dates=["date"])
    ctx = Context(cfg, points)
    matrix = module.build_features(ctx, points.copy())
    if not isinstance(matrix, pd.DataFrame) or not matrix.index.equals(frame.index):
        raise ValueError("build_features must return a DataFrame in unchanged point order")
    if not 1 <= len(matrix.columns) <= 200 or matrix.columns.duplicated().any():
        raise ValueError("Need 1..200 unique features")
    if set(matrix) & {"label", "fold", "block", "sample_id", "cell", "date", "x", "y"}:
        raise ValueError("Labels, split identifiers and coordinates are not predictors")
    if np.isinf(matrix.to_numpy(dtype=float)).any() or matrix.isna().all().any():
        raise ValueError("Features contain infinity or an entirely missing column")
    dev = frame.fold >= 0
    def model():
        return module.make_model(seed=cfg["project"]["seed"], cpus=cfg["limits"]["cpus"])

    scores, predictions = [], np.full(len(frame), np.nan)
    for fold in range(cfg["evaluation"]["folds"]):
        tr, va = dev & (frame.fold != fold), frame.fold == fold
        if frame.loc[tr, "label"].nunique() != 2 or frame.loc[va, "label"].nunique() != 2:
            raise ValueError("Every frozen training/validation fold must have both classes")
        fitted = model().fit(matrix.loc[tr], frame.loc[tr, "label"])
        prob = positive_probability(fitted, matrix.loc[va])
        predictions[va] = prob
        score = {"fold": fold, "average_precision": float(average_precision_score(frame.loc[va, "label"], prob)),
                 "roc_auc": float(roc_auc_score(frame.loc[va, "label"], prob)),
                 "prevalence": float(frame.loc[va, "label"].mean()), "n_train": int(tr.sum()), "n_validation": int(va.sum())}
        scores.append(score)
        print(score, flush=True)
    fitted = model().fit(matrix.loc[dev], frame.loc[dev, "label"])
    with Path("model.pkl").open("wb") as f:
        pickle.dump(fitted, f)
    matrix.to_csv("features.csv", index=False)
    pred = frame.loc[dev, ["sample_id", "fold", "label"]].copy()
    pred["probability"] = predictions[dev]
    pred.to_csv("cv_predictions.csv", index=False)
    result = {"cv_average_precision": float(np.mean([s["average_precision"] for s in scores])),
              "cv_roc_auc": float(np.mean([s["roc_auc"] for s in scores])), "folds": scores,
              "features": list(matrix), "missing_per_feature": matrix.isna().sum().to_dict(),
              "description": json.loads(Path("request.json").read_text())["description"], "holdout_evaluated": False,
              "interpretation": "Conditional on the sampled case/control population; not calibrated regional event rates."}
    prior = best_run(cfg)
    result["status"] = "keep" if prior is None or result["cv_average_precision"] > json.loads((prior / "metrics.json").read_text())["cv_average_precision"] else "discard"
    write_json("metrics.json", result)
    if hasattr(fitted, "term_importances"):
        write_json("importance.json", [{"term": name, "importance": float(value)} for name, value in zip(fitted.term_names_, fitted.term_importances())])
    write_json("provenance.json", {"study_manifest": json.loads((prepared / "manifest.json").read_text()),
        "feature_sha256": file_hash("features.csv"), "model_sha256": file_hash("model.pkl"),
        "packages": {n: version(n) for n in ("numpy", "pandas", "scikit-learn", "interpret", "rasterio", "xarray")}})
    print(f"cv_average_precision: {result['cv_average_precision']:.6f}", flush=True)


def evaluate_holdout(cfg, prepared):
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score
    run = best_run(cfg)
    if run is None:
        raise ValueError("Train at least one experiment first")
    frame = pd.read_csv(prepared / "samples.csv")
    provenance = json.loads((run / "provenance.json").read_text())
    for name, key in (("features.csv", "feature_sha256"), ("model.pkl", "model_sha256")):
        if file_hash(run / name) != provenance[key]:
            raise ValueError(f"Saved experiment integrity check failed: {name}")
    if file_hash(prepared / "samples.csv") != provenance["study_manifest"]["samples_sha256"]:
        raise ValueError("Saved model does not match the frozen samples")
    matrix = pd.read_csv(run / "features.csv")
    hold = frame.fold == -1
    load_experiment(run / "train.py")  # make experiment-local estimator classes importable
    with (run / "model.pkl").open("rb") as f:
        model = pickle.load(f)
    p = positive_probability(model, matrix.loc[hold])
    report = {"run": run.name, "average_precision": float(average_precision_score(frame.loc[hold, "label"], p)),
              "roc_auc": float(roc_auc_score(frame.loc[hold, "label"], p)), "n": int(hold.sum()),
              "prevalence": float(frame.loc[hold, "label"].mean()),
              "warning": "Do not tune against this holdout. Spatial holdout does not establish future-year/storm transfer."}
    write_json("metrics.json", report)
    print(report)


