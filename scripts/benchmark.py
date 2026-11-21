"""Protocole de mesure des performances d'exécution.

* Matériel : relevé automatiquement (plateforme, CPU, nb de cœurs, RAM).
* Échauffement : 50 appels non mesurés.
* Prédiction isolée : 1 000 répétitions, 1 transaction par appel, pipeline.predict_proba.
* Requête « type API » : 1 000 répétitions, dict JSON -> validation -> predict_fraud
  (inclut conversion et contrôle du schéma, mais pas la couche HTTP).
* Lot : 10 répétitions sur 10 000 transactions -> débit (transactions/s).
* Mesure : time.perf_counter ; mémoire RSS via psutil ; pic d'allocation via tracemalloc.

Usage : python -m scripts.benchmark [--model models/fraud_pipeline.joblib]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import tracemalloc
from pathlib import Path

import numpy as np
import psutil

from app.config import (
    DEFAULT_MODEL_PATH,
    FEATURE_COLUMNS,
    FULL_DATA_PATH,
    PERFORMANCE_CRITERIA,
    REPORTS_DIR,
    SAMPLE_DATA_PATH,
)
from app.prediction import load_model, predict_fraud
from app.preprocessing import load_data


def _percentiles(samples_s):
    a = np.asarray(samples_s) * 1000
    return {
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "max_ms": float(a.max()),
    }


def run(
    model_path: Path, data_path: Path, n_single: int = 1000, batch_size: int = 10_000, n_batch: int = 10
) -> dict:
    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss
    artifact = load_model(model_path)
    rss_loaded = proc.memory_info().rss
    pipe = artifact["pipeline"]

    df = load_data(data_path)
    X = df[FEATURE_COLUMNS]
    rng = np.random.default_rng(0)
    rows = X.iloc[rng.integers(0, len(X), n_single)]

    for i in range(50):  # échauffement
        pipe.predict_proba(rows.iloc[[i]])

    single = []
    for i in range(n_single):
        x = rows.iloc[[i]]
        t0 = time.perf_counter()
        pipe.predict_proba(x)
        single.append(time.perf_counter() - t0)

    payloads = rows.to_dict(orient="records")
    api = []
    for p in payloads:
        t0 = time.perf_counter()
        predict_fraud(artifact, p)
        api.append(time.perf_counter() - t0)

    batch = X.iloc[rng.integers(0, len(X), batch_size)]
    tracemalloc.start()
    batch_times = []
    for _ in range(n_batch):
        t0 = time.perf_counter()
        pipe.predict_proba(batch)
        batch_times.append(time.perf_counter() - t0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_end = proc.memory_info().rss

    throughput = batch_size / float(np.median(batch_times))
    res = {
        "protocol": {
            "warmup": 50,
            "n_single": n_single,
            "batch_size": batch_size,
            "n_batch": n_batch,
            "timer": "time.perf_counter",
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor() or "n/a",
            "cpu_count": os.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
            "python": platform.python_version(),
        },
        "single_prediction": _percentiles(single),
        "api_like_request": _percentiles(api),
        "batch": {"median_s": float(np.median(batch_times)), "throughput_tx_per_s": throughput},
        "model_size_mb": model_path.stat().st_size / 1e6,
        "memory": {
            "rss_start_mb": rss0 / 1e6,
            "rss_after_load_mb": rss_loaded / 1e6,
            "rss_end_mb": rss_end / 1e6,
            "increase_mb": (rss_end - rss0) / 1e6,
            "tracemalloc_peak_batch_mb": peak / 1e6,
        },
        "criteria": PERFORMANCE_CRITERIA,
    }
    c = PERFORMANCE_CRITERIA
    res["checks"] = {
        "single_p95": res["single_prediction"]["p95_ms"] <= c["single_p95_ms_max"],
        "api_p95": res["api_like_request"]["p95_ms"] <= c["api_p95_ms_max"],
        "throughput": throughput >= c["batch_throughput_min"],
        "model_size": res["model_size_mb"] <= c["model_size_mb_max"],
        "memory": res["memory"]["increase_mb"] <= c["memory_increase_mb_max"],
    }
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument(
        "--data", type=Path, default=FULL_DATA_PATH if FULL_DATA_PATH.exists() else SAMPLE_DATA_PATH
    )
    args = ap.parse_args(argv)
    res = run(args.model, args.data)
    out = REPORTS_DIR / "performance.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
