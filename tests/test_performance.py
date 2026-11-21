"""Partie VIII — Tests de performance (version rapide pour la CI).

Le protocole complet (1 000 répétitions, lot de 10 000) est dans scripts/benchmark.py.
Les seuils viennent de app.config.PERFORMANCE_CRITERIA.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from app.config import PERFORMANCE_CRITERIA
from app.prediction import predict_fraud

pytestmark = pytest.mark.perf


def _p95(fn, n=200):
    for _ in range(20):
        fn()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(samples, 95))


def test_single_prediction_latency_p95(trained_pipeline, splits):
    x = splits.X_test.iloc[[0]]
    p95 = _p95(lambda: trained_pipeline.predict_proba(x))
    assert p95 <= PERFORMANCE_CRITERIA["single_p95_ms_max"], f"P95={p95:.2f} ms"


def test_api_like_latency_p95(artifact, splits):
    payload = splits.X_test.iloc[0].to_dict()
    p95 = _p95(lambda: predict_fraud(artifact, payload))
    assert p95 <= PERFORMANCE_CRITERIA["api_p95_ms_max"], f"P95={p95:.2f} ms"


def test_batch_throughput(trained_pipeline, splits):
    X = splits.X_test.sample(10_000, replace=True, random_state=0)
    trained_pipeline.predict_proba(X)
    t0 = time.perf_counter()
    trained_pipeline.predict_proba(X)
    throughput = len(X) / (time.perf_counter() - t0)
    assert throughput >= PERFORMANCE_CRITERIA["batch_throughput_min"]


def test_model_size(saved_model_path):
    size_mb = saved_model_path.stat().st_size / 1e6
    assert size_mb <= PERFORMANCE_CRITERIA["model_size_mb_max"]
