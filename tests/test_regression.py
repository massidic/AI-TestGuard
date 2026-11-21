"""Partie VI — Reproductibilité et non-régression par rapport à une référence FIGÉE.

La référence (reports/baseline_ci.json) a été produite une fois par :
    python -m scripts.run_pipeline --data data/sample_creditcard.csv --quick \
        --write-baseline --baseline-path reports/baseline_ci.json
Elle n'est jamais régénérée par les tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import (
    BASELINE_CI_PATH,
    FEATURE_COLUMNS,
    REPRODUCIBILITY_ATOL,
    SAMPLE_DATA_PATH,
    THRESHOLD_RECALL_TARGET,
)
from app.evaluation import evaluate_model, select_threshold
from app.preprocessing import deduplicate, load_data, split_data, validate_data
from app.regression import ACCEPT, INVESTIGATE, REJECT, compare_to_baseline, load_baseline
from app.training import file_sha256, train_model


@pytest.fixture(scope="module")
def baseline():
    return load_baseline(BASELINE_CI_PATH)


@pytest.fixture(scope="module")
def ci_splits():
    df, _ = deduplicate(validate_data(load_data(SAMPLE_DATA_PATH)))
    return split_data(df)


def _valid_metrics(splits, model_name, params):
    pipe = train_model(splits.X_train, splits.y_train, model_name, params)
    p = pipe.predict_proba(splits.X_valid[FEATURE_COLUMNS])[:, 1]
    thr = select_threshold(splits.y_valid, p, THRESHOLD_RECALL_TARGET)["threshold"]
    return evaluate_model(splits.y_valid, (p >= thr).astype(int), p), thr


# T06 — reproductibilité
def test_training_is_reproducible(ci_splits):
    a = train_model(ci_splits.X_train, ci_splits.y_train, "tree", {"max_depth": 6})
    b = train_model(ci_splits.X_train, ci_splits.y_train, "tree", {"max_depth": 6})
    np.testing.assert_allclose(
        a.predict_proba(ci_splits.X_test), b.predict_proba(ci_splits.X_test), atol=REPRODUCIBILITY_ATOL
    )


def test_baseline_contains_required_fields(baseline):
    for key in (
        "environment",
        "random_state",
        "model_name",
        "params",
        "threshold",
        "dataset",
        "valid_metrics",
    ):
        assert key in baseline
    assert {"app_version", "scikit-learn", "numpy", "pandas"} <= baseline["environment"].keys()


def test_dataset_matches_baseline(baseline):
    assert file_sha256(SAMPLE_DATA_PATH) == baseline["dataset"]["sha256"], (
        "L'échantillon a changé : la comparaison à la référence n'est plus valable"
    )


def test_current_version_does_not_regress(ci_splits, baseline):
    metrics, thr = _valid_metrics(ci_splits, baseline["model_name"], baseline["params"])
    result = compare_to_baseline(baseline["valid_metrics"], metrics)
    assert result["decision"] == ACCEPT, result["details"]
    assert thr == pytest.approx(baseline["threshold"], abs=1e-6)


def test_degraded_version_is_detected(ci_splits, baseline):
    """Un modèle volontairement dégradé doit être rejeté (le contrôle détecte bien)."""
    metrics, _ = _valid_metrics(ci_splits, "tree", {"max_depth": 1})
    assert compare_to_baseline(baseline["valid_metrics"], metrics)["decision"] == REJECT


# Logique de décision
BASE = {"pr_auc": 0.80, "roc_auc": 0.97, "recall": 0.80, "precision": 0.85}


def test_decision_accept_on_improvement():
    cand = {k: v + 0.01 for k, v in BASE.items()}
    assert compare_to_baseline(BASE, cand)["decision"] == ACCEPT


def test_decision_investigate_on_small_drop():
    cand = {**BASE, "pr_auc": BASE["pr_auc"] - 0.015}
    assert compare_to_baseline(BASE, cand)["decision"] == INVESTIGATE


def test_decision_reject_on_large_drop():
    cand = {**BASE, "recall": BASE["recall"] - 0.10}
    res = compare_to_baseline(BASE, cand)
    assert res["decision"] == REJECT and res["details"]["recall"]["status"] == REJECT


def test_decision_investigate_if_dataset_differs():
    assert compare_to_baseline(BASE, BASE, same_dataset=False)["decision"] == INVESTIGATE


def test_missing_metric_raises():
    with pytest.raises(KeyError):
        compare_to_baseline(BASE, {"pr_auc": 0.8})


def test_missing_baseline_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_baseline(tmp_path / "absent.json")
