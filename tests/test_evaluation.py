"""Partie III — Tests unitaires des métriques, du choix du seuil et des critères."""

from __future__ import annotations

import numpy as np
import pytest

from app.evaluation import (
    bootstrap_ci,
    check_acceptance,
    evaluate_model,
    metrics_by_group,
    select_threshold,
)


def test_confusion_matrix_and_metrics_hand_computed():
    #           TP TP FN  FP  TN TN TN TN TN TN
    y_true = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    m = evaluate_model(y_true, y_pred)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 6)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 7)
    assert m["fnr"] == pytest.approx(1 / 3)
    assert m["accuracy"] == pytest.approx(0.8)


def test_high_accuracy_but_zero_recall():
    """Illustre le piège de l'accuracy sur classes déséquilibrées."""
    y_true = np.array([1] * 2 + [0] * 998)
    y_pred = np.zeros(1000, dtype=int)
    m = evaluate_model(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(0.998)
    assert m["recall"] == 0.0
    assert m["precision"] == 0.0  # zero_division géré sans exception


def test_auc_metrics_with_perfect_scores():
    y = [0, 0, 1, 1]
    m = evaluate_model(y, [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0


def test_auc_undefined_with_single_class():
    m = evaluate_model([0, 0, 0], [0, 0, 1], [0.1, 0.2, 0.9])
    assert m["roc_auc"] is None and m["pr_auc"] is None
    assert m["fpr"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "y_true,y_pred,y_proba,msg",
    [
        ([0, 1], [0], None, "Longueurs"),
        ([0, 2], [0, 1], None, "binaires"),
        ([0, 1], [0, 1], [0.2, 1.3], r"\[0, 1\]"),
        ([0, 1], [0, 1], [0.2, np.nan], r"\[0, 1\]"),
        ([], [], None, "Aucune"),
    ],
)
def test_invalid_inputs(y_true, y_pred, y_proba, msg):
    with pytest.raises(ValueError, match=msg):
        evaluate_model(y_true, y_pred, y_proba)


def test_cost_weights_false_negatives_more():
    m_fn = evaluate_model([1, 0], [0, 0])
    m_fp = evaluate_model([1, 0], [1, 1])
    assert m_fn["cost"] > m_fp["cost"]


def test_select_threshold_meets_recall_target():
    rng = np.random.default_rng(0)
    y = np.array([1] * 50 + [0] * 950)
    proba = np.clip(np.where(y == 1, rng.normal(0.7, 0.15, 1000), rng.normal(0.2, 0.15, 1000)), 0, 1)
    out = select_threshold(y, proba, recall_target=0.8)
    assert out["strategy"] == "max_precision_sous_contrainte_rappel"
    assert out["valid_recall"] >= 0.8
    achieved = evaluate_model(y, (proba >= out["threshold"]).astype(int))
    assert achieved["recall"] >= 0.8
    assert achieved["precision"] == pytest.approx(out["valid_precision"])


def test_select_threshold_fallback():
    # rappel cible > 1 impossible -> repli max F1
    out = select_threshold([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.3], recall_target=1.01)
    assert out["strategy"] == "repli_max_f1"


def test_check_acceptance():
    good = {"recall": 0.9, "precision": 0.8, "fpr": 0.0005, "pr_auc": 0.8, "roc_auc": 0.97}
    assert check_acceptance(good)["all_passed"]
    bad = {**good, "fpr": 0.01}
    res = check_acceptance(bad)
    assert not res["all_passed"] and not res["fpr"]["passed"] and res["recall"]["passed"]
    assert not check_acceptance({**good, "pr_auc": None})["all_passed"]


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = np.array([1] * 60 + [0] * 940)
    p = np.clip(np.where(y == 1, rng.normal(0.7, 0.2, 1000), rng.normal(0.2, 0.2, 1000)), 0, 1)
    ci = bootstrap_ci(y, p, 0.5, n_boot=200)
    point = evaluate_model(y, (p >= 0.5).astype(int), p)
    for k in ("recall", "precision"):
        assert ci[k][0] <= point[k] <= ci[k][1]


def test_metrics_by_group():
    df = metrics_by_group([1, 0, 1, 0], [1, 1, 0, 0], ["A", "A", "B", "B"])
    a = df.set_index("group").loc["A"]
    assert a["recall"] == 1.0 and a["fpr"] == 1.0
    b = df.set_index("group").loc["B"]
    assert b["recall"] == 0.0 and b["fpr"] == 0.0


def test_plots_are_written(tmp_path):
    from app.evaluation import plot_confusion, plot_roc_pr, plot_threshold_tradeoff

    rng = np.random.default_rng(2)
    y = np.array([1] * 30 + [0] * 470)
    p = np.clip(np.where(y == 1, rng.normal(0.7, 0.2, 500), rng.normal(0.2, 0.2, 500)), 0, 1)
    paths = plot_roc_pr({"A": (y, p), "B": (y, p**2)}, y, tmp_path)
    paths.append(plot_confusion(evaluate_model(y, (p > 0.5).astype(int)), "cm", tmp_path / "cm.png"))
    paths.append(plot_threshold_tradeoff(y, p, 0.5, tmp_path / "thr.png", "t"))
    for path in paths:
        assert path.exists() and path.stat().st_size > 1000
