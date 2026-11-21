"""Métriques adaptées aux classes déséquilibrées, choix du seuil et graphiques."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from app.config import ACCEPTANCE_CRITERIA, COST_FN, COST_FP, RANDOM_STATE


def _check_inputs(y_true, y_pred, y_proba=None):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    if len(y_true) == 0:
        raise ValueError("Aucune observation à évaluer")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Longueurs différentes : y_true={len(y_true)}, y_pred={len(y_pred)}")
    if not set(np.unique(y_true)) <= {0, 1} or not set(np.unique(y_pred)) <= {0, 1}:
        raise ValueError("y_true et y_pred doivent être binaires (0/1)")
    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float).ravel()
        if len(y_proba) != len(y_true):
            raise ValueError("y_proba n'a pas la même longueur que y_true")
        if not np.isfinite(y_proba).all() or (y_proba < 0).any() or (y_proba > 1).any():
            raise ValueError("y_proba doit contenir des probabilités finies dans [0, 1]")
    return y_true, y_pred, y_proba


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def evaluate_model(y_true, y_pred, y_proba=None) -> dict[str, Any]:
    """Calcule matrice de confusion et métriques (précision, rappel, F1, FPR, FNR,
    ROC-AUC, PR-AUC, accuracy, coût pondéré FN/FP)."""
    y_true, y_pred, y_proba = _check_inputs(y_true, y_pred, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": _safe_div(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "fpr": _safe_div(fp, fp + tn),
        "fnr": _safe_div(fn, fn + tp),
        "alert_rate": _safe_div(tp + fp, len(y_true)),
        "cost": float(COST_FN * fn + COST_FP * fp),
    }
    both_classes = len(np.unique(y_true)) == 2
    if y_proba is not None and both_classes:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def select_threshold(y_true, y_proba, recall_target: float) -> dict[str, float]:
    """Seuil maximisant la précision sous contrainte rappel >= recall_target.

    À appeler UNIQUEMENT avec les données de validation. Si la contrainte est
    impossible, retombe sur le seuil maximisant le F1 (et le signale).
    """
    y_true = np.asarray(y_true).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision/recall ont un élément de plus que thresholds
    precision, recall = precision[:-1], recall[:-1]
    feasible = recall >= recall_target
    if feasible.any():
        idx_candidates = np.where(feasible)[0]
        best = idx_candidates[np.argmax(precision[idx_candidates])]
        strategy = "max_precision_sous_contrainte_rappel"
    else:
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
        best = int(np.argmax(f1))
        strategy = "repli_max_f1"
    return {
        "threshold": float(thresholds[best]),
        "valid_precision": float(precision[best]),
        "valid_recall": float(recall[best]),
        "strategy": strategy,
    }


def check_acceptance(metrics: dict[str, Any], criteria: dict[str, float] | None = None) -> dict:
    """Compare les métriques aux critères d'acceptation définis a priori."""
    c = criteria or ACCEPTANCE_CRITERIA
    checks = {
        "recall": (metrics["recall"], ">=", c["recall_min"]),
        "precision": (metrics["precision"], ">=", c["precision_min"]),
        "fpr": (metrics["fpr"], "<=", c["fpr_max"]),
        "pr_auc": (metrics.get("pr_auc"), ">=", c["pr_auc_min"]),
        "roc_auc": (metrics.get("roc_auc"), ">=", c["roc_auc_min"]),
    }
    out = {}
    for name, (value, op, limit) in checks.items():
        ok = value is not None and (value >= limit if op == ">=" else value <= limit)
        out[name] = {"value": value, "rule": f"{op} {limit}", "passed": bool(ok)}
    out["all_passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    return out


def bootstrap_ci(
    y_true,
    y_proba,
    threshold: float,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = RANDOM_STATE,
) -> dict[str, tuple[float, float]]:
    """Intervalles de confiance bootstrap (percentiles) des métriques au seuil donné."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    rng = np.random.default_rng(random_state)
    keys = ["precision", "recall", "f1", "fpr", "pr_auc"]
    samples: dict[str, list[float]] = {k: [] for k in keys}
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        m = evaluate_model(yt, (y_proba[idx] >= threshold).astype(int), y_proba[idx])
        for k in keys:
            samples[k].append(m[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in samples.items()}


def metrics_by_group(y_true, y_pred, groups) -> pd.DataFrame:
    """Rappel, précision et FPR par groupe (audit d'équité si un attribut fiable existe)."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred), "g": np.asarray(groups)})
    rows = []
    for g, sub in df.groupby("g"):
        m = evaluate_model(sub["y"], sub["p"])
        rows.append(
            {
                "group": g,
                "n": m["n"],
                "n_fraud": m["n_fraud"],
                "recall": m["recall"],
                "precision": m["precision"],
                "fpr": m["fpr"],
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Graphiques
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_roc_pr(curves: dict[str, tuple[np.ndarray, np.ndarray]], y_true, out_dir: Path) -> list[Path]:
    """curves : {nom_modèle: (y_true, y_proba)}. Trace ROC et Précision-Rappel."""
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#2a6fdb", "#e07b39", "#3a9d5d", "#8e5bd0"]
    paths = []

    fig, ax = plt.subplots(figsize=(6, 5))
    for (name, (yt, yp)), col in zip(curves.items(), colors[: len(curves)], strict=True):
        fpr, tpr, _ = roc_curve(yt, yp)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC={roc_auc_score(yt, yp):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=1, label="Aléatoire")
    ax.set_xlabel("Taux de faux positifs (FPR)")
    ax.set_ylabel("Taux de vrais positifs (rappel)")
    ax.set_title("Courbes ROC — jeu de test")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "roc_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(6, 5))
    for (name, (yt, yp)), col in zip(curves.items(), colors[: len(curves)], strict=True):
        prec, rec, _ = precision_recall_curve(yt, yp)
        ax.plot(rec, prec, color=col, lw=2, label=f"{name} (AP={average_precision_score(yt, yp):.3f})")
    prevalence = float(np.mean(y_true))
    ax.axhline(prevalence, ls="--", color="#999999", lw=1, label=f"Aléatoire (prévalence={prevalence:.4f})")
    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_title("Courbes Précision-Rappel — jeu de test")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "pr_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths


def plot_confusion(metrics: dict[str, Any], title: str, path: Path) -> Path:
    plt = _plt()
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    labels = [["TN", "FP"], ["FN", "TP"]]
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(np.log1p(cm), cmap="Blues")
    for i in range(2):
        for j in range(2):
            color = "white" if np.log1p(cm[i, j]) > np.log1p(cm).max() * 0.6 else "black"
            ax.text(
                j,
                i,
                f"{labels[i][j]}\n{cm[i, j]:,}".replace(",", " "),
                ha="center",
                va="center",
                color=color,
                fontsize=12,
            )
    ax.set_xticks([0, 1], ["Prédit normal", "Prédit fraude"])
    ax.set_yticks([0, 1], ["Réel normal", "Réel fraude"])
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_threshold_tradeoff(y_true, y_proba, chosen: float, path: Path, title: str) -> Path:
    """Précision, rappel et FPR en fonction du seuil (données de validation).

    L'axe des abscisses représente 1 - seuil en échelle logarithmique : avec
    class_weight="balanced", les probabilités se concentrent près de 1 et une
    échelle linéaire masquerait la zone utile.
    """
    plt = _plt()
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)
    gaps = np.logspace(-8, np.log10(0.99), 160)
    prec, rec, fpr = [], [], []
    for g in gaps:
        m = evaluate_model(y_true, (y_proba >= 1 - g).astype(int))
        prec.append(m["precision"])
        rec.append(m["recall"])
        fpr.append(m["fpr"])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(gaps, prec, color="#2a6fdb", lw=2, label="Précision")
    ax.plot(gaps, rec, color="#e07b39", lw=2, label="Rappel")
    ax2 = ax.twinx()
    ax2.plot(gaps, np.array(fpr) * 100, color="#3a9d5d", lw=1.5, ls=":", label="FPR (%)")
    ax2.set_yscale("log")
    ax2.set_ylabel("FPR (%) — échelle log")
    ax.set_xscale("log")
    ax.invert_xaxis()
    gap = max(1 - chosen, 1e-8)
    ax.axvline(gap, color="#444444", ls="--", lw=1)
    ax.text(gap, 0.03, f" seuil = 1 - {gap:.1e}", fontsize=8)
    ax.set_xlabel("1 - seuil (échelle log ; seuil croissant vers la droite)")
    ax.set_ylabel("Précision / Rappel")
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center left", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
