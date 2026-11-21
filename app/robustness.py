"""Perturbations réalistes et mesure de sensibilité du modèle.

Les colonnes n'ont pas la même nature, les perturbations sont donc différenciées :
  * V1..V28 (composantes PCA standardisées) : bruit gaussien d'écart-type
    proportionnel à l'écart-type de la variable mesuré sur l'ENTRAÎNEMENT ;
  * Amount : variation multiplicative bornée (ex. ±5 %), arrondie au centime,
    jamais négative (simule frais, conversion de devise, arrondis) ;
  * Time : décalage de quelques secondes/minutes (horloge, latence de collecte).
On mesure la sensibilité (variation des probabilités, changements de classe,
variation des métriques) plutôt que d'exiger une invariance universelle.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.config import PCA_COLUMNS, RANDOM_STATE
from app.evaluation import evaluate_model


def perturb_pca_noise(
    X: pd.DataFrame, train_std: pd.Series, rel_sigma: float, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    out = X.copy()
    noise = rng.normal(0.0, 1.0, size=(len(X), len(PCA_COLUMNS)))
    out[PCA_COLUMNS] = X[PCA_COLUMNS].to_numpy() + noise * (rel_sigma * train_std[PCA_COLUMNS].to_numpy())
    return out


def perturb_amount(X: pd.DataFrame, rel: float, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    out = X.copy()
    factor = rng.uniform(1 - rel, 1 + rel, size=len(X))
    out["Amount"] = np.round(np.clip(X["Amount"].to_numpy() * factor, 0, None), 2)
    return out


def shift_time(X: pd.DataFrame, seconds: float) -> pd.DataFrame:
    out = X.copy()
    out["Time"] = np.clip(X["Time"].to_numpy() + seconds, 0, None)
    return out


def set_extreme_amount(X: pd.DataFrame, value: float) -> pd.DataFrame:
    out = X.copy()
    out["Amount"] = float(value)
    return out


def inject_missing(
    X: pd.DataFrame, columns: list[str], frac: float, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    out = X.copy().astype(float)
    mask = rng.random(len(X)) < frac
    out.loc[mask, columns] = np.nan
    return out


def scale_amount_distribution(X: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Dérive de distribution simulée : tous les montants multipliés (ex. inflation, devise)."""
    out = X.copy()
    out["Amount"] = np.round(X["Amount"].to_numpy() * factor, 2)
    return out


def sensitivity(
    pipeline, X: pd.DataFrame, y: pd.Series, X_pert: pd.DataFrame, threshold: float
) -> dict[str, Any]:
    """Compare prédictions d'origine et perturbées."""
    p0 = pipeline.predict_proba(X)[:, 1]
    valid = np.isfinite(X_pert.to_numpy(dtype=float)).all(axis=1)
    p1 = np.full(len(X), np.nan)
    if valid.any():
        p1[valid] = pipeline.predict_proba(X_pert.loc[valid])[:, 1]
    d = np.abs(p1[valid] - p0[valid])
    c0 = (p0 >= threshold).astype(int)
    c1 = (p1 >= threshold).astype(int)
    flips = (c0 != c1) & valid
    m0 = evaluate_model(y, c0, p0)
    m1 = evaluate_model(np.asarray(y)[valid], c1[valid], p1[valid]) if valid.any() else None
    y_arr = np.asarray(y)
    return {
        "n": int(len(X)),
        "n_invalid": int((~valid).sum()),
        "mean_abs_dproba": float(d.mean()) if d.size else None,
        "p99_abs_dproba": float(np.percentile(d, 99)) if d.size else None,
        "max_abs_dproba": float(d.max()) if d.size else None,
        "flip_rate": float(flips.sum() / max(valid.sum(), 1)),
        "n_flips": int(flips.sum()),
        "n_flips_fraud": int((flips & (y_arr == 1)).sum()),
        "recall_before": m0["recall"],
        "recall_after": m1["recall"] if m1 else None,
        "precision_before": m0["precision"],
        "precision_after": m1["precision"] if m1 else None,
        "fpr_before": m0["fpr"],
        "fpr_after": m1["fpr"] if m1 else None,
    }
