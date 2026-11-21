"""Surveillance post-déploiement : dérive des données (PSI) et taux d'alertes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.config import PSI_ALERT, PSI_WARNING


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index avec classes définies par quantiles de la référence."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if reference.size == 0 or current.size == 0:
        raise ValueError("Échantillons vides pour le calcul du PSI")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / reference.size
    cur_pct = np.histogram(current, edges)[0] / current.size
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_status(value: float) -> str:
    if value >= PSI_ALERT:
        return "ALERTE"
    if value >= PSI_WARNING:
        return "SURVEILLER"
    return "STABLE"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        value = psi(reference[col].to_numpy(), current[col].to_numpy())
        rows.append({"feature": col, "psi": value, "status": psi_status(value)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def alert_rate_check(
    predictions: np.ndarray, expected_rate: float, tolerance_factor: float = 2.0
) -> dict[str, Any]:
    """Compare le taux de transactions signalées à la valeur attendue (sans labels)."""
    rate = float(np.mean(np.asarray(predictions) == 1))
    low, high = expected_rate / tolerance_factor, expected_rate * tolerance_factor
    return {
        "observed_rate": rate,
        "expected_rate": expected_rate,
        "band": (low, high),
        "status": "OK" if low <= rate <= high else "ALERTE",
    }
