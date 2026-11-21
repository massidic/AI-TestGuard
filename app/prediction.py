"""Chargement sécurisé du modèle et production des prédictions.

Politique de traitement des entrées à l'inférence (voir rapport, Partie V) :

| Situation                         | Comportement                                     |
|-----------------------------------|--------------------------------------------------|
| Champ requis manquant             | REJET (PredictionInputError)                     |
| Champ supplémentaire              | IGNORÉ + avertissement (nom du champ uniquement) |
| Type non numérique / booléen      | REJET                                            |
| NaN / ±inf                        | REJET (``on_invalid="reject"``, défaut) ou       |
|                                   | ligne marquée invalide (``on_invalid="flag"``)   |
| Amount / Time négatif             | identique à NaN                                  |
| Amount > k × max d'entraînement   | TRAITÉ + alerte ``extreme_amount``               |

Les journaux ne contiennent jamais de valeurs de transaction (montants, V1..V28).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.config import EXTREME_AMOUNT_FACTOR, FEATURE_COLUMNS
from app.training import file_sha256

logger = logging.getLogger(__name__)


class PredictionInputError(ValueError):
    """Entrée de prédiction non conforme au contrat."""


class ModelLoadError(IOError):
    """Le modèle ne peut pas être chargé de façon sûre."""


# --------------------------------------------------------------------------- #
# Chargement du modèle
# --------------------------------------------------------------------------- #
def load_model(path: str | Path, verify_hash: bool = True) -> dict[str, Any]:
    """Charge l'artefact {pipeline, threshold, metadata} et vérifie son intégrité."""
    path = Path(path)
    if not path.exists():
        raise ModelLoadError(f"Modèle introuvable : {path.name}")
    if path.suffix != ".joblib":
        raise ModelLoadError(f"Extension refusée : {path.suffix!r} (attendu .joblib)")

    if verify_hash:
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not sidecar.exists():
            raise ModelLoadError("Empreinte SHA-256 absente : chargement refusé")
        expected = sidecar.read_text(encoding="utf-8").strip()
        if file_sha256(path) != expected:
            raise ModelLoadError("Empreinte SHA-256 différente : fichier modifié ou corrompu")

    try:
        artifact = joblib.load(path)
    except Exception as exc:  # fichier tronqué, format inconnu, etc.
        raise ModelLoadError(f"Fichier modèle illisible : {path.name}") from exc

    if not isinstance(artifact, dict) or not {"pipeline", "threshold"} <= artifact.keys():
        raise ModelLoadError("Structure d'artefact inattendue")
    if not hasattr(artifact["pipeline"], "predict_proba"):
        raise ModelLoadError("L'objet chargé n'est pas un classifieur probabiliste")
    artifact.setdefault("metadata", {})
    return artifact


# --------------------------------------------------------------------------- #
# Validation des entrées
# --------------------------------------------------------------------------- #
def validate_input(X: Any) -> pd.DataFrame:
    """Convertit l'entrée (DataFrame, dict ou liste de dicts) et contrôle le schéma.

    Retourne un DataFrame float avec exactement FEATURE_COLUMNS, dans l'ordre.
    Les valeurs NaN/inf ne sont PAS rejetées ici (voir ``predict_fraud``).
    """
    if isinstance(X, dict):
        X = pd.DataFrame([X])
    elif isinstance(X, list):
        if not X or not all(isinstance(r, dict) for r in X):
            raise PredictionInputError("Liste vide ou éléments non dictionnaires")
        X = pd.DataFrame(X)
    elif not isinstance(X, pd.DataFrame):
        raise PredictionInputError(f"Type d'entrée non supporté : {type(X).__name__}")

    if len(X) == 0:
        raise PredictionInputError("Aucune transaction à scorer")
    if X.columns.duplicated().any():
        raise PredictionInputError("Champs dupliqués dans l'entrée")

    missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
    if missing:
        raise PredictionInputError(f"Champs requis manquants : {missing}")
    extra = [c for c in X.columns if c not in FEATURE_COLUMNS]
    if extra:
        logger.warning("Champs supplémentaires ignorés : %s", sorted(map(str, extra)))

    X = X[FEATURE_COLUMNS]
    bad_types = [
        c
        for c in FEATURE_COLUMNS
        if pd.api.types.is_bool_dtype(X[c]) or not pd.api.types.is_numeric_dtype(X[c])
    ]
    if bad_types:
        raise PredictionInputError(f"Types invalides (numérique attendu) : {bad_types}")
    return X.astype(float)


def _invalid_rows(X: pd.DataFrame) -> np.ndarray:
    values = X.to_numpy(dtype=float)
    invalid = ~np.isfinite(values).all(axis=1)
    invalid |= (X["Amount"].to_numpy() < 0) | (X["Time"].to_numpy() < 0)
    return invalid


# --------------------------------------------------------------------------- #
# Prédiction
# --------------------------------------------------------------------------- #
def predict_fraud(
    model: Pipeline | dict[str, Any],
    X: Any,
    threshold: float | None = None,
    on_invalid: str = "reject",
) -> pd.DataFrame:
    """Score les transactions.

    Paramètres
    ----------
    model : Pipeline entraîné ou artefact retourné par ``load_model``.
    X : DataFrame / dict / liste de dicts contenant FEATURE_COLUMNS.
    threshold : seuil de décision ; par défaut celui de l'artefact (sinon 0.5).
    on_invalid : "reject" (lève une erreur) ou "flag" (ligne marquée invalide,
        probabilité NaN, prédiction -1).

    Retour
    ------
    DataFrame avec : fraud_probability, prediction (0/1, -1 si invalide),
    valid (bool), alert (str, "" si aucune).
    """
    if on_invalid not in {"reject", "flag"}:
        raise ValueError("on_invalid doit valoir 'reject' ou 'flag'")

    metadata: dict[str, Any] = {}
    if isinstance(model, dict):
        pipeline = model["pipeline"]
        metadata = model.get("metadata", {})
        if threshold is None:
            threshold = model.get("threshold")
    else:
        pipeline = model
    threshold = 0.5 if threshold is None else float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Seuil invalide : {threshold}")

    X = validate_input(X)
    invalid = _invalid_rows(X)
    if invalid.any() and on_invalid == "reject":
        raise PredictionInputError(
            f"{int(invalid.sum())} transaction(s) avec valeurs manquantes, infinies ou négatives"
        )

    proba = np.full(len(X), np.nan)
    if (~invalid).any():
        proba[~invalid] = pipeline.predict_proba(X.loc[~invalid])[:, 1]

    prediction = np.where(invalid, -1, (proba >= threshold).astype(int))
    alerts = np.array([""] * len(X), dtype=object)
    alerts[invalid] = "invalid_input"
    amount_max = metadata.get("train_amount_max")
    if amount_max:
        extreme = (~invalid) & (X["Amount"].to_numpy() > EXTREME_AMOUNT_FACTOR * amount_max)
        alerts[extreme] = "extreme_amount"

    logger.info(
        "Scoring : %d transactions, %d invalides, %d alertes fraude",
        len(X),
        int(invalid.sum()),
        int((prediction == 1).sum()),
    )
    return pd.DataFrame(
        {
            "fraud_probability": proba,
            "prediction": prediction,
            "valid": ~invalid,
            "alert": alerts,
        },
        index=X.index,
    )
