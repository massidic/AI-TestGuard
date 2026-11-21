"""Construction, entraînement, sélection d'hyperparamètres et sauvegarde du pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import app
from app.config import FEATURE_COLUMNS, RANDOM_STATE
from app.preprocessing import DataValidationError, engineer_features

logger = logging.getLogger(__name__)

# Grilles volontairement réduites (modèles « légers ») ; sélection sur VALIDATION.
MODEL_GRIDS: dict[str, list[dict[str, Any]]] = {
    "logreg": [{"C": c} for c in (0.01, 0.1, 1.0)],
    "tree": [{"max_depth": d, "min_samples_leaf": leaf} for d in (4, 6, 8) for leaf in (1, 20)],
}


def _make_estimator(model_name: str, params: dict[str, Any] | None, random_state: int):
    params = dict(params or {})
    if model_name == "logreg":
        defaults = {"class_weight": "balanced", "max_iter": 2000, "solver": "lbfgs"}
        defaults.update(params)
        return LogisticRegression(random_state=random_state, **defaults)
    if model_name == "tree":
        defaults = {"class_weight": "balanced", "max_depth": 6}
        defaults.update(params)
        return DecisionTreeClassifier(random_state=random_state, **defaults)
    raise ValueError(f"Modèle inconnu : {model_name!r} (attendu 'logreg' ou 'tree')")


def build_pipeline(
    model_name: str = "logreg",
    params: dict[str, Any] | None = None,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Pipeline scikit-learn : variables dérivées -> StandardScaler -> classifieur.

    Regrouper les étapes garantit que le scaler n'est ajusté que lors de
    ``pipeline.fit`` (donc sur l'entraînement) et qu'il est sauvegardé avec le modèle.
    """
    return Pipeline(
        steps=[
            ("features", FunctionTransformer(engineer_features, validate=False)),
            ("scaler", StandardScaler()),
            ("clf", _make_estimator(model_name, params, random_state)),
        ]
    )


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "logreg",
    params: dict[str, Any] | None = None,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Entraîne le pipeline. Vérifie les préconditions avant l'ajustement."""
    if len(X) == 0:
        raise DataValidationError("Impossible d'entraîner sur des données vides")
    if len(X) != len(y):
        raise DataValidationError(f"Dimensions incohérentes : X={len(X)} lignes, y={len(y)}")
    missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
    if missing:
        raise DataValidationError(f"Colonnes manquantes : {missing}")
    classes = set(pd.unique(pd.Series(y)).tolist())
    if classes != {0, 1}:
        raise DataValidationError(f"Les deux classes 0 et 1 sont requises, reçu : {classes}")

    pipeline = build_pipeline(model_name, params, random_state)
    pipeline.fit(X[FEATURE_COLUMNS], np.asarray(y).astype(int))
    logger.info("Modèle %s entraîné sur %d lignes", model_name, len(X))
    return pipeline


def select_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    model_name: str,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, Any], Pipeline, list[dict[str, Any]]]:
    """Choisit les hyperparamètres maximisant la PR-AUC sur la VALIDATION.

    Le jeu de test n'intervient jamais ici.
    """
    results = []
    best = (None, None, -np.inf)
    for params in MODEL_GRIDS[model_name]:
        pipe = train_model(X_train, y_train, model_name, params, random_state)
        proba = pipe.predict_proba(X_valid[FEATURE_COLUMNS])[:, 1]
        ap = float(average_precision_score(y_valid, proba))
        results.append({"params": params, "valid_pr_auc": ap})
        if ap > best[2]:
            best = (params, pipe, ap)
    return best[0], best[1], results


# --------------------------------------------------------------------------- #
# Sauvegarde
# --------------------------------------------------------------------------- #
def environment_info() -> dict[str, str]:
    return {
        "app_version": app.__version__,
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_model(
    pipeline: Pipeline,
    path: str | Path,
    threshold: float,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Sauvegarde {pipeline, seuil, métadonnées} + empreinte SHA-256 (fichier .sha256).

    L'empreinte permet à ``load_model`` de refuser un fichier altéré : un fichier
    joblib/pickle peut exécuter du code au chargement, il ne doit provenir que
    d'une source de confiance.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Seuil invalide : {threshold}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "threshold": float(threshold),
        "metadata": {**environment_info(), **(metadata or {})},
    }
    joblib.dump(artifact, path)
    digest = file_sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(digest, encoding="utf-8")
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(artifact["metadata"] | {"threshold": threshold, "sha256": digest}, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Modèle sauvegardé (%s)", path.name)
    return path
