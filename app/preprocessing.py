"""Chargement, validation du contrat de données, préparation et partitionnement.

Contrat de données (entraînement) — politique « rejeter » :
  * toutes les colonnes de REQUIRED_COLUMNS présentes, sans doublon de nom ;
  * types numériques uniquement ;
  * aucune valeur manquante ni infinie ;
  * Amount >= 0 et Time >= 0 ;
  * cible Class ∈ {0, 1}.
Toute violation lève DataValidationError avec un message explicite.

Règle de correction documentée : les lignes strictement dupliquées sont supprimées
AVANT le partitionnement (deduplicate) afin d'éviter qu'une même transaction se
retrouve à la fois dans l'entraînement et dans le test (fuite de données).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.config import (
    FEATURE_COLUMNS,
    PCA_COLUMNS,
    RANDOM_STATE,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
    VALID_SIZE,
    VALID_TARGET_VALUES,
)

logger = logging.getLogger(__name__)

ENGINEERED_COLUMNS = [*PCA_COLUMNS, "log_amount", "hour_sin", "hour_cos"]


class DataValidationError(ValueError):
    """Donnée non conforme au contrat."""


class DataLoadError(IOError):
    """Impossible de charger le fichier de données."""


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
def load_data(path: str | Path) -> pd.DataFrame:
    """Charge un CSV et signale explicitement les erreurs de chargement."""
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"Fichier introuvable : {path.name}")
    if path.suffix.lower() != ".csv":
        raise DataLoadError(f"Extension non supportée : {path.suffix!r} (attendu .csv)")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"Fichier vide : {path.name}") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataLoadError(f"Fichier illisible : {path.name}") from exc
    logger.info("Données chargées : %d lignes, %d colonnes", *df.shape)
    return df


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_data(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Vérifie le schéma, les types et les valeurs. Retourne une copie ordonnée.

    Lève DataValidationError si le contrat n'est pas respecté.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("L'entrée doit être un pandas.DataFrame")
    if df.empty or len(df) == 0:
        raise DataValidationError("Jeu de données vide")

    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        raise DataValidationError(f"Colonnes dupliquées : {sorted(set(duplicated))}")

    required = REQUIRED_COLUMNS if require_target else FEATURE_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Colonnes manquantes : {missing}")

    subset = df[required]
    non_numeric = [c for c in required if not pd.api.types.is_numeric_dtype(subset[c])]
    if non_numeric:
        raise DataValidationError(f"Types non numériques : {non_numeric}")
    bool_cols = [c for c in required if pd.api.types.is_bool_dtype(subset[c])]
    if bool_cols:
        raise DataValidationError(f"Types booléens non autorisés : {bool_cols}")

    values = subset.to_numpy(dtype=float)
    if np.isnan(values).any():
        cols = subset.columns[subset.isna().any()].tolist()
        raise DataValidationError(f"Valeurs manquantes dans : {cols}")
    if np.isinf(values).any():
        cols = subset.columns[np.isinf(values).any(axis=0)].tolist()
        raise DataValidationError(f"Valeurs infinies dans : {cols}")

    if (subset["Amount"] < 0).any():
        raise DataValidationError("Amount doit être >= 0")
    if (subset["Time"] < 0).any():
        raise DataValidationError("Time doit être >= 0")

    if require_target:
        target_values = set(pd.unique(subset[TARGET_COLUMN]).tolist())
        unexpected = target_values - VALID_TARGET_VALUES
        if unexpected:
            raise DataValidationError(f"Valeurs de cible invalides : {sorted(unexpected)}")
        # tolère 0.0 / 1.0 stockés en float mais impose un entier en sortie
        subset = subset.astype({TARGET_COLUMN: int})

    return subset.copy()


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Supprime les lignes strictement identiques (règle de correction documentée)."""
    before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(out)
    if removed:
        logger.info("Doublons supprimés : %d", removed)
    return out, removed


# --------------------------------------------------------------------------- #
# Préparation des variables (transformations SANS apprentissage)
# --------------------------------------------------------------------------- #
def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Transformation déterministe ligne à ligne (aucun paramètre appris).

    * Amount -> log1p(Amount) : réduit l'asymétrie très forte des montants ;
    * Time   -> heure de la journée encodée en sin/cos : Time brut (secondes depuis
      la 1re transaction du fichier) n'est pas généralisable hors de la fenêtre
      de collecte de 48 h, l'heure de la journée l'est davantage ;
    * V1..V28 conservées telles quelles (composantes PCA anonymisées).

    La normalisation (StandardScaler), elle, est APPRISE et vit dans le Pipeline :
    elle n'est ajustée que sur l'entraînement.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
    if missing:
        raise DataValidationError(f"Colonnes manquantes : {missing}")

    out = X[PCA_COLUMNS].astype(float).copy()
    amount = X["Amount"].astype(float).clip(lower=0.0)
    out["log_amount"] = np.log1p(amount)
    hour = (X["Time"].astype(float) % 86_400) / 3_600
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out[ENGINEERED_COLUMNS]


def preprocess_data(df: pd.DataFrame, require_target: bool = True) -> tuple[pd.DataFrame, pd.Series | None]:
    """Valide puis prépare les variables. Retourne (X préparé, y ou None).

    Garantit : même nombre de lignes en entrée et en sortie, aucune valeur non finie.
    """
    clean = validate_data(df, require_target=require_target)
    X = engineer_features(clean[FEATURE_COLUMNS])
    if not np.isfinite(X.to_numpy()).all():  # garde-fou : ne doit jamais arriver
        raise DataValidationError("Le prétraitement a produit des valeurs non finies")
    y = clean[TARGET_COLUMN] if require_target else None
    return X, y


# --------------------------------------------------------------------------- #
# Partitionnement
# --------------------------------------------------------------------------- #
@dataclass
class DataSplits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    def summary(self) -> dict:
        out = {}
        for name in ("train", "valid", "test"):
            y = getattr(self, f"y_{name}")
            out[name] = {
                "n": int(len(y)),
                "n_fraud": int(y.sum()),
                "fraud_rate": float(y.mean()),
            }
        return out


def split_data(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> DataSplits:
    """Partition stratifiée entraînement / validation / test (60 / 20 / 20).

    Le partitionnement intervient AVANT tout ajustement de transformation apprise.
    Retourne les variables BRUTES (FEATURE_COLUMNS) : le Pipeline s'occupe du reste.
    """
    clean = validate_data(df, require_target=True)
    X = clean[FEATURE_COLUMNS]
    y = clean[TARGET_COLUMN]
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=random_state
    )
    valid_ratio = VALID_SIZE / (1.0 - TEST_SIZE)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_tmp, y_tmp, test_size=valid_ratio, stratify=y_tmp, random_state=random_state
    )
    return DataSplits(X_train, y_train, X_valid, y_valid, X_test, y_test)
