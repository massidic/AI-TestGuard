"""Fixtures partagées.

* ``make_df`` : générateur de données synthétiques conformes au contrat (tests unitaires
  rapides, sans dépendance au fichier Kaggle).
* ``sample_df`` / ``splits`` / ``artifact`` : échantillon versionné réel
  (data/sample_creditcard.csv) et pipeline entraîné dessus.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config import FEATURE_COLUMNS, PCA_COLUMNS, SAMPLE_DATA_PATH, THRESHOLD_RECALL_TARGET
from app.evaluation import select_threshold
from app.prediction import load_model
from app.preprocessing import load_data, split_data
from app.training import save_model, train_model


def _synthetic(n: int = 400, n_fraud: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {"Time": np.sort(rng.uniform(0, 172_800, n))}
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, n_fraud, replace=False)] = 1
    for col in PCA_COLUMNS:
        data[col] = rng.normal(0, 1, n)
    df = pd.DataFrame(data)
    # signal artificiel : les fraudes sont décalées sur V14 et V4
    df.loc[y == 1, "V14"] -= 4
    df.loc[y == 1, "V4"] += 3
    df["Amount"] = np.round(rng.lognormal(3, 1.2, n), 2)
    df["Class"] = y
    return df


@pytest.fixture
def make_df():
    return _synthetic


@pytest.fixture
def valid_df():
    return _synthetic()


@pytest.fixture(scope="session")
def sample_df():
    return load_data(SAMPLE_DATA_PATH)


@pytest.fixture(scope="session")
def splits(sample_df):
    return split_data(sample_df)


@pytest.fixture(scope="session")
def trained_pipeline(splits):
    return train_model(splits.X_train, splits.y_train, "logreg", {"C": 1.0})


@pytest.fixture(scope="session")
def artifact(tmp_path_factory, splits, trained_pipeline):
    p_val = trained_pipeline.predict_proba(splits.X_valid[FEATURE_COLUMNS])[:, 1]
    thr = select_threshold(splits.y_valid, p_val, THRESHOLD_RECALL_TARGET)["threshold"]
    path = tmp_path_factory.mktemp("models") / "model.joblib"
    save_model(trained_pipeline, path, thr, {"train_amount_max": float(splits.X_train["Amount"].max())})
    return load_model(path)


@pytest.fixture(scope="session")
def saved_model_path(tmp_path_factory, trained_pipeline):
    path = tmp_path_factory.mktemp("saved") / "model.joblib"
    save_model(trained_pipeline, path, 0.5)
    return path
