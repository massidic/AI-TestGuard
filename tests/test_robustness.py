"""Partie V — Robustesse : entrées perturbées, extrêmes, invalides et dérive.

Les seuils de sensibilité ci-dessous sont des TOLÉRANCES définies a priori :
une perturbation de 1 % de l'écart-type ne doit quasiment rien changer ; on
mesure la sensibilité, on n'exige pas l'invariance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app.config import FEATURE_COLUMNS, PCA_COLUMNS
from app.monitoring import alert_rate_check, drift_report, psi
from app.prediction import PredictionInputError, predict_fraud
from app.robustness import (
    inject_missing,
    perturb_amount,
    perturb_pca_noise,
    scale_amount_distribution,
    sensitivity,
    shift_time,
)


def test_prediction_probabilities_are_valid(trained_pipeline, splits):
    probabilities = trained_pipeline.predict_proba(splits.X_test)[:, 1]
    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


# Scénario 1 — données manquantes : REJET (strict) ou MARQUAGE (flag)
def test_missing_values_rejected_or_flagged(artifact, splits):
    X = inject_missing(splits.X_test, ["V14"], frac=0.1)
    with pytest.raises(PredictionInputError):
        predict_fraud(artifact, X)
    res = predict_fraud(artifact, X, on_invalid="flag")
    assert (~res["valid"]).sum() == X["V14"].isna().sum() > 0


# Scénario 2 — infinis / non numériques : REJET
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_values_rejected(artifact, splits, value):
    X = splits.X_test.head(10).copy()
    X.iloc[2, X.columns.get_loc("Amount")] = value
    with pytest.raises(PredictionInputError):
        predict_fraud(artifact, X)


def test_non_numeric_values_rejected(artifact, splits):
    X = splits.X_test.head(3).astype(object)
    X.loc[X.index[0], "V7"] = "NaN-text"
    with pytest.raises(PredictionInputError):
        predict_fraud(artifact, X)


# Scénario 3 — montants extrêmes : TRAITEMENT + ALERTE
@pytest.mark.parametrize("amount", [1e5, 1e7, 1e12])
def test_extreme_amount_processed_with_alert(artifact, splits, amount):
    X = splits.X_test.head(20).copy()
    X["Amount"] = amount
    res = predict_fraud(artifact, X)
    assert res["fraud_probability"].between(0, 1).all()
    assert (res["alert"] == "extreme_amount").all()


# Scénario 4 — perturbations limitées : SENSIBILITÉ MESURÉE
def test_small_pca_noise_low_sensitivity(trained_pipeline, splits, artifact):
    X, y = splits.X_test, splits.y_test
    s = sensitivity(
        trained_pipeline, X, y, perturb_pca_noise(X, splits.X_train.std(), 0.01), artifact["threshold"]
    )
    assert s["n_invalid"] == 0
    assert s["mean_abs_dproba"] < 0.01
    assert s["flip_rate"] < 0.01


def test_sensitivity_increases_with_noise(trained_pipeline, splits, artifact):
    X, y, std = splits.X_test, splits.y_test, splits.X_train.std()
    d = [
        sensitivity(trained_pipeline, X, y, perturb_pca_noise(X, std, r), artifact["threshold"])[
            "mean_abs_dproba"
        ]
        for r in (0.01, 0.05, 0.20)
    ]
    assert d[0] < d[1] < d[2]


def test_amount_and_time_perturbations(trained_pipeline, splits, artifact):
    X, y = splits.X_test, splits.y_test
    for Xp in (perturb_amount(X, 0.05), shift_time(X, 60)):
        s = sensitivity(trained_pipeline, X, y, Xp, artifact["threshold"])
        assert s["mean_abs_dproba"] < 0.01
        assert s["flip_rate"] < 0.01
        assert (Xp["Amount"] >= 0).all()


def test_perturbations_do_not_modify_input(splits):
    X = splits.X_test.copy()
    perturb_pca_noise(X, splits.X_train.std(), 0.1)
    perturb_amount(X, 0.1)
    pd.testing.assert_frame_equal(X, splits.X_test)


# Scénario 5 — dérive : TRAITEMENT + ALERTE PSI
def test_psi_stable_for_same_distribution(splits):
    assert psi(splits.X_train["V14"], splits.X_valid["V14"]) < 0.1


def test_psi_detects_shift(splits):
    shifted = scale_amount_distribution(splits.X_test, 3.0)
    rep = drift_report(splits.X_train, shifted, ["Amount"])
    assert rep.loc[0, "status"] in {"SURVEILLER", "ALERTE"}
    X = splits.X_test.copy()
    X["V1"] = X["V1"] + 2 * splits.X_train["V1"].std()
    assert drift_report(splits.X_train, X, ["V1"]).loc[0, "status"] == "ALERTE"


def test_psi_rejects_empty():
    with pytest.raises(ValueError):
        psi(np.array([]), np.array([1.0]))


def test_alert_rate_check():
    assert alert_rate_check(np.array([0] * 998 + [1] * 2), 0.002)["status"] == "OK"
    assert alert_rate_check(np.array([0] * 900 + [1] * 100), 0.002)["status"] == "ALERTE"


# Scénario 6 — schéma invalide : REJET
@pytest.mark.parametrize("drop", ["Time", "V1", "Amount"])
def test_invalid_schema_rejected(artifact, splits, drop):
    with pytest.raises(PredictionInputError):
        predict_fraud(artifact, splits.X_test.drop(columns=[drop]).head(3))


# Génération de cas avec Hypothesis : toute entrée valide => probabilité valide
row_strategy = st.fixed_dictionaries(
    {
        "Time": st.floats(0, 200_000, allow_nan=False),
        "Amount": st.floats(0, 50_000, allow_nan=False),
        **{c: st.floats(-60, 60, allow_nan=False) for c in PCA_COLUMNS},
    }
)


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(row=row_strategy)
def test_any_valid_row_yields_valid_probability(artifact, row):
    res = predict_fraud(artifact, row)
    p = res["fraud_probability"].iloc[0]
    assert np.isfinite(p) and 0.0 <= p <= 1.0
    assert res["prediction"].iloc[0] in (0, 1)
    assert set(row) == set(FEATURE_COLUMNS)
