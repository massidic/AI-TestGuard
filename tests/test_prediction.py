"""Tests unitaires : entraînement, chargement du modèle et prédiction (contrat de sortie)."""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pytest

from app.config import FEATURE_COLUMNS
from app.prediction import ModelLoadError, PredictionInputError, load_model, predict_fraud
from app.preprocessing import DataValidationError
from app.training import build_pipeline, save_model, train_model


# --------------------------------------------------------------------------- #
# train_model
# --------------------------------------------------------------------------- #
def test_train_model_returns_fitted_pipeline(valid_df):
    pipe = train_model(valid_df[FEATURE_COLUMNS], valid_df["Class"])
    assert list(pipe.named_steps) == ["features", "scaler", "clf"]
    assert pipe.named_steps["scaler"].n_samples_seen_ == len(valid_df)


@pytest.mark.parametrize("model_name", ["logreg", "tree"])
def test_both_models_train(valid_df, model_name):
    pipe = train_model(valid_df[FEATURE_COLUMNS], valid_df["Class"], model_name)
    proba = pipe.predict_proba(valid_df[FEATURE_COLUMNS])[:, 1]
    assert proba.shape == (len(valid_df),)


def test_train_model_rejects_single_class(valid_df):
    with pytest.raises(DataValidationError, match="deux classes"):
        train_model(valid_df[FEATURE_COLUMNS], np.zeros(len(valid_df), dtype=int))


def test_train_model_rejects_length_mismatch(valid_df):
    with pytest.raises(DataValidationError, match="incohérentes"):
        train_model(valid_df[FEATURE_COLUMNS], valid_df["Class"].iloc[:-1])


def test_train_model_rejects_empty(valid_df):
    with pytest.raises(DataValidationError, match="vides"):
        train_model(valid_df[FEATURE_COLUMNS].iloc[:0], valid_df["Class"].iloc[:0])


def test_unknown_model_name():
    with pytest.raises(ValueError, match="inconnu"):
        build_pipeline("xgboost")


# --------------------------------------------------------------------------- #
# T04 — contrat de sortie de predict_fraud
# --------------------------------------------------------------------------- #
def test_probabilities_are_valid(artifact, splits):
    res = predict_fraud(artifact, splits.X_test)
    p = res["fraud_probability"].to_numpy()
    assert len(res) == len(splits.X_test)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    assert set(res["prediction"].unique()) <= {0, 1}
    assert res["valid"].all()


def test_prediction_uses_artifact_threshold(artifact, splits):
    res = predict_fraud(artifact, splits.X_test)
    expected = (res["fraud_probability"] >= artifact["threshold"]).astype(int)
    assert (res["prediction"] == expected).all()


def test_threshold_override_extremes(trained_pipeline, splits):
    all_fraud = predict_fraud(trained_pipeline, splits.X_test, threshold=0.0)
    strict = predict_fraud(trained_pipeline, splits.X_test, threshold=1.0)
    assert (all_fraud["prediction"] == 1).all()
    # seuil 1.0 : seules les probabilités exactement égales à 1 sont signalées
    expected = (strict["fraud_probability"] >= 1.0).astype(int)
    assert (strict["prediction"] == expected).all()
    assert strict["prediction"].sum() <= all_fraud["prediction"].sum()


@pytest.mark.parametrize("thr", [-0.1, 1.5])
def test_invalid_threshold(trained_pipeline, splits, thr):
    with pytest.raises(ValueError, match="Seuil"):
        predict_fraud(trained_pipeline, splits.X_test.head(3), threshold=thr)


def test_single_dict_input(artifact, splits):
    row = splits.X_test.iloc[0].to_dict()
    res = predict_fraud(artifact, row)
    assert len(res) == 1 and 0 <= res["fraud_probability"].iloc[0] <= 1


def test_list_of_dicts_input(artifact, splits):
    rows = splits.X_test.head(4).to_dict(orient="records")
    assert len(predict_fraud(artifact, rows)) == 4


@pytest.mark.parametrize("bad", [None, "transaction", 42, [], [1, 2]])
def test_unsupported_input_types(artifact, bad):
    with pytest.raises(PredictionInputError):
        predict_fraud(artifact, bad)


def test_missing_field_rejected(artifact, splits):
    with pytest.raises(PredictionInputError, match="manquants.*Amount"):
        predict_fraud(artifact, splits.X_test.drop(columns=["Amount"]).head(3))


def test_extra_field_ignored_with_warning(artifact, splits, caplog):
    X = splits.X_test.head(3).assign(customer_note="x")
    with caplog.at_level(logging.WARNING):
        res = predict_fraud(artifact, X)
    assert len(res) == 3
    assert "customer_note" in caplog.text


def test_wrong_type_rejected(artifact, splits):
    X = splits.X_test.head(3).astype(object)
    X["V2"] = "abc"
    with pytest.raises(PredictionInputError, match="Types invalides.*V2"):
        predict_fraud(artifact, X)


def test_nan_rejected_by_default(artifact, splits):
    X = splits.X_test.head(5).copy()
    X.iloc[1, 0] = np.nan
    with pytest.raises(PredictionInputError, match="1 transaction"):
        predict_fraud(artifact, X)


def test_nan_flag_mode_marks_only_invalid_rows(artifact, splits):
    X = splits.X_test.head(5).copy()
    X.iloc[1, 3] = np.nan
    X.iloc[3, 5] = np.inf
    res = predict_fraud(artifact, X, on_invalid="flag")
    assert res["valid"].tolist() == [True, False, True, False, True]
    assert (res.loc[~res["valid"], "prediction"] == -1).all()
    assert res.loc[~res["valid"], "fraud_probability"].isna().all()
    assert (res.loc[~res["valid"], "alert"] == "invalid_input").all()
    assert res.loc[res["valid"], "fraud_probability"].between(0, 1).all()


def test_invalid_on_invalid_option(artifact, splits):
    with pytest.raises(ValueError):
        predict_fraud(artifact, splits.X_test.head(2), on_invalid="ignore")


def test_negative_amount_rejected(artifact, splits):
    X = splits.X_test.head(2).copy()
    X["Amount"] = -5.0
    with pytest.raises(PredictionInputError):
        predict_fraud(artifact, X)


def test_extreme_amount_alert(artifact, splits):
    X = splits.X_test.head(3).copy()
    X["Amount"] = artifact["metadata"]["train_amount_max"] * 100
    res = predict_fraud(artifact, X)
    assert (res["alert"] == "extreme_amount").all()
    assert res["fraud_probability"].between(0, 1).all()


# --------------------------------------------------------------------------- #
# Chargement du modèle : erreurs explicites
# --------------------------------------------------------------------------- #
def test_load_missing_model(tmp_path):
    with pytest.raises(ModelLoadError, match="introuvable"):
        load_model(tmp_path / "nope.joblib")


def test_load_wrong_extension(tmp_path):
    p = tmp_path / "model.pkl"
    p.write_bytes(b"x")
    with pytest.raises(ModelLoadError, match="Extension"):
        load_model(p)


def test_load_without_hash_sidecar(tmp_path, trained_pipeline):
    p = tmp_path / "m.joblib"
    joblib.dump({"pipeline": trained_pipeline, "threshold": 0.5}, p)
    with pytest.raises(ModelLoadError, match="absente"):
        load_model(p)
    assert load_model(p, verify_hash=False)["threshold"] == 0.5


def test_load_tampered_model(tmp_path, trained_pipeline):
    p = save_model(trained_pipeline, tmp_path / "m.joblib", 0.5)
    with open(p, "ab") as fh:
        fh.write(b"tampered")
    with pytest.raises(ModelLoadError, match="SHA-256"):
        load_model(p)


def test_load_corrupted_file(tmp_path):
    p = tmp_path / "m.joblib"
    p.write_bytes(b"not a joblib file")
    with pytest.raises(ModelLoadError, match="illisible"):
        load_model(p, verify_hash=False)


def test_load_unexpected_structure(tmp_path):
    p = tmp_path / "m.joblib"
    joblib.dump({"foo": 1}, p)
    with pytest.raises(ModelLoadError, match="Structure"):
        load_model(p, verify_hash=False)


def test_save_rejects_invalid_threshold(tmp_path, trained_pipeline):
    with pytest.raises(ValueError):
        save_model(trained_pipeline, tmp_path / "m.joblib", 1.2)
