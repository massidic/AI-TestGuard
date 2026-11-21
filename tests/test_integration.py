"""Partie IV — Tests d'intégration de bout en bout et prévention des fuites de données."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.config import FEATURE_COLUMNS, SAMPLE_DATA_PATH, THRESHOLD_RECALL_TARGET
from app.evaluation import evaluate_model, select_threshold
from app.prediction import ModelLoadError, load_model, predict_fraud
from app.preprocessing import (
    DataLoadError,
    DataValidationError,
    deduplicate,
    engineer_features,
    load_data,
    split_data,
    validate_data,
)
from app.training import save_model, train_model

pytestmark = pytest.mark.integration


def test_end_to_end_scenario(tmp_path):
    # 1-2. charger + vérifier le schéma
    df, _ = deduplicate(validate_data(load_data(SAMPLE_DATA_PATH)))
    # 3. séparer
    s = split_data(df)
    # 4. ajuster sur l'entraînement uniquement
    pipe = train_model(s.X_train, s.y_train, "logreg", {"C": 1.0})
    # 5. seuil choisi sur la validation
    p_val = pipe.predict_proba(s.X_valid)[:, 1]
    thr = select_threshold(s.y_valid, p_val, THRESHOLD_RECALL_TARGET)["threshold"]
    # 6-7. sauvegarder / recharger
    path = save_model(pipe, tmp_path / "model.joblib", thr)
    artifact = load_model(path)
    assert artifact["threshold"] == pytest.approx(thr)
    # 8. prédire sur le test
    res = predict_fraud(artifact, s.X_test)
    assert len(res) == len(s.X_test)
    # 9. rapport de métriques
    m = evaluate_model(s.y_test, res["prediction"], res["fraud_probability"])
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == len(s.X_test)
    assert m["roc_auc"] > 0.9  # garde-fou fonctionnel sur l'échantillon (pas un critère métier)


def test_no_non_finite_after_preprocessing_in_pipeline(trained_pipeline, splits):
    Xt = trained_pipeline[:-1].transform(splits.X_test)
    assert np.isfinite(Xt).all()


def test_dimensions_consistent(trained_pipeline, splits):
    Xt = trained_pipeline[:-1].transform(splits.X_test)
    assert Xt.shape == (len(splits.X_test), 31)
    assert trained_pipeline.predict_proba(splits.X_test).shape == (len(splits.X_test), 2)


def test_scaler_fitted_on_training_data_only(trained_pipeline, splits):
    scaler = trained_pipeline.named_steps["scaler"]
    assert scaler.n_samples_seen_ == len(splits.X_train)
    train_means = engineer_features(splits.X_train).mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, train_means, rtol=1e-10, atol=1e-12)
    test_means = engineer_features(splits.X_test).mean().to_numpy()
    assert not np.allclose(scaler.mean_, test_means)


def test_test_set_disjoint_from_train_and_valid(splits):
    test_idx = set(splits.X_test.index)
    assert not test_idx & set(splits.X_train.index)
    assert not test_idx & set(splits.X_valid.index)


def test_no_duplicate_transactions_across_splits(sample_df):
    df, _ = deduplicate(validate_data(sample_df))
    s = split_data(df)
    train_rows = set(map(tuple, s.X_train.round(6).to_numpy()))
    test_rows = set(map(tuple, s.X_test.round(6).to_numpy()))
    assert not train_rows & test_rows


def test_predictions_identical_after_reload(trained_pipeline, splits, saved_model_path):
    before = trained_pipeline.predict_proba(splits.X_test)[:, 1]
    after = load_model(saved_model_path)["pipeline"].predict_proba(splits.X_test)[:, 1]
    np.testing.assert_allclose(before, after, rtol=0, atol=1e-12)


def test_explicit_errors_on_bad_inputs(tmp_path, artifact):
    with pytest.raises(DataLoadError):
        load_data(tmp_path / "missing.csv")
    with pytest.raises(ModelLoadError):
        load_model(tmp_path / "missing.joblib")
    bad = tmp_path / "bad.csv"
    bad.write_text("Time,V1\n0,abc\n")
    with pytest.raises(DataValidationError):
        validate_data(load_data(bad))


@pytest.mark.slow
def test_run_pipeline_script_on_sample(tmp_path):
    from scripts.run_pipeline import main

    out = tmp_path / "results.json"
    res = main(
        [
            "--data",
            str(SAMPLE_DATA_PATH),
            "--quick",
            "--out",
            str(out),
            "--model-path",
            str(tmp_path / "m.joblib"),
        ]
    )
    assert out.exists()
    saved = json.loads(out.read_text())
    assert saved["reload_check"]["max_abs_diff"] <= 1e-12
    assert saved["champion"] in {"logreg", "tree"}
    # le champion est choisi sur la validation, jamais sur le test
    champ = res["champion"]
    assert all(
        res["models"][champ]["valid"]["pr_auc"] >= m["valid"]["pr_auc"] for m in res["models"].values()
    )
    assert {r["id"] for r in saved["robustness"]} >= {"R1", "R2", "R3", "R6"}


def test_feature_columns_order_irrelevant(trained_pipeline, splits):
    X = splits.X_test.head(20)
    shuffled = X[list(reversed(FEATURE_COLUMNS))]
    a = predict_fraud(trained_pipeline, X)["fraud_probability"].to_numpy()
    b = predict_fraud(trained_pipeline, shuffled)["fraud_probability"].to_numpy()
    np.testing.assert_allclose(a, b)
