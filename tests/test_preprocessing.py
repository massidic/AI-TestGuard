"""Partie II — Tests unitaires du chargement, de la validation et du prétraitement."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.preprocessing import StandardScaler

from app.config import FEATURE_COLUMNS, REQUIRED_COLUMNS
from app.preprocessing import (
    ENGINEERED_COLUMNS,
    DataLoadError,
    DataValidationError,
    deduplicate,
    engineer_features,
    load_data,
    preprocess_data,
    split_data,
    validate_data,
)


# --------------------------------------------------------------------------- #
# Exemple fourni par l'énoncé
# --------------------------------------------------------------------------- #
def test_standard_scaler():
    data = pd.DataFrame({"Amount": [100.0, 200.0, 300.0]})
    scaled = StandardScaler().fit_transform(data)
    assert np.isfinite(scaled).all()
    assert abs(scaled.mean()) < 1e-10
    assert np.isclose(scaled.std(), 1.0)


# --------------------------------------------------------------------------- #
# T01 / T02 — schéma et types
# --------------------------------------------------------------------------- #
def test_validate_accepts_valid_data_and_orders_columns(valid_df):
    shuffled = valid_df[list(reversed(valid_df.columns))]
    out = validate_data(shuffled)
    assert list(out.columns) == REQUIRED_COLUMNS
    assert len(out) == len(valid_df)
    assert out["Class"].dtype.kind == "i"


@pytest.mark.parametrize("column", ["Time", "V1", "V14", "V28", "Amount", "Class"])
def test_missing_required_column_is_rejected(valid_df, column):
    with pytest.raises(DataValidationError, match="Colonnes manquantes"):
        validate_data(valid_df.drop(columns=[column]))


def test_target_optional_for_inference(valid_df):
    out = validate_data(valid_df.drop(columns=["Class"]), require_target=False)
    assert list(out.columns) == FEATURE_COLUMNS


def test_duplicated_column_is_rejected(valid_df):
    dup = pd.concat([valid_df, valid_df[["V1"]]], axis=1)
    with pytest.raises(DataValidationError, match="dupliquées.*V1"):
        validate_data(dup)


def test_wrong_type_is_rejected(valid_df):
    df = valid_df.copy()
    df["V3"] = df["V3"].astype(str)
    with pytest.raises(DataValidationError, match="non numériques.*V3"):
        validate_data(df)


def test_boolean_type_is_rejected(valid_df):
    df = valid_df.copy()
    df["V5"] = df["V5"] > 0
    with pytest.raises(DataValidationError, match="(booléens|non numériques)"):
        validate_data(df)


def test_non_dataframe_is_rejected():
    with pytest.raises(DataValidationError, match="DataFrame"):
        validate_data([[1, 2, 3]])


# --------------------------------------------------------------------------- #
# Valeurs manquantes, infinies, vides, bornes
# --------------------------------------------------------------------------- #
def test_missing_values_are_rejected_with_column_name(valid_df):
    df = valid_df.copy()
    df.loc[3, "Amount"] = np.nan
    with pytest.raises(DataValidationError, match="manquantes.*Amount"):
        validate_data(df)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_values_are_rejected(valid_df, value):
    df = valid_df.copy()
    df.loc[0, "V10"] = value
    with pytest.raises(DataValidationError, match="infinies.*V10"):
        validate_data(df)


def test_empty_dataframe_is_rejected():
    with pytest.raises(DataValidationError, match="vide"):
        validate_data(pd.DataFrame())


def test_empty_dataframe_with_columns_is_rejected():
    with pytest.raises(DataValidationError, match="vide"):
        validate_data(pd.DataFrame(columns=REQUIRED_COLUMNS))


def test_negative_amount_is_rejected(valid_df):
    df = valid_df.copy()
    df.loc[0, "Amount"] = -0.01
    with pytest.raises(DataValidationError, match="Amount"):
        validate_data(df)


def test_zero_amount_is_accepted_boundary(valid_df):
    df = valid_df.copy()
    df.loc[0, "Amount"] = 0.0
    X, _ = preprocess_data(df)
    assert X.loc[0, "log_amount"] == 0.0


def test_negative_time_is_rejected(valid_df):
    df = valid_df.copy()
    df.loc[0, "Time"] = -1
    with pytest.raises(DataValidationError, match="Time"):
        validate_data(df)


# --------------------------------------------------------------------------- #
# Cible
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [2, -1, 0.5])
def test_invalid_target_values_are_rejected(valid_df, bad):
    df = valid_df.copy()
    df["Class"] = df["Class"].astype(float)
    df.loc[0, "Class"] = bad
    with pytest.raises(DataValidationError, match="cible invalides"):
        validate_data(df)


def test_float_target_zero_one_is_accepted_and_cast(valid_df):
    df = valid_df.copy()
    df["Class"] = df["Class"].astype(float)
    out = validate_data(df)
    assert set(out["Class"].unique()) <= {0, 1}
    assert out["Class"].dtype.kind == "i"


# --------------------------------------------------------------------------- #
# T03 — prétraitement : dimensions et valeurs finies
# --------------------------------------------------------------------------- #
def test_preprocess_dimensions_and_finiteness(valid_df):
    X, y = preprocess_data(valid_df)
    assert X.shape == (len(valid_df), len(ENGINEERED_COLUMNS))
    assert list(X.columns) == ENGINEERED_COLUMNS
    assert len(y) == len(valid_df)
    assert np.isfinite(X.to_numpy()).all()


def test_preprocess_single_observation(valid_df):
    X, y = preprocess_data(valid_df.iloc[[0]])
    assert X.shape == (1, len(ENGINEERED_COLUMNS))
    assert np.isfinite(X.to_numpy()).all()
    assert y.iloc[0] == valid_df["Class"].iloc[0]


def test_preprocess_without_target(valid_df):
    X, y = preprocess_data(valid_df.drop(columns=["Class"]), require_target=False)
    assert y is None
    assert len(X) == len(valid_df)


def test_engineer_features_values():
    df = pd.DataFrame(
        [
            {**{c: 0.0 for c in FEATURE_COLUMNS}, "Time": 0.0, "Amount": np.e - 1},
            {**{c: 1.0 for c in FEATURE_COLUMNS}, "Time": 6 * 3600.0, "Amount": 0.0},
            {**{c: 1.0 for c in FEATURE_COLUMNS}, "Time": 86_400.0, "Amount": 0.0},
        ]
    )
    out = engineer_features(df)
    assert np.isclose(out.loc[0, "log_amount"], 1.0)
    assert np.isclose(out.loc[0, "hour_sin"], 0.0) and np.isclose(out.loc[0, "hour_cos"], 1.0)
    assert np.isclose(out.loc[1, "hour_sin"], 1.0) and np.isclose(out.loc[1, "hour_cos"], 0.0, atol=1e-12)
    # 24 h plus tard = même heure de la journée
    assert np.isclose(out.loc[2, "hour_sin"], out.loc[0, "hour_sin"], atol=1e-12)
    assert (out["V1"] == df["V1"]).all()


def test_engineer_features_does_not_modify_input(valid_df):
    before = valid_df.copy()
    engineer_features(valid_df)
    pd.testing.assert_frame_equal(valid_df, before)


def test_engineer_features_missing_column():
    with pytest.raises(DataValidationError):
        engineer_features(pd.DataFrame({"Time": [0.0]}))


@settings(max_examples=60, deadline=None)
@given(
    amount=st.floats(min_value=0, max_value=1e9, allow_nan=False, allow_infinity=False),
    time=st.floats(min_value=0, max_value=1e8, allow_nan=False, allow_infinity=False),
    v=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
)
def test_engineer_features_always_finite(amount, time, v):
    row = {c: v for c in FEATURE_COLUMNS}
    row.update({"Amount": amount, "Time": time})
    out = engineer_features(pd.DataFrame([row]))
    assert np.isfinite(out.to_numpy()).all()
    assert -1.0 <= out["hour_sin"].iloc[0] <= 1.0


# --------------------------------------------------------------------------- #
# Doublons et partitionnement
# --------------------------------------------------------------------------- #
def test_deduplicate_removes_exact_duplicates(valid_df):
    df = pd.concat([valid_df, valid_df.iloc[:5]], ignore_index=True)
    out, removed = deduplicate(df)
    assert removed == 5
    assert len(out) == len(valid_df)
    assert not out.duplicated().any()


def test_split_is_stratified_disjoint_and_complete(make_df):
    df = make_df(n=2000, n_fraud=100)
    s = split_data(df)
    sizes = [len(s.y_train), len(s.y_valid), len(s.y_test)]
    assert sum(sizes) == len(df)
    assert sizes == [1200, 400, 400]
    for y in (s.y_train, s.y_valid, s.y_test):
        assert abs(y.mean() - 0.05) < 0.005
    idx = [set(s.X_train.index), set(s.X_valid.index), set(s.X_test.index)]
    assert not (idx[0] & idx[1]) and not (idx[0] & idx[2]) and not (idx[1] & idx[2])


def test_split_is_reproducible(make_df):
    df = make_df(n=1000, n_fraud=50)
    a, b = split_data(df), split_data(df)
    assert list(a.X_test.index) == list(b.X_test.index)


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
def test_load_missing_file(tmp_path):
    with pytest.raises(DataLoadError, match="introuvable"):
        load_data(tmp_path / "absent.csv")


def test_load_wrong_extension(tmp_path):
    p = tmp_path / "data.txt"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(DataLoadError, match="Extension"):
        load_data(p)


def test_load_empty_file(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(DataLoadError, match="vide"):
        load_data(p)


def test_load_roundtrip(tmp_path, valid_df):
    p = tmp_path / "ok.csv"
    valid_df.to_csv(p, index=False)
    out = load_data(p)
    assert out.shape == valid_df.shape
