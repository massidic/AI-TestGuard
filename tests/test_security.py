"""Partie VIII — Sécurité opérationnelle : secrets, journaux, fichiers, entrées."""

from __future__ import annotations

import logging
import re

import pytest

from app.config import ROOT_DIR
from app.prediction import PredictionInputError, predict_fraud

SECRET_PATTERNS = {
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "Private key": re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "Kaggle key": re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"'),
    "Password assignment": re.compile(r"(?i)(password|passwd|secret|api_key)\s*=\s*['\"][^'\"]{6,}['\"]"),
}
SCANNED_SUFFIXES = {".py", ".yml", ".yaml", ".json", ".md", ".txt", ".toml", ".cfg", ".ini", ".env"}
EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "data"}


def _repo_files():
    for p in ROOT_DIR.rglob("*"):
        if p.is_file() and p.suffix in SCANNED_SUFFIXES and not (set(p.parts) & EXCLUDED_DIRS):
            if p.name != "test_security.py":
                yield p


def test_no_secrets_in_repository():
    findings = []
    for path in _repo_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{path.relative_to(ROOT_DIR)}: {name}")
    assert not findings, findings


def test_no_env_or_credentials_files():
    forbidden = {".env", "kaggle.json", "credentials.json", "id_rsa"}
    present = [
        p.name for p in ROOT_DIR.rglob("*") if p.name in forbidden and not (set(p.parts) & EXCLUDED_DIRS)
    ]
    assert not present


def test_gitignore_excludes_sensitive_and_heavy_files():
    content = (ROOT_DIR / ".gitignore").read_text(encoding="utf-8")
    for entry in ("data/creditcard.csv", "models/", ".env", "kaggle.json"):
        assert entry in content, f"{entry} absent du .gitignore"


def test_logs_do_not_leak_transaction_values(artifact, splits, caplog):
    X = splits.X_test.head(3).copy()
    X["Amount"] = 98765.43
    X["V14"] = -12.3456789
    with caplog.at_level(logging.DEBUG):
        predict_fraud(artifact, X)
        try:
            bad = X.copy()
            bad["V14"] = float("nan")
            predict_fraud(artifact, bad)
        except PredictionInputError as exc:
            logging.getLogger("test").error(str(exc))
    text = caplog.text
    assert "98765" not in text and "12.345" not in text
    assert "Scoring" in text  # le journal existe, mais sans valeurs sensibles


def test_error_messages_do_not_contain_values(artifact, splits):
    X = splits.X_test.head(1).copy()
    X["Amount"] = -4321.5
    with pytest.raises(PredictionInputError) as exc:
        predict_fraud(artifact, X)
    assert "4321" not in str(exc.value)


def test_extra_and_missing_fields(artifact, splits):
    row = splits.X_test.iloc[0].to_dict()
    assert len(predict_fraud(artifact, {**row, "card_number": "XXXX"})) == 1
    row.pop("V9")
    with pytest.raises(PredictionInputError, match="V9"):
        predict_fraud(artifact, row)


@pytest.mark.parametrize("value", ["12.5", None, True, [1.0], {"a": 1}])
def test_incorrect_types_rejected(artifact, splits, value):
    row = splits.X_test.iloc[0].to_dict()
    row["Amount"] = value
    with pytest.raises(PredictionInputError):
        predict_fraud(artifact, row)
