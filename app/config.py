"""Configuration centrale : contrat de données, graines, chemins et critères d'acceptation.

Les critères d'acceptation sont fixés AVANT toute évaluation sur le jeu de test,
à partir des exigences métier (voir rapport, section 3.4). Ils ne doivent pas être
modifiés après observation des résultats de test.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT_DIR / "models"

FULL_DATA_PATH = DATA_DIR / "creditcard.csv"
SAMPLE_DATA_PATH = DATA_DIR / "sample_creditcard.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "fraud_pipeline.joblib"
BASELINE_FULL_PATH = REPORTS_DIR / "baseline_full.json"
BASELINE_CI_PATH = REPORTS_DIR / "baseline_ci.json"

# --------------------------------------------------------------------------- #
# Contrat de données
# --------------------------------------------------------------------------- #
PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = ["Time", *PCA_COLUMNS, "Amount"]
TARGET_COLUMN = "Class"
REQUIRED_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]
VALID_TARGET_VALUES = {0, 1}

# --------------------------------------------------------------------------- #
# Reproductibilité
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TRAIN_SIZE = 0.60
VALID_SIZE = 0.20
TEST_SIZE = 0.20

# --------------------------------------------------------------------------- #
# Critères d'acceptation métier (définis a priori)
# --------------------------------------------------------------------------- #
ACCEPTANCE_CRITERIA = {
    # Au moins 3 fraudes sur 4 détectées : un FN coûte le montant fraudé + gestion.
    "recall_min": 0.75,
    # Au plus une fausse alerte par vraie fraude : capacité de l'équipe d'analystes.
    "precision_min": 0.50,
    # Au plus 1 client légitime sur 1 000 dérangé (blocage / appel de vérification).
    "fpr_max": 0.001,
    # Capacité de classement sur classe rare (indépendante du seuil).
    "pr_auc_min": 0.70,
    "roc_auc_min": 0.95,
}

# Politique de choix du seuil (appliquée sur la VALIDATION uniquement) :
# maximiser la précision sous contrainte de rappel >= cible (marge de sécurité
# de 5 points au-dessus du minimum d'acceptation).
THRESHOLD_RECALL_TARGET = 0.80

# Coûts relatifs utilisés pour l'analyse économique (unités arbitraires) :
# un FN (fraude manquée) coûte en moyenne ~25 fois un FP (vérification client).
COST_FN = 25.0
COST_FP = 1.0

# --------------------------------------------------------------------------- #
# Non-régression
# --------------------------------------------------------------------------- #
REGRESSION_TOLERANCES = {
    # baisse absolue tolérée avant "investiguer" / "rejeter"
    "pr_auc": {"investigate": 0.01, "reject": 0.02},
    "roc_auc": {"investigate": 0.005, "reject": 0.01},
    "recall": {"investigate": 0.02, "reject": 0.05},
    "precision": {"investigate": 0.03, "reject": 0.07},
}
REPRODUCIBILITY_ATOL = 1e-9  # écart toléré entre deux exécutions identiques

# --------------------------------------------------------------------------- #
# Exigences opérationnelles
# --------------------------------------------------------------------------- #
PERFORMANCE_CRITERIA = {
    "single_p95_ms_max": 50.0,  # part du modèle dans un budget d'autorisation ~300 ms
    "api_p95_ms_max": 100.0,  # validation + conversion + prédiction
    "batch_throughput_min": 5_000.0,  # transactions / seconde (scoring par lot)
    "model_size_mb_max": 50.0,
    "memory_increase_mb_max": 500.0,
}

# Alerte "montant extrême" : au-delà de k fois le maximum vu à l'entraînement.
EXTREME_AMOUNT_FACTOR = 2.0
# Seuils PSI usuels : < 0.1 stable, 0.1-0.25 surveiller, > 0.25 dérive significative.
PSI_WARNING = 0.10
PSI_ALERT = 0.25
