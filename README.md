# AI-TestGuard

Stratégie de test et qualification d'un modèle de détection de fraude bancaire
(jeu *Credit Card Fraud Detection*, Kaggle) : tests unitaires, intégration,
robustesse, non-régression, performance, sécurité opérationnelle et CI GitHub Actions.

## Architecture

```
AI-TestGuard/
├── app/
│   ├── config.py          # contrat de données, graines, critères d'acceptation
│   ├── preprocessing.py   # load_data, validate_data, preprocess_data, split_data
│   ├── training.py        # build_pipeline, train_model, select_hyperparameters, save_model
│   ├── prediction.py      # load_model (SHA-256), validate_input, predict_fraud
│   ├── evaluation.py      # evaluate_model, select_threshold, bootstrap_ci, graphiques
│   ├── robustness.py      # perturbations réalistes, mesure de sensibilité
│   ├── monitoring.py      # PSI (dérive), contrôle du taux d'alertes
│   └── regression.py      # comparaison à la référence figée
├── scripts/
│   ├── run_pipeline.py    # pipeline de bout en bout + rapports
│   ├── benchmark.py       # protocole de mesure latence / débit / mémoire
│   ├── compare_to_baseline.py
│   └── make_sample.py
├── tests/                 # 145 tests pytest (+1 échec volontaire désactivé)
├── data/                  # README + échantillon versionné (jeu complet ignoré)
├── reports/               # résultats JSON, référence, figures
├── .github/workflows/tests.yml
├── pyproject.toml         # pytest, couverture (seuil 85 %), ruff
└── requirements.txt
```

Pipeline scikit-learn : `engineer_features` (log1p(Amount), heure sin/cos) →
`StandardScaler` (ajusté sur l'entraînement uniquement) → classifieur
(régression logistique ou arbre de décision, `class_weight="balanced"`).

## Installation (Windows PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux / macOS : `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

## Utilisation

```powershell
# Tests rapides (comme la CI des Pull Requests)
pytest -m "not slow" --cov=app --cov-report=term-missing

# Tous les tests
pytest

# Style
ruff check .
ruff format --check .

# Pipeline complet (nécessite data\creditcard.csv, voir data/README.md)
python -m scripts.run_pipeline
python -m scripts.benchmark
python -m scripts.compare_to_baseline --name v1.1 --params "{\"C\": 0.01, \"class_weight\": null}"

# Démonstration d'échec volontaire
$env:DEMO_FAILURE = "1"; pytest tests/test_ci_failure_demo.py; Remove-Item Env:DEMO_FAILURE
```

## Intégration continue

`.github/workflows/tests.yml` :

* **Push / Pull Request sur `main`** : Ruff (lint + format) puis Pytest (Python 3.10 et 3.12),
  couverture ≥ 85 % obligatoire, rapports JUnit et couverture publiés en artefacts.
* **Manuel (`workflow_dispatch`)** : option `demo_failure` (échec volontaire) et
  `full_evaluation` (évaluation complète).
* **Planifié (lundi 03:17 UTC)** : téléchargement Kaggle (secrets `KAGGLE_USERNAME`,
  `KAGGLE_KEY`), pipeline complet, benchmark, non-régression.

Protection de branche recommandée : *Settings → Branches → Add rule* sur `main`,
« Require status checks to pass » (`Style (Ruff)`, `Pytest (Python 3.10)`,
`Pytest (Python 3.12)`) et « Require a pull request before merging ».

## Résultats principaux (jeu de test, modèle retenu : régression logistique)

| Métrique | Valeur | Critère | Statut |
|---|---|---|---|
| Rappel | 0,737 | ≥ 0,75 | ✗ |
| Précision | 0,854 | ≥ 0,50 | ✓ |
| FPR | 0,021 % | ≤ 0,1 % | ✓ |
| PR-AUC | 0,678 | ≥ 0,70 | ✗ |
| ROC-AUC | 0,962 | ≥ 0,95 | ✓ |

Décision : le système réussit les tests techniques, mais le modèle n'atteint pas deux
critères de performance métier : **pas d'autorisation en mode décisionnel** ;
pilote possible uniquement en mode observation (*shadow*). Détails dans le rapport.
