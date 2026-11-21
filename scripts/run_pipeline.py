"""Pipeline de bout en bout : chargement -> validation -> split -> entraînement ->
choix du seuil (validation) -> sauvegarde -> rechargement -> test -> rapports.

Usage :
    python -m scripts.run_pipeline                      # jeu complet data/creditcard.csv
    python -m scripts.run_pipeline --data data/sample_creditcard.csv --quick
    python -m scripts.run_pipeline --write-baseline     # fige la référence de non-régression
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from app.config import (
    ACCEPTANCE_CRITERIA,
    BASELINE_FULL_PATH,
    DEFAULT_MODEL_PATH,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    FULL_DATA_PATH,
    PCA_COLUMNS,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
    THRESHOLD_RECALL_TARGET,
)
from app.evaluation import (
    bootstrap_ci,
    check_acceptance,
    evaluate_model,
    plot_confusion,
    plot_roc_pr,
    plot_threshold_tradeoff,
    select_threshold,
)
from app.monitoring import alert_rate_check, drift_report
from app.prediction import load_model, predict_fraud
from app.preprocessing import deduplicate, load_data, split_data, validate_data
from app.robustness import (
    inject_missing,
    perturb_amount,
    perturb_pca_noise,
    scale_amount_distribution,
    sensitivity,
    set_extreme_amount,
    shift_time,
)
from app.training import environment_info, file_sha256, save_model, select_hyperparameters, train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_pipeline")

MODEL_LABELS = {"logreg": "Régression logistique", "tree": "Arbre de décision"}


def _proba(pipe, X):
    return pipe.predict_proba(X[FEATURE_COLUMNS])[:, 1]


def evaluate_candidate(name, splits, quick=False):
    t0 = time.perf_counter()
    params, pipe, grid = select_hyperparameters(
        splits.X_train, splits.y_train, splits.X_valid, splits.y_valid, name
    )
    fit_seconds = time.perf_counter() - t0
    p_val = _proba(pipe, splits.X_valid)
    thr = select_threshold(splits.y_valid, p_val, THRESHOLD_RECALL_TARGET)
    t = thr["threshold"]
    p_train = _proba(pipe, splits.X_train)
    p_test = _proba(pipe, splits.X_test)
    res = {
        "model": name,
        "label": MODEL_LABELS[name],
        "best_params": params,
        "grid": grid,
        "grid_search_seconds": fit_seconds,
        "threshold": thr,
        "train": evaluate_model(splits.y_train, (p_train >= t).astype(int), p_train),
        "valid": evaluate_model(splits.y_valid, (p_val >= t).astype(int), p_val),
        "test": evaluate_model(splits.y_test, (p_test >= t).astype(int), p_test),
        "test_at_0.5": evaluate_model(splits.y_test, (p_test >= 0.5).astype(int), p_test),
    }
    res["test_ci95"] = bootstrap_ci(splits.y_test, p_test, t, n_boot=200 if quick else 1000)
    return res, pipe, p_val, p_test


def error_analysis(X_test, y_test, proba, threshold):
    y = np.asarray(y_test)
    pred = (proba >= threshold).astype(int)
    df = X_test.copy()
    df["proba"] = proba
    df["y"] = y
    df["hour"] = (df["Time"] % 86400) // 3600
    groups = {
        "TP": df[(y == 1) & (pred == 1)],
        "FN": df[(y == 1) & (pred == 0)],
        "FP": df[(y == 0) & (pred == 1)],
        "TN": df[(y == 0) & (pred == 0)],
    }
    out = {}
    for k, g in groups.items():
        out[k] = {
            "n": int(len(g)),
            "proba_median": float(g["proba"].median()) if len(g) else None,
            "proba_q90": float(g["proba"].quantile(0.9)) if len(g) else None,
            "amount_median": float(g["Amount"].median()) if len(g) else None,
            "amount_mean": float(g["Amount"].mean()) if len(g) else None,
            "amount_sum": float(g["Amount"].sum()) if len(g) else 0.0,
            "night_share_0_6h": float((g["hour"] < 6).mean()) if len(g) else None,
        }
    fn = groups["FN"]
    out["FN_far_from_threshold"] = int((fn["proba"] < 0.1).sum())
    out["FN_close_to_threshold"] = int((fn["proba"] >= threshold / 2).sum())
    fp = groups["FP"]
    out["FP_very_confident"] = int((fp["proba"] > 0.99).sum())
    return out


def robustness_scenarios(artifact, splits):
    pipe = artifact["pipeline"]
    t = artifact["threshold"]
    X, y = splits.X_test, splits.y_test
    train_std = splits.X_train.std()
    amount_max = float(splits.X_train["Amount"].max())
    rows = []

    def add(sid, scenario, amplitude, expected, X_pert, observed_behaviour=None):
        s = sensitivity(pipe, X, y, X_pert, t)
        rows.append(
            {
                "id": sid,
                "scenario": scenario,
                "amplitude": amplitude,
                "expected": expected,
                "observed": observed_behaviour or "",
                **s,
            }
        )

    # R1 : données manquantes -> rejet (strict) / marquage (flag)
    X_nan = inject_missing(X, ["V14", "Amount"], frac=0.05)
    try:
        predict_fraud(artifact, X_nan, on_invalid="reject")
        beh = "ANOMALIE : non rejeté"
    except ValueError:
        res = predict_fraud(artifact, X_nan, on_invalid="flag")
        beh = f"Rejet strict OK ; mode flag : {int((~res['valid']).sum())} lignes marquées invalides"
    add(
        "R1",
        "Valeurs manquantes (V14, Amount) sur 5 % des lignes",
        "5 % des lignes",
        "Rejeter (ou marquer invalide)",
        X_nan,
        beh,
    )

    # R2 : valeurs infinies / non numériques
    X_inf = X.copy().astype(float)
    X_inf.iloc[:10, X_inf.columns.get_loc("V1")] = np.inf
    try:
        predict_fraud(artifact, X_inf)
        beh_inf = "ANOMALIE : non rejeté"
    except ValueError:
        beh_inf = "Rejet OK"
    X_str = X.head(5).copy().astype(object)
    X_str["Amount"] = "abc"
    try:
        predict_fraud(artifact, X_str)
        beh_inf += " ; texte : ANOMALIE"
    except ValueError:
        beh_inf += " ; texte 'abc' dans Amount : rejet OK"
    add(
        "R2",
        "Valeurs infinies (V1) / non numériques (Amount)",
        "10 lignes ±inf ; 5 lignes texte",
        "Rejeter",
        X_inf,
        beh_inf,
    )

    # R3 : montants extrêmes
    X_ext = set_extreme_amount(X, 10 * amount_max)
    res = predict_fraud(artifact, X_ext)
    beh_ext = (
        f"Traité ; {int((res['alert'] == 'extreme_amount').sum())} alertes 'extreme_amount' ; "
        f"proba finies={bool(np.isfinite(res['fraud_probability']).all())}"
    )
    add(
        "R3",
        "Amount extrême (10 × max entraînement)",
        f"{10 * amount_max:,.0f}",
        "Traiter + alerte",
        X_ext,
        beh_ext,
    )

    # R4 : perturbations limitées
    for rel in (0.01, 0.05, 0.10):
        add(
            f"R4-V{int(rel * 100)}",
            "Bruit gaussien V1..V28",
            f"σ = {rel:.0%} × écart-type train",
            "Traiter ; mesurer la sensibilité",
            perturb_pca_noise(X, train_std, rel),
        )
    add(
        "R4-A5",
        "Variation multiplicative Amount",
        "±5 %, arrondi au centime",
        "Traiter ; mesurer la sensibilité",
        perturb_amount(X, 0.05),
    )
    add("R4-T60", "Décalage Time", "+60 s", "Traiter ; mesurer la sensibilité", shift_time(X, 60))

    # R5 : dérive de distribution
    X_shift = scale_amount_distribution(X, 1.5)
    drift = drift_report(splits.X_train, X_shift, ["Amount", "V14", "V4"])
    beh_shift = "PSI Amount = {:.3f} ({})".format(
        drift.loc[drift.feature == "Amount", "psi"].iloc[0],
        drift.loc[drift.feature == "Amount", "status"].iloc[0],
    )
    add(
        "R5-A",
        "Dérive : montants × 1,5 (inflation / devise)",
        "× 1,5",
        "Traiter + alerte de dérive (PSI)",
        X_shift,
        beh_shift,
    )
    X_cov = X.copy()
    X_cov[PCA_COLUMNS[:5]] = X_cov[PCA_COLUMNS[:5]] + 0.5 * train_std[PCA_COLUMNS[:5]].to_numpy()
    drift2 = drift_report(splits.X_train, X_cov, PCA_COLUMNS[:5])
    add(
        "R5-B",
        "Dérive : décalage moyen +0,5σ sur V1..V5",
        "+0,5 σ",
        "Traiter + alerte de dérive (PSI)",
        X_cov,
        f"PSI max = {drift2['psi'].max():.3f} ({drift2['status'].iloc[0]})",
    )

    # R6 : schéma invalide
    behs = []
    for label, Xbad in {
        "colonne V3 absente": X.drop(columns=["V3"]).head(5),
        "colonne en double": pd.concat([X.head(5), X.head(5)[["V1"]]], axis=1),
    }.items():
        try:
            predict_fraud(artifact, Xbad)
            behs.append(f"{label} : ANOMALIE")
        except ValueError:
            behs.append(f"{label} : rejet OK")
    res_extra = predict_fraud(artifact, X.head(5).assign(merchant_id=1))
    behs.append(f"colonne supplémentaire : ignorée ({len(res_extra)} prédictions)")
    rows.append(
        {
            "id": "R6",
            "scenario": "Schéma invalide",
            "amplitude": "—",
            "expected": "Rejeter (manquant/doublon), ignorer (supplémentaire)",
            "observed": " ; ".join(behs),
        }
    )
    return rows


def temporal_holdout(df, model_name, params):
    """Validation hors période : entraînement sur le début, test sur la fin (ordre temporel)."""
    d = df.sort_values("Time").reset_index(drop=True)
    n = len(d)
    tr, va, te = d.iloc[: int(0.6 * n)], d.iloc[int(0.6 * n) : int(0.8 * n)], d.iloc[int(0.8 * n) :]
    pipe = train_model(tr[FEATURE_COLUMNS], tr[TARGET_COLUMN], model_name, params)
    pv = _proba(pipe, va)
    thr = select_threshold(va[TARGET_COLUMN], pv, THRESHOLD_RECALL_TARGET)["threshold"]
    pt = _proba(pipe, te)
    return {
        "threshold": thr,
        "test": evaluate_model(te[TARGET_COLUMN], (pt >= thr).astype(int), pt),
        "periods_hours": [
            float(tr.Time.max() / 3600),
            float(va.Time.max() / 3600),
            float(te.Time.max() / 3600),
        ],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=FULL_DATA_PATH)
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "results.json")
    ap.add_argument("--quick", action="store_true", help="bootstrap réduit, pas de graphiques")
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--baseline-path", type=Path, default=BASELINE_FULL_PATH)
    args = ap.parse_args(argv)

    results: dict = {
        "environment": environment_info(),
        "random_state": RANDOM_STATE,
        "acceptance_criteria": ACCEPTANCE_CRITERIA,
        "threshold_recall_target": THRESHOLD_RECALL_TARGET,
    }

    # 1-2. Chargement + validation du schéma
    raw = load_data(args.data)
    clean = validate_data(raw)
    clean, n_dups = deduplicate(clean)
    results["dataset"] = {
        "file": args.data.name,
        "sha256": file_sha256(args.data),
        "n_raw": int(len(raw)),
        "n_fraud_raw": int(raw[TARGET_COLUMN].sum()),
        "duplicates_removed": int(n_dups),
        "n": int(len(clean)),
        "n_fraud": int(clean[TARGET_COLUMN].sum()),
        "fraud_rate": float(clean[TARGET_COLUMN].mean()),
        "time_span_hours": float(clean["Time"].max() / 3600),
        "amount_stats": clean["Amount"].describe().to_dict(),
    }

    # 3. Split stratifié
    splits = split_data(clean)
    results["splits"] = splits.summary()

    # Référence naïve
    dummy = DummyClassifier(strategy="most_frequent").fit(splits.X_train, splits.y_train)
    yd = dummy.predict(splits.X_test)
    results["dummy_test"] = evaluate_model(splits.y_test, yd, np.zeros(len(yd)) + 1e-12)

    # 4-5. Entraînement + hyperparamètres + seuil (validation uniquement)
    models, pipes, val_probas, test_probas = {}, {}, {}, {}
    for name in ("logreg", "tree"):
        res, pipe, pv, pt = evaluate_candidate(name, splits, args.quick)
        models[name], pipes[name], val_probas[name], test_probas[name] = res, pipe, pv, pt
        logger.info(
            "%s : valid PR-AUC=%.4f test PR-AUC=%.4f", name, res["valid"]["pr_auc"], res["test"]["pr_auc"]
        )
    results["models"] = models

    # Sélection du champion sur la VALIDATION
    champion = max(models, key=lambda m: models[m]["valid"]["pr_auc"])
    results["champion"] = champion
    thr = models[champion]["threshold"]["threshold"]

    # 6-7. Sauvegarde + rechargement
    meta = {
        "model_name": champion,
        "params": models[champion]["best_params"],
        "random_state": RANDOM_STATE,
        "train_amount_max": float(splits.X_train["Amount"].max()),
        "dataset_sha256": results["dataset"]["sha256"],
        "threshold_policy": f"max précision s.c. rappel_valid >= {THRESHOLD_RECALL_TARGET}",
    }
    save_model(pipes[champion], args.model_path, thr, meta)
    artifact = load_model(args.model_path)
    p_before = test_probas[champion]
    p_after = predict_fraud(artifact, splits.X_test)["fraud_probability"].to_numpy()
    results["reload_check"] = {
        "max_abs_diff": float(np.max(np.abs(p_before - p_after))),
        "identical_classes": bool(np.array_equal(p_before >= thr, p_after >= thr)),
        "model_size_bytes": int(args.model_path.stat().st_size),
    }

    # 8-9. Rapport sur le test (une seule fois, pour le champion figé)
    results["acceptance"] = check_acceptance(models[champion]["test"])
    results["error_analysis"] = error_analysis(splits.X_test, splits.y_test, p_before, thr)
    results["robustness"] = robustness_scenarios(artifact, splits)
    results["temporal_holdout"] = temporal_holdout(clean, champion, models[champion]["best_params"])

    # Surveillance simulée : jour 1 (référence) vs jour 2
    day1, day2 = clean[clean.Time < 86400], clean[clean.Time >= 86400]
    dr = drift_report(day1, day2, [*PCA_COLUMNS, "Amount"])
    preds_day2 = predict_fraud(artifact, day2[FEATURE_COLUMNS])["prediction"].to_numpy()
    results["monitoring"] = {
        "drift_day1_vs_day2_top5": dr.head(5).to_dict(orient="records"),
        "n_features_alert": int((dr.status == "ALERTE").sum()),
        "n_features_warning": int((dr.status == "SURVEILLER").sum()),
        "alert_rate_day2": alert_rate_check(preds_day2, models[champion]["valid"]["alert_rate"]),
        "fraud_rate_day1": float(day1[TARGET_COLUMN].mean()),
        "fraud_rate_day2": float(day2[TARGET_COLUMN].mean()),
    }

    if not args.quick:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        curves = {MODEL_LABELS[m]: (splits.y_test, test_probas[m]) for m in models}
        plot_roc_pr(curves, splits.y_test, FIGURES_DIR)
        for m in models:
            plot_confusion(
                models[m]["test"],
                f"{MODEL_LABELS[m]} — test (seuil={models[m]['threshold']['threshold']:.6f})",
                FIGURES_DIR / f"confusion_{m}.png",
            )
        plot_threshold_tradeoff(
            splits.y_valid,
            val_probas[champion],
            thr,
            FIGURES_DIR / "threshold_tradeoff.png",
            f"Compromis selon le seuil — {MODEL_LABELS[champion]} (validation)",
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("Résultats écrits dans %s", args.out)

    if args.write_baseline:
        baseline = {
            "created_by": "scripts.run_pipeline --write-baseline",
            "environment": results["environment"],
            "random_state": RANDOM_STATE,
            "model_name": champion,
            "params": models[champion]["best_params"],
            "threshold": thr,
            "dataset": {
                k: results["dataset"][k] for k in ("file", "sha256", "n", "n_fraud", "duplicates_removed")
            },
            "valid_metrics": {
                k: models[champion]["valid"][k]
                for k in ("pr_auc", "roc_auc", "recall", "precision", "f1", "fpr")
            },
        }
        args.baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        logger.info("Référence écrite dans %s", args.baseline_path)
    return results


if __name__ == "__main__":
    main()
