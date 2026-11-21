"""Entraîne une version candidate et la compare à la référence figée.

Exemples :
    python -m scripts.compare_to_baseline --name v1.1-sans-class-weight \
        --params "{\"C\": 0.01, \"class_weight\": null}"
    python -m scripts.compare_to_baseline --name v1.2-arbre --model tree \
        --params "{\"max_depth\": 8, \"min_samples_leaf\": 20}"

Code de sortie : 0 si ACCEPTER, 1 si REJETER, 2 si INVESTIGUER (exploitable en CI).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.config import (
    BASELINE_FULL_PATH,
    FEATURE_COLUMNS,
    FULL_DATA_PATH,
    REPORTS_DIR,
    THRESHOLD_RECALL_TARGET,
)
from app.evaluation import evaluate_model, select_threshold
from app.preprocessing import deduplicate, load_data, split_data, validate_data
from app.regression import ACCEPT, REJECT, compare_to_baseline, load_baseline
from app.training import environment_info, file_sha256, train_model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="logreg", choices=["logreg", "tree"])
    ap.add_argument("--params", default="{}")
    ap.add_argument("--data", type=Path, default=FULL_DATA_PATH)
    ap.add_argument("--baseline", type=Path, default=BASELINE_FULL_PATH)
    args = ap.parse_args(argv)

    baseline = load_baseline(args.baseline)
    params = json.loads(args.params)
    df, _ = deduplicate(validate_data(load_data(args.data)))
    splits = split_data(df)  # même graine => même jeu de validation fixe
    pipe = train_model(splits.X_train, splits.y_train, args.model, params)
    p_val = pipe.predict_proba(splits.X_valid[FEATURE_COLUMNS])[:, 1]
    thr = select_threshold(splits.y_valid, p_val, THRESHOLD_RECALL_TARGET)
    metrics = evaluate_model(splits.y_valid, (p_val >= thr["threshold"]).astype(int), p_val)

    same = file_sha256(args.data) == baseline["dataset"]["sha256"]
    result = compare_to_baseline(baseline["valid_metrics"], metrics, same_dataset=same)
    report = {
        "candidate": {
            "name": args.name,
            "model": args.model,
            "params": params,
            "threshold": thr["threshold"],
            "environment": environment_info(),
            "valid_metrics": {
                k: metrics[k] for k in ("pr_auc", "roc_auc", "recall", "precision", "f1", "fpr")
            },
        },
        "baseline": {
            "model": baseline["model_name"],
            "params": baseline["params"],
            "threshold": baseline["threshold"],
        },
        **result,
    }
    out = REPORTS_DIR / f"regression_{args.name}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "details": result["details"]}, indent=2))
    return 0 if result["decision"] == ACCEPT else (1 if result["decision"] == REJECT else 2)


if __name__ == "__main__":
    sys.exit(main())
