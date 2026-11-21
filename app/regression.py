"""Comparaison d'une version candidate à la référence figée (baseline).

La référence est un fichier JSON versionné, créé explicitement
(``python -m scripts.run_pipeline --write-baseline``). Elle n'est JAMAIS recalculée
automatiquement par les tests : cela masquerait les régressions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import REGRESSION_TOLERANCES

ACCEPT, INVESTIGATE, REJECT = "ACCEPTER", "INVESTIGUER", "REJETER"


def load_baseline(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Référence absente : {path.name}. Créez-la explicitement.")
    return json.loads(path.read_text(encoding="utf-8"))


def compare_to_baseline(
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    tolerances: dict[str, dict[str, float]] | None = None,
    same_dataset: bool = True,
) -> dict[str, Any]:
    """Décision ACCEPTER / INVESTIGUER / REJETER métrique par métrique.

    Une baisse > tolérance « reject » => REJETER ; > « investigate » => INVESTIGUER.
    Si les jeux de données diffèrent, la comparaison n'est pas valable => INVESTIGUER.
    """
    tol = tolerances or REGRESSION_TOLERANCES
    details = {}
    worst = ACCEPT
    rank = {ACCEPT: 0, INVESTIGATE: 1, REJECT: 2}
    for metric, limits in tol.items():
        if metric not in baseline_metrics or metric not in candidate_metrics:
            raise KeyError(f"Métrique absente pour la comparaison : {metric}")
        base, cand = float(baseline_metrics[metric]), float(candidate_metrics[metric])
        drop = base - cand
        if drop > limits["reject"]:
            status = REJECT
        elif drop > limits["investigate"]:
            status = INVESTIGATE
        else:
            status = ACCEPT
        details[metric] = {"baseline": base, "candidate": cand, "delta": cand - base, "status": status}
        if rank[status] > rank[worst]:
            worst = status
    if not same_dataset and worst == ACCEPT:
        worst = INVESTIGATE
    return {"decision": worst, "same_dataset": same_dataset, "details": details}
