"""Crée l'échantillon versionné data/sample_creditcard.csv utilisé par la CI.

Échantillon stratifié ENRICHI en fraudes (≈3 %) pour que les tests rapides
disposent d'assez de positifs. Il n'est pas représentatif de la prévalence réelle
(0,17 %) : il sert à tester le code, pas à qualifier la performance du modèle.
"""

from __future__ import annotations

import pandas as pd

from app.config import FULL_DATA_PATH, RANDOM_STATE, SAMPLE_DATA_PATH, TARGET_COLUMN
from app.preprocessing import deduplicate, load_data, validate_data

N_NORMAL = 4_850
N_FRAUD = 150


def main():
    df, _ = deduplicate(validate_data(load_data(FULL_DATA_PATH)))
    normal = df[df[TARGET_COLUMN] == 0].sample(N_NORMAL, random_state=RANDOM_STATE)
    fraud = df[df[TARGET_COLUMN] == 1].sample(N_FRAUD, random_state=RANDOM_STATE)
    sample = pd.concat([normal, fraud]).sort_values("Time").reset_index(drop=True)
    sample.to_csv(SAMPLE_DATA_PATH, index=False, float_format="%.6f")
    print(f"Échantillon : {len(sample)} lignes, {int(sample[TARGET_COLUMN].sum())} fraudes")


if __name__ == "__main__":
    main()
