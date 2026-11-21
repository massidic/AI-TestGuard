"""Partie IX — Échec volontaire pour prouver que la CI signale bien un test en échec.

Ignoré par défaut. Activé par la variable d'environnement DEMO_FAILURE=1
(input « demo_failure » du workflow GitHub Actions lancé manuellement).
"""

import os

import pytest


@pytest.mark.skipif(os.getenv("DEMO_FAILURE") != "1", reason="démonstration d'échec désactivée")
def test_intentional_failure():
    recall_observed, recall_required = 0.60, 0.75
    assert recall_observed >= recall_required, "Échec volontaire : rappel sous le seuil d'acceptation"
