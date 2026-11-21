# Données

## Jeu complet (non versionné)

*Credit Card Fraud Detection* — Kaggle (`mlg-ulb/creditcardfraud`), licence ODbL.
284 807 transactions (septembre 2013, porteurs européens, 48 h), dont 492 fraudes.

Téléchargement :

```bash
pip install kaggle          # nécessite ~/.kaggle/kaggle.json (ne JAMAIS le versionner)
kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip
```

Placer le fichier sous `data/creditcard.csv`. Il est exclu par `.gitignore`.

Empreinte SHA-256 du fichier utilisé pour la qualification :
`33a178be65174943a07d1736acf5898f6c8e1fa80950dbbe598c17eda968ec59`

Remarque : la qualification a été exécutée sur une copie publique du jeu Kaggle
(miroir GitHub `nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection`, 102 Mo, mêmes
284 807 lignes et 492 fraudes, valeurs arrondies à 9 décimales). Le fichier officiel
Kaggle (~144 Mo) a une empreinte différente : `compare_to_baseline` renverra alors
INVESTIGUER (jeu différent). Dans ce cas, régénérer explicitement la référence une fois
avec `python -m scripts.run_pipeline --write-baseline` et la versionner.

## Échantillon versionné : `sample_creditcard.csv`

5 000 lignes (4 850 normales + 150 fraudes), tirées après dédoublonnage avec
`random_state=42` par `python -m scripts.make_sample`.

**Attention :** l'échantillon est enrichi en fraudes (3 % au lieu de 0,17 %).
Il sert uniquement aux tests automatisés rapides (CI) ; il ne permet pas de
qualifier la performance du modèle.

## Contrat de données

| Colonne      | Type    | Règle                                          |
|--------------|---------|------------------------------------------------|
| Time         | float   | ≥ 0, secondes depuis la 1re transaction        |
| V1 … V28     | float   | finies (composantes PCA anonymisées)           |
| Amount       | float   | ≥ 0                                            |
| Class        | int     | ∈ {0, 1} (entraînement uniquement)             |

Aucune valeur manquante ni infinie ; aucun nom de colonne dupliqué.
Les lignes strictement dupliquées sont supprimées avant le partitionnement.
