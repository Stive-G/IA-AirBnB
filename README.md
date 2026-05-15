# AirBnB - Prédiction de Prix par IA

> Prédire le prix optimal d'un nouveau logement AirBnB à partir des données réelles de **Lyon**, **Paris** et **Bordeaux** — Juin 2025.

---

## Contexte

Un propriétaire souhaite mettre son logement sur AirBnB. La question est : **quel prix proposer ?**

Ce projet applique des modèles de **régression linéaire** (simple et multiple) sur les données détaillées AirBnB (79 colonnes) issues de [Inside AirBnB](https://insideairbnb.com/get-the-data/).

---

## Données — Juin 2025

| Ville | Date | Annonces brutes | Après nettoyage | URL |
|-------|------|-----------------|-----------------|-----|
| Lyon | 15/06/2025 | 9 691 | 5 382 | `auvergne-rhone-alpes/lyon/2025-06-15` |
| Paris | 06/06/2025 | 84 055 | 52 924 | `ile-de-france/paris/2025-06-06` |
| Bordeaux | 15/06/2025 | 12 383 | 8 195 | `nouvelle-aquitaine/bordeaux/2025-06-15` |

Ces fichiers détaillés contiennent **79 colonnes** (vs 18 dans les fichiers de visualisation),
incluant les scores de reviews, les infos hôte, le nombre de chambres, la salle de bain, etc.

---

## Architecture du Projet

```
AirBnb_Projet/
│
├── README.md
├── SOUTENANCE.md               ← Guide de préparation à l'oral
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── lyon/listings_detail.csv        (18.9 Mo — 9 691 annonces × 79 colonnes)
│   │   ├── paris/listings_detail.csv       (173 Mo  — 84 055 annonces × 79 colonnes)
│   │   └── bordeaux/listings_detail.csv    (25 Mo   — 12 383 annonces × 79 colonnes)
│   └── processed/
│       ├── lyon_clean.csv       / lyon_features.csv
│       ├── paris_clean.csv      / paris_features.csv
│       └── bordeaux_clean.csv   / bordeaux_features.csv
│
├── notebook/
│   └── AirBnB_Prediction_Prix.ipynb    ← LIVRABLE : notebook unique complet
│
├── src/
│   ├── data/
│   │   ├── collect.py          Téléchargement + décompression .gz
│   │   └── preprocess.py       Sélection 26 colonnes, nettoyage, typage
│   ├── features/
│   │   └── build_features.py   Encodage, outliers, 18 features ML
│   └── models/
│       ├── train.py            Régression linéaire simple & multiple
│       └── predict.py          Prédiction pour un nouveau logement
│
└── reports/
    └── figures/                Graphiques PNG générés automatiquement
        ├── prix_distribution.png
        ├── prix_par_type.png
        ├── top_quartiers.png
        ├── boxplot_prix.png
        ├── regression_simple.png
        ├── regression_multiple.png
        ├── coefficients.png
        └── residus.png
```

---

## Pipeline

### 1. Collecte
```python
from src.data.collect import download_all
download_all(raw_dir="data/raw")   # télécharge et décompresse les .gz
```

### 2. Prétraitement (4 étapes)

| Étape | Action | Résultat |
|-------|--------|---------|
| **a. Sélection des colonnes** | 79 → 26 colonnes pertinentes | Suppression URL, textes libres... |
| **b. Suppression des doublons** | Basée sur `id` | 0 doublon détecté |
| **c. Valeurs manquantes** | `reviews_per_month` → 0, prix manquant → supprimé | Données complètes |
| **d. Conversion des types** | `price` → float, `bathrooms_text` → float, `host_is_superhost` → 0/1 | Types cohérents |

### 3. Feature Engineering

| Feature créée | Méthode | Justification |
|---------------|---------|---------------|
| `room_type_code` | Encodage ordinal (0=Shared → 3=Entire) | Ordre logique de valeur |
| `neighbourhood_freq` | Fréquence du quartier dans les données | Proxy de popularité |
| `property_type_freq` | Fréquence du type de propriété | Proxy de demande |
| Suppression outliers | Percentiles 1% – 99% sur le prix | Éviter les anomalies |

### 4. Modélisation

**Régression simple** — 1 variable : `accommodates`

**Régression multiple** — 18 variables :
`accommodates`, `bedrooms`, `beds`, `bathrooms`, `minimum_nights`,
`availability_365`, `number_of_reviews`, `reviews_per_month`,
`review_scores_rating`, `review_scores_cleanliness`, `review_scores_location`,
`host_is_superhost`, `host_response_rate`, `calculated_host_listings_count`,
`instant_bookable`, `room_type_code`, `neighbourhood_freq`, `property_type_freq`

---

## Résultats

### Lyon

| Modèle | MAE (€) | RMSE (€) | R² |
|--------|---------|----------|----|
| Régression simple | 36.46 | 54.54 | 0.27 |
| **Régression multiple** | **32.94** | **50.21** | **0.38** |

### Paris

| Modèle | MAE (€) | RMSE (€) | R² |
|--------|---------|----------|----|
| Régression simple | 117.52 | 192.41 | 0.23 |
| **Régression multiple** | **105.31** | **179.34** | **0.33** |

### Bordeaux

| Modèle | MAE (€) | RMSE (€) | R² |
|--------|---------|----------|----|
| Régression simple | 38.08 | 56.71 | 0.51 |
| **Régression multiple** | **32.54** | **49.88** | **0.62** |

**Exemple de prédiction** — T2, 4 personnes, note 4.8, superhost :

| Lyon | Paris | Bordeaux |
|------|-------|----------|
| 130.64€/nuit | 305.73€/nuit | 115.36€/nuit |

---

## Notebooks

Le projet contient **7 notebooks** :

| Notebook | Rôle |
|----------|------|
| `01_data_collection.ipynb` | Révision — Collecte & aperçu |
| `02_preprocessing.ipynb` | Révision — Nettoyage (4 étapes) |
| `03_descriptive_analysis.ipynb` | Révision — Statistiques & graphiques |
| `04_feature_engineering.ipynb` | Révision — Recodage & aberrants |
| `05_modeling.ipynb` | Révision — Régression simple & multiple |
| `06_evaluation.ipynb` | Révision — Métriques & prédiction |
| **`AirBnB_Prediction_Prix.ipynb`** | **Livrable — Notebook unique complet** |

---

## Installation

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Ordre d'exécution : `01 → 02 → 03 → 04 → 05 → 06` (ou directement `AirBnB_Prediction_Prix.ipynb`)

---

## Technologies

`Python 3.13` · `pandas` · `numpy` · `scikit-learn` · `matplotlib` · `Jupyter`

## Source des données

[https://insideairbnb.com/get-the-data/](https://insideairbnb.com/get-the-data/) — Données juin 2025
