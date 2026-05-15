"""
Feature engineering : recodage, outliers, préparation ML.

Ce module transforme les données nettoyées en features exploitables
par scikit-learn. Trois types de transformations :
  1. Encodage ordinal  → room_type_code
  2. Encodage fréquentiel → neighbourhood_freq, property_type_freq
  3. Suppression des outliers de prix (percentiles 1%–99%)
"""
import numpy as np
import pandas as pd

# Encodage ordinal du type de logement.
# L'ordre reflète le prix généralement observé : logement entier > hôtel > privé > partagé.
# On utilise des entiers pour que la régression linéaire puisse exploiter cet ordre.
ROOM_TYPE_MAP = {
    "Entire home/apt": 3,   # le plus cher en général
    "Hotel room": 2,
    "Private room": 1,
    "Shared room": 0,       # le moins cher
}

# Seuils pour la suppression des outliers de prix.
# On enlève le 1% le plus bas (annonces à prix anormalement bas ou nuls)
# et le 1% le plus haut (annonces de luxe atypiques qui fausseraient la régression).
PRICE_CAP_QUANTILE = 0.99
PRICE_FLOOR_QUANTILE = 0.01


def encode_room_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée la colonne `room_type_code` à partir de `room_type`.

    Utilise ROOM_TYPE_MAP pour remplacer la chaîne de caractères par un entier.
    fillna(0) : si un type inconnu est rencontré, on lui attribue 0 (shared room).
    astype(int) : force le type entier pour éviter les float avec .0.
    """
    df = df.copy()
    df["room_type_code"] = df["room_type"].map(ROOM_TYPE_MAP).fillna(0).astype(int)
    return df


def encode_neighbourhood(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée `neighbourhood_freq` = fréquence relative du quartier dans le dataset.

    Pourquoi l'encodage fréquentiel plutôt que one-hot ?
    → Le one-hot créerait des centaines de colonnes (une par quartier).
    → La fréquence capture la popularité du quartier sur une seule colonne :
      un quartier très présent dans les données = marché actif = tendance prix plus élevés.

    value_counts(normalize=True) retourne les proportions (somme = 1.0).
    """
    df = df.copy()
    col = "neighbourhood_cleansed" if "neighbourhood_cleansed" in df.columns else "neighbourhood"
    freq = df[col].value_counts(normalize=True)
    df["neighbourhood_freq"] = df[col].map(freq)
    return df


def encode_property_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée `property_type_freq` = fréquence relative du type de propriété.

    Même logique que pour le quartier : les types de propriétés les plus courants
    (ex: "Entire rental unit") ont une fréquence élevée et correspondent
    souvent aux prix du marché standard.
    """
    df = df.copy()
    if "property_type" in df.columns:
        freq = df["property_type"].value_counts(normalize=True)
        df["property_type_freq"] = df["property_type"].map(freq)
    return df


def remove_price_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes dont le prix est en dehors des percentiles 1% – 99%.

    Sans cette étape, quelques annonces à 5000€/nuit tireraient la droite de régression
    vers le haut, ce qui fausserait les prédictions pour les cas courants.

    reset_index(drop=True) : reindexe le DataFrame de 0 à N-1 après suppression.
    """
    df = df.copy()
    cap = df["price"].quantile(PRICE_CAP_QUANTILE)
    floor = df["price"].quantile(PRICE_FLOOR_QUANTILE)
    n_before = len(df)
    df = df[(df["price"] >= floor) & (df["price"] <= cap)]
    print(f"  Aberrants prix supprimés : {n_before - len(df)} lignes (seuil: {floor:.0f}€ – {cap:.0f}€)")
    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline complet de feature engineering.

    Enchaîne dans l'ordre :
    1. encode_room_type       → colonne room_type_code
    2. encode_neighbourhood   → colonne neighbourhood_freq
    3. encode_property_type   → colonne property_type_freq
    4. remove_price_outliers  → suppression des cas extrêmes

    Retourne un DataFrame enrichi prêt pour la modélisation.
    """
    df = encode_room_type(df)
    df = encode_neighbourhood(df)
    df = encode_property_type(df)
    df = remove_price_outliers(df)
    return df


# Variable unique pour la régression simple (1 seule feature).
# `accommodates` est choisie car c'est la variable la plus corrélée au prix :
# plus un logement accueille de personnes, plus il est grand, donc plus cher.
FEATURE_COLS_SIMPLE = ["accommodates"]

# 18 variables pour la régression multiple.
# On combine des features de taille (accommodates, bedrooms, beds, bathrooms),
# de qualité (review_scores_*), d'hôte (host_is_superhost, host_response_rate),
# de disponibilité (availability_365, minimum_nights) et les encodages créés ci-dessus.
FEATURE_COLS_MULTIPLE = [
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "reviews_per_month",
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_location",
    "host_is_superhost",
    "host_response_rate",
    "calculated_host_listings_count",
    "instant_bookable",
    "room_type_code",
    "neighbourhood_freq",
    "property_type_freq",
]

TARGET = "price"  # variable cible : le prix par nuit en euros


def get_X_y(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extrait les matrices X (features) et y (cible) depuis le DataFrame.

    Étapes :
    1. avail : on ne garde que les features effectivement présentes dans df
       (sécurité si une colonne est manquante)
    2. dropna() : supprime les lignes avec au moins un NaN parmi les features
       ou la cible — scikit-learn ne tolère pas les valeurs manquantes
    3. Retourne X (DataFrame des features) et y (Series du prix)
    """
    avail = [f for f in features if f in df.columns]
    df_clean = df[avail + [TARGET]].dropna()
    return df_clean[avail], df_clean[TARGET]
