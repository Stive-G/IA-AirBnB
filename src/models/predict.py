"""
Prédiction du prix d'un nouveau logement AirBnB.

Ce module utilise un modèle déjà entraîné (LinearRegression) pour estimer
le prix d'une annonce hypothétique à partir de ses caractéristiques.
"""
import pandas as pd
from sklearn.linear_model import LinearRegression

# Même mapping que dans build_features.py — on en a besoin ici pour
# convertir le type de logement (chaîne) en code numérique avant la prédiction.
ROOM_TYPE_MAP = {
    "Entire home/apt": 3,
    "Hotel room": 2,
    "Private room": 1,
    "Shared room": 0,
}


def predict_price(
    model: LinearRegression,
    feature_names: list[str],
    minimum_nights: int = 2,
    number_of_reviews: int = 10,
    reviews_per_month: float = 1.0,
    calculated_host_listings_count: int = 1,
    availability_365: int = 180,
    room_type: str = "Entire home/apt",
    neighbourhood_freq: float = 0.05,
) -> float:
    """
    Prédit le prix d'un logement à partir de ses caractéristiques.

    Paramètres :
    - model            : modèle LinearRegression déjà entraîné (via train_linear)
    - feature_names    : liste des features dans l'ordre exact utilisé à l'entraînement
    - room_type        : type de logement en texte → converti en code via ROOM_TYPE_MAP
    - neighbourhood_freq : fréquence du quartier dans les données d'entraînement
                           (0.05 = quartier présent dans 5% des annonces, valeur typique)
    - les autres paramètres : valeurs par défaut représentant un logement standard

    Fonctionnement :
    1. Convertit room_type en room_type_code via ROOM_TYPE_MAP
    2. Construit un dictionnaire de toutes les features avec leurs valeurs
    3. Crée un DataFrame d'une seule ligne dans l'ordre exact de feature_names
       (scikit-learn exige que l'ordre des colonnes soit identique à l'entraînement)
    4. Appelle model.predict() et retourne le prix arrondi à 2 décimales

    Retourne : float — prix estimé en euros par nuit
    """
    room_code = ROOM_TYPE_MAP.get(room_type, 1)  # 1 = Private room si type inconnu

    # Dictionnaire de toutes les valeurs possibles pour les features
    values = {
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count": calculated_host_listings_count,
        "availability_365": availability_365,
        "room_type_code": room_code,
        "neighbourhood_freq": neighbourhood_freq,
    }

    # Construction du DataFrame dans le bon ordre de colonnes
    X = pd.DataFrame([[values[f] for f in feature_names]], columns=feature_names)
    return round(float(model.predict(X)[0]), 2)


if __name__ == "__main__":
    print("Utilisez predict_price() avec un modèle entraîné.")
