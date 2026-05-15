"""
Prétraitement des données AirBnB (fichiers détaillés, 79 colonnes).

Pipeline : load_raw -> select_columns -> remove_duplicates -> handle_missing
           -> convert_types -> filtre prix > 0 -> save_processed
"""
import re
import pandas as pd
from pathlib import Path

# On garde 26 colonnes sur 79 : celles qui ont un sens pour prédire le prix.
# Les colonnes écartées sont : URLs, textes libres (description, house_rules...),
# photos, identifiants inutiles, données calendrier trop granulaires.
SELECTED_COLUMNS = [
    "id",                              # identifiant unique (pour détecter les doublons)
    "neighbourhood_cleansed",          # quartier normalisé par AirBnB
    "room_type",                       # Entire home / Private room / Shared room / Hotel
    "property_type",                   # type précis : "Entire rental unit", "Private room in condo"...
    "accommodates",                    # nombre max de personnes — variable la plus corrélée au prix
    "bedrooms",                        # nombre de chambres
    "beds",                            # nombre de lits (peut différer de bedrooms)
    "bathrooms_text",                  # ex: "1 bath", "1.5 baths" — sera converti en float
    "price",                           # VARIABLE CIBLE — ex: "$120.00"
    "minimum_nights",                  # durée minimale de séjour imposée par l'hôte
    "maximum_nights",                  # durée maximale
    "availability_365",                # nombre de jours disponibles sur l'année
    "number_of_reviews",               # total avis reçus
    "number_of_reviews_ltm",           # avis sur les 12 derniers mois (ltm = last twelve months)
    "reviews_per_month",               # fréquence moyenne des avis
    "review_scores_rating",            # note globale /5
    "review_scores_cleanliness",       # note propreté
    "review_scores_location",          # note localisation
    "review_scores_value",             # rapport qualité/prix perçu
    "host_is_superhost",               # "t" ou "f" — sera converti en 1/0
    "host_response_rate",              # "95%" — sera converti en 95.0
    "host_acceptance_rate",            # taux d'acceptation des demandes
    "calculated_host_listings_count",  # nombre d'annonces gérées par cet hôte
    "instant_bookable",                # "t"/"f" — réservation sans validation de l'hôte
    "latitude",                        # coordonnées GPS
    "longitude",
]

# Stratégie de remplissage des valeurs manquantes.
# Pour les scores : 0 signifie "pas encore noté" (logement récent), ce n'est pas une note de 0.
# Pour bedrooms/beds : on met 1 par défaut (le cas le plus courant).
FILL_STRATEGIES = {
    "reviews_per_month": 0.0,
    "number_of_reviews": 0,
    "number_of_reviews_ltm": 0,
    "review_scores_rating": 0.0,
    "review_scores_cleanliness": 0.0,
    "review_scores_location": 0.0,
    "review_scores_value": 0.0,
    "bedrooms": 1.0,
    "beds": 1.0,
}


def load_raw(city: str, raw_dir: str = "data/raw") -> pd.DataFrame:
    """
    Charge le fichier CSV brut (79 colonnes) pour une ville donnée.

    low_memory=False évite un avertissement pandas sur les colonnes à types mixtes
    (fréquent sur des fichiers aussi volumineux que celui de Paris, 173 Mo).
    """
    path = Path(raw_dir) / city / "listings_detail.csv"
    return pd.read_csv(path, low_memory=False)


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Réduit le DataFrame de 79 à 26 colonnes pertinentes.

    On utilise une compréhension de liste pour ne garder que les colonnes
    qui existent réellement dans le fichier (robustesse si une colonne manque).
    """
    cols = [c for c in SELECTED_COLUMNS if c in df.columns]
    return df[cols].copy()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes ayant le même identifiant `id`.

    Sur les données AirBnB, les doublons sont rares mais peuvent survenir
    si une annonce apparaît dans plusieurs catégories de la plateforme.
    """
    n_before = len(df)
    df = df.drop_duplicates(subset=["id"])
    removed = n_before - len(df)
    if removed:
        print(f"  {removed} doublon(s) supprimé(s)")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gère les valeurs manquantes selon FILL_STRATEGIES.

    - Les scores à NaN concernent des logements sans avis → on met 0.
    - Les lignes sans `price` sont supprimées car on ne peut pas entraîner
      un modèle sans la variable cible.
    """
    for col, val in FILL_STRATEGIES.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    df = df.dropna(subset=["price"])  # supprime les annonces sans prix renseigné
    return df


def _extract_bathrooms(text: str) -> float:
    """
    Convertit le texte 'bathrooms_text' en nombre flottant.

    Exemples de valeurs rencontrées dans les données :
      "1 bath"       → 1.0
      "1.5 baths"    → 1.5
      "Half-bath"    → 0.0  (aucun chiffre trouvé → 1.0 par défaut)
      NaN            → 1.0

    On utilise re.findall pour extraire le premier nombre présent dans la chaîne.
    """
    if pd.isna(text):
        return 1.0
    nums = re.findall(r"[\d.]+", str(text))
    return float(nums[0]) if nums else 1.0


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit toutes les colonnes dans leurs types corrects.

    Transformations effectuées :
    - price        : "$150.00" → 150.0  (suppression du $ et de la virgule)
    - bathrooms_text : "1.5 baths" → 1.5  (via _extract_bathrooms)
    - host_response_rate / host_acceptance_rate : "95%" → 95.0
    - host_is_superhost / instant_bookable : "t"/"f" → 1/0
    - colonnes entières (accommodates, minimum_nights...) : → int

    L'option errors='coerce' dans pd.to_numeric transforme les valeurs
    non convertibles en NaN (plutôt que de planter), qu'on remplace ensuite par 0.
    """
    # Prix : supprimer $ et , puis convertir en float
    if "price" in df.columns:
        df["price"] = (
            df["price"]
            .astype(str)
            .str.replace(r"[$,]", "", regex=True)
            .pipe(pd.to_numeric, errors="coerce")
        )

    # Salle de bain : extraire le nombre depuis le texte
    if "bathrooms_text" in df.columns:
        df["bathrooms"] = df["bathrooms_text"].apply(_extract_bathrooms)
        df = df.drop(columns=["bathrooms_text"])  # on n'a plus besoin du texte brut

    # Pourcentages : enlever le symbole % et convertir
    pct_cols = ["host_response_rate", "host_acceptance_rate"]
    for col in pct_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace("%", "", regex=False)
                .pipe(pd.to_numeric, errors="coerce")
                .fillna(0.0)
            )

    # Booléens AirBnB : "t" = True = 1, "f" = False = 0
    bool_cols = ["host_is_superhost", "instant_bookable"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({"t": 1, "f": 0, True: 1, False: 0}).fillna(0).astype(int)

    # Entiers : forcer le type int (accommodates, minimum_nights, etc.)
    int_cols = [
        "minimum_nights", "maximum_nights", "number_of_reviews",
        "number_of_reviews_ltm", "calculated_host_listings_count",
        "availability_365", "accommodates",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


def preprocess(city: str, raw_dir: str = "data/raw") -> pd.DataFrame:
    """
    Pipeline complet de nettoyage pour une ville.

    Enchaîne dans l'ordre :
    load_raw → select_columns → remove_duplicates → handle_missing
    → convert_types → suppression des prix nuls ou manquants

    Retourne un DataFrame propre prêt pour le feature engineering.
    """
    print(f"\n=== Prétraitement : {city.upper()} ===")
    df = load_raw(city, raw_dir)
    print(f"  Chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    df = select_columns(df)
    df = remove_duplicates(df)
    df = handle_missing(df)
    df = convert_types(df)
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]  # supprime les annonces à 0€ (erreurs de saisie)
    print(f"  Après nettoyage : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    return df.reset_index(drop=True)


def save_processed(df: pd.DataFrame, city: str, out_dir: str = "data/processed") -> None:
    """
    Sauvegarde le DataFrame nettoyé en CSV dans data/processed/<ville>_clean.csv.

    index=False : on n'écrit pas l'index pandas dans le fichier
    (évite une colonne "Unnamed: 0" au rechargement).
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{city}_clean.csv"
    df.to_csv(path, index=False)
    print(f"  Sauvegardé -> {path}")


if __name__ == "__main__":
    for city in ["lyon", "paris", "bordeaux"]:
        df = preprocess(city)
        save_processed(df, city)
