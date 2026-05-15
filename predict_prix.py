"""
IA de prédiction de prix AirBnB
--------------------------------
Ce script pose des questions sur un logement et prédit
le prix optimal par nuit pour Lyon, Paris et Bordeaux.

Utilisation :
    py -3.13 predict_prix.py
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ── Constantes ────────────────────────────────────────────────────────────────

CITIES = {"Lyon": "lyon", "Paris": "paris", "Bordeaux": "bordeaux"}

FEATURES = [
    "accommodates", "bedrooms", "beds", "bathrooms", "minimum_nights",
    "availability_365", "number_of_reviews", "reviews_per_month",
    "review_scores_rating", "review_scores_cleanliness", "review_scores_location",
    "host_is_superhost", "host_response_rate", "calculated_host_listings_count",
    "instant_bookable", "room_type_code", "neighbourhood_freq", "property_type_freq",
]

ROOM_TYPE_MAP = {
    "1": ("Logement entier",   3),
    "2": ("Chambre privée",    1),
    "3": ("Chambre d'hôtel",   2),
    "4": ("Chambre partagée",  0),
}

TARGET = "price"

# ── Chargement et entraînement ─────────────────────────────────────────────────

def charger_et_entrainer():
    """Charge les données nettoyées et entraîne un modèle par ville."""
    modeles = {}
    freq_quartiers = {}

    data_dir = os.path.join(os.path.dirname(__file__), "data", "processed")

    for ville, cle in CITIES.items():
        path = os.path.join(data_dir, f"{cle}_features.csv")
        if not os.path.exists(path):
            print(f"  [!] Fichier manquant : {path}")
            print(f"      Exécute d'abord le notebook AirBnB_Prediction_Prix.ipynb")
            sys.exit(1)

        df = pd.read_csv(path)
        feats = [f for f in FEATURES if f in df.columns]
        d = df[feats + [TARGET]].dropna()
        X, y = d[feats], d[TARGET]

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression().fit(X_train, y_train)

        modeles[ville] = (model, feats)

        # Fréquence médiane du quartier pour cette ville (valeur par défaut)
        freq_quartiers[ville] = float(df["neighbourhood_freq"].median())

    return modeles, freq_quartiers


# ── Interface utilisateur ──────────────────────────────────────────────────────

def demander_int(question, defaut, mini=0, maxi=9999):
    """Pose une question et retourne un entier validé."""
    while True:
        rep = input(f"  {question} [défaut: {defaut}] : ").strip()
        if rep == "":
            return defaut
        try:
            val = int(rep)
            if mini <= val <= maxi:
                return val
            print(f"    → Valeur entre {mini} et {maxi} attendue.")
        except ValueError:
            print("    → Entier attendu.")


def demander_float(question, defaut, mini=0.0, maxi=9999.0):
    """Pose une question et retourne un float validé."""
    while True:
        rep = input(f"  {question} [défaut: {defaut}] : ").strip()
        if rep == "":
            return defaut
        try:
            val = float(rep)
            if mini <= val <= maxi:
                return val
            print(f"    → Valeur entre {mini} et {maxi} attendue.")
        except ValueError:
            print("    → Nombre décimal attendu.")


def demander_oui_non(question, defaut="o"):
    """Pose une question oui/non et retourne 1 ou 0."""
    rep = input(f"  {question} (o/n) [défaut: {defaut}] : ").strip().lower()
    if rep == "":
        rep = defaut
    return 1 if rep in ("o", "oui", "y", "yes") else 0


def collecter_caracteristiques(freq_quartiers):
    """Pose toutes les questions et retourne un dictionnaire de features."""
    print("\n" + "═" * 55)
    print("  CARACTÉRISTIQUES DE VOTRE LOGEMENT")
    print("═" * 55)

    # Type de logement
    print("\n  Type de logement :")
    for k, (label, _) in ROOM_TYPE_MAP.items():
        print(f"    {k}. {label}")
    while True:
        choix = input("  Votre choix [défaut: 1] : ").strip() or "1"
        if choix in ROOM_TYPE_MAP:
            room_label, room_code = ROOM_TYPE_MAP[choix]
            print(f"    → {room_label} sélectionné")
            break
        print("    → Choix invalide (1, 2, 3 ou 4).")

    print()
    accommodates = demander_int("Nombre de personnes max accueillies",  2,  1, 20)
    bedrooms     = demander_int("Nombre de chambres",                   1,  0, 20)
    beds         = demander_int("Nombre de lits",                       1,  0, 30)
    bathrooms    = demander_float("Nombre de salles de bain",           1.0, 0, 10)
    minimum_nights = demander_int("Nuits minimum imposées",             2,  1, 365)
    availability_365 = demander_int("Jours disponibles par an",        180,  0, 365)

    print("\n" + "─" * 55)
    print("  INFORMATIONS SUR L'HÔTE")
    print("─" * 55)
    superhost     = demander_oui_non("Êtes-vous superhost ?",          "n")
    response_rate = demander_float("Taux de réponse (%) ",             90.0, 0, 100)
    nb_listings   = demander_int("Nombre total d'annonces que vous gérez", 1, 1, 500)
    instant       = demander_oui_non("Réservation instantanée ?",     "n")

    print("\n" + "─" * 55)
    print("  AVIS ET NOTES")
    print("─" * 55)
    nb_reviews     = demander_int("Nombre total d'avis reçus",         0,   0, 9999)
    reviews_month  = demander_float("Avis par mois en moyenne",        0.0, 0, 99)
    note_globale   = demander_float("Note globale /5 (0 = pas encore noté)", 0.0, 0, 5)
    note_proprete  = demander_float("Note propreté /5",                0.0, 0, 5)
    note_location  = demander_float("Note localisation /5",            0.0, 0, 5)

    # Fréquence de quartier : on utilise la médiane de chaque ville par défaut
    # (représente un quartier "moyen" dans les données)
    property_type_freq = 0.35  # valeur typique pour les appartements entiers

    return {
        "accommodates"                  : accommodates,
        "bedrooms"                      : bedrooms,
        "beds"                          : beds,
        "bathrooms"                     : bathrooms,
        "minimum_nights"                : minimum_nights,
        "availability_365"              : availability_365,
        "number_of_reviews"             : nb_reviews,
        "reviews_per_month"             : reviews_month,
        "review_scores_rating"          : note_globale,
        "review_scores_cleanliness"     : note_proprete,
        "review_scores_location"        : note_location,
        "host_is_superhost"             : superhost,
        "host_response_rate"            : response_rate,
        "calculated_host_listings_count": nb_listings,
        "instant_bookable"              : instant,
        "room_type_code"                : room_code,
        "property_type_freq"            : property_type_freq,
        # neighbourhood_freq sera remplacé par la médiane de chaque ville
    }, freq_quartiers


# ── Prédiction ─────────────────────────────────────────────────────────────────

def predire(modeles, caracteristiques, freq_quartiers):
    """Calcule le prix estimé pour chaque ville et affiche les résultats."""
    print("\n" + "═" * 55)
    print("  PRIX ESTIMÉ PAR L'IA")
    print("═" * 55)

    caract, _ = caracteristiques

    for ville, (model, feats) in modeles.items():
        # On utilise la fréquence médiane du quartier pour cette ville
        caract["neighbourhood_freq"] = freq_quartiers[ville]

        X = pd.DataFrame([[caract.get(f, 0) for f in feats]], columns=feats)
        prix = round(float(model.predict(X)[0]), 2)

        # Fourchette ±15% pour donner une marge réaliste
        bas  = round(prix * 0.85, 2)
        haut = round(prix * 1.15, 2)

        print(f"\n  {ville}")
        print(f"    Prix estimé  : {prix} €/nuit")
        print(f"    Fourchette   : {bas} € – {haut} €/nuit")

    print("\n" + "═" * 55)
    print("  Note : estimation basée sur les données AirBnB juin 2025.")
    print("  Le prix réel peut varier selon la saison, les photos,")
    print("  la qualité de la description et les événements locaux.")
    print("═" * 55 + "\n")


# ── Point d'entrée ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 55)
    print("  IA AIRBNB — PRÉDICTION DE PRIX")
    print("  Cours Python · IA Supervisée · M. Sayf BEJAOUI")
    print("═" * 55)
    print("\n  Chargement des modèles...")

    modeles, freq_quartiers = charger_et_entrainer()
    print("  Modèles prêts (Lyon, Paris, Bordeaux).\n")

    while True:
        caract = collecter_caracteristiques(freq_quartiers)
        predire(modeles, caract, freq_quartiers)

        continuer = input("  Estimer un autre logement ? (o/n) : ").strip().lower()
        if continuer not in ("o", "oui", "y", "yes"):
            print("\n  Au revoir !\n")
            break


if __name__ == "__main__":
    main()
