from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent

CITIES = {
    "Lyon": "lyon",
    "Paris": "paris",
    "Bordeaux": "bordeaux",
}

FEATURES = [
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


def lire_int(question, valeur_defaut):
    reponse = input(question + " [" + str(valeur_defaut) + "] : ")
    if reponse == "":
        return valeur_defaut
    return int(reponse)


def lire_float(question, valeur_defaut):
    reponse = input(question + " [" + str(valeur_defaut) + "] : ")
    if reponse == "":
        return valeur_defaut
    return float(reponse)


def lire_oui_non(question):
    reponse = input(question + " (o/n) : ").lower()
    if reponse == "o" or reponse == "oui":
        return 1
    return 0


def charger_modeles():
    modeles = {}
    frequences_quartiers = {}
    dossier = BASE_DIR / "data" / "processed"

    for ville, nom_fichier in CITIES.items():
        chemin = dossier / (nom_fichier + "_features.csv")

        if not chemin.exists():
            print("Fichier manquant :", chemin)
            print("Il faut d'abord exécuter le notebook.")
            return None, None

        df = pd.read_csv(chemin)
        colonnes = []

        for col in FEATURES:
            if col in df.columns:
                colonnes.append(col)

        data = df[colonnes + ["price"]].dropna()
        X = data[colonnes]
        y = data["price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        # On entraîne un modèle pour chaque ville.
        modele = LinearRegression()
        modele.fit(X_train, y_train)

        modeles[ville] = (modele, colonnes)
        frequences_quartiers[ville] = df["neighbourhood_freq"].median()

    return modeles, frequences_quartiers


def demander_infos():
    print("\nInformations du logement")

    print("Type de logement")
    print("1 - Logement entier")
    print("2 - Chambre privée")
    print("3 - Chambre d'hôtel")
    print("4 - Chambre partagée")

    choix = input("Votre choix [1] : ")
    if choix == "":
        choix = "1"

    if choix == "1":
        room_type_code = 3
    elif choix == "2":
        room_type_code = 1
    elif choix == "3":
        room_type_code = 2
    else:
        room_type_code = 0

    valeurs = {}
    valeurs["accommodates"] = lire_int("Nombre de personnes", 2)
    valeurs["bedrooms"] = lire_int("Nombre de chambres", 1)
    valeurs["beds"] = lire_int("Nombre de lits", 1)
    valeurs["bathrooms"] = lire_float("Nombre de salles de bain", 1.0)
    valeurs["minimum_nights"] = lire_int("Nuits minimum", 2)
    valeurs["availability_365"] = lire_int("Jours disponibles par an", 180)
    valeurs["number_of_reviews"] = lire_int("Nombre d'avis", 0)
    valeurs["reviews_per_month"] = lire_float("Avis par mois", 0.0)
    valeurs["review_scores_rating"] = lire_float("Note globale /5", 0.0)
    valeurs["review_scores_cleanliness"] = lire_float("Note propreté /5", 0.0)
    valeurs["review_scores_location"] = lire_float("Note localisation /5", 0.0)
    valeurs["host_is_superhost"] = lire_oui_non("Superhost")
    valeurs["host_response_rate"] = lire_float("Taux de réponse", 90.0)
    valeurs["calculated_host_listings_count"] = lire_int("Nombre d'annonces de l'hôte", 1)
    valeurs["instant_bookable"] = lire_oui_non("Réservation instantanée")
    valeurs["room_type_code"] = room_type_code
    valeurs["property_type_freq"] = 0.35

    return valeurs


def predire_prix(modeles, frequences_quartiers, valeurs):
    print("\nPrix estimés")

    for ville, infos_modele in modeles.items():
        modele = infos_modele[0]
        colonnes = infos_modele[1]

        valeurs["neighbourhood_freq"] = frequences_quartiers[ville]

        ligne = []
        for col in colonnes:
            ligne.append(valeurs.get(col, 0))

        X = pd.DataFrame([ligne], columns=colonnes)
        prix = modele.predict(X)[0]

        print(ville, ":", round(float(prix), 2), "euros / nuit")


def main():
    print("IA Airbnb - prédiction de prix")

    modeles, frequences_quartiers = charger_modeles()
    if modeles is None:
        return

    continuer = "o"
    while continuer == "o":
        valeurs = demander_infos()
        predire_prix(modeles, frequences_quartiers, valeurs)
        continuer = input("\nFaire une autre prédiction ? (o/n) : ").lower()


if __name__ == "__main__":
    main()
