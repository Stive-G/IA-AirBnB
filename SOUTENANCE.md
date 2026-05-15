# Guide de Soutenance — Projet AirBnB IA : Prédiction de Prix
## Rima Ouchefoune — Préparation à l'oral

---

# PARTIE 1 — C'EST QUOI UN JUPYTER NOTEBOOK ?

## Définition simple

Un Jupyter Notebook est un fichier (extension `.ipynb`) qui réunit dans le même
document trois types de contenus :

    1. Du TEXTE      -> titres, explications, commentaires (format Markdown)
    2. Du CODE       -> cellules Python qu'on exécute une par une
    3. Des RÉSULTATS -> graphiques, tableaux, valeurs qui s'affichent sous le code

C'est comme un cahier de laboratoire numérique : on explique ce qu'on fait,
on le fait, on voit le résultat — tout au même endroit, dans l'ordre.

## Pourquoi c'est le livrable standard en Data Science ?

    - On voit le raisonnement ET le code en même temps
    - Les graphiques sont intégrés directement dans le document
    - Sur GitHub, les notebooks se RENDENT automatiquement dans le navigateur
      (pas besoin de télécharger ou d'installer quoi que ce soit pour lire le rendu)
    - Un jury / recruteur peut lire le projet sans exécuter une seule ligne

## À quoi ça ressemble concrètement ?

    +----------------------------------------------------------+
    |  [Cellule MARKDOWN]                                      |
    |  ## 3b. Suppression des doublons                         |
    |  Nous supprimons les lignes en double...                 |
    +----------------------------------------------------------+
    |  [Cellule CODE]                                          |
    |  >>> n_dup = lyon1.duplicated(subset=['id']).sum()       |
    |  >>> print(f"Doublons : {n_dup}")                        |
    +----------------------------------------------------------+
    |  [RÉSULTAT affiché automatiquement]                      |
    |  Doublons : 0                                            |
    +----------------------------------------------------------+

## On a 7 notebooks — pourquoi ?

    6 notebooks de travail (01 a 06) -> pour réviser et comprendre chaque étape
    1 notebook unique (AirBnB_Prediction_Prix.ipynb) -> LE LIVRABLE a remettre

    Sur GitHub, le livrable s'affiche directement dans le navigateur
    avec tous les graphiques rendus — c'est visuellement propre et professionnel.

---

# PARTIE 2 — ARCHITECTURE DÉTAILLÉE DU PROJET

## Vue d'ensemble du problème

    PROBLÈME :
        Un propriétaire veut mettre son logement sur AirBnB.
        Il ne sait pas quel prix afficher.
        Un prix trop bas = il perd de l'argent.
        Un prix trop haut = personne ne réserve.

    SOLUTION :
        On entraîne un modèle de Machine Learning sur les vraies annonces AirBnB
        de Lyon, Paris et Bordeaux (juin 2025).
        Le modèle apprend les patterns de prix selon les caractéristiques
        d'un logement (type, quartier, capacité, scores...).
        Ensuite, pour un NOUVEAU logement, il prédit le prix optimal.

## Schéma du pipeline complet

    Données brutes      Nettoyage       Analyse       Modèle        Prédiction
    insideairbnb.com -> prétraitement -> stats    ->  ML       ->   prix (€)
         |                  |              |            |              |
      Lyon (9 691)     Sélection cols  Histogrammes  Régression   131€/nuit
      Paris (84 055)   Doublons        Corrélations  Linéaire     306€/nuit
      Bordeaux (12 383) Manquantes     Boxplots      Simple &     115€/nuit
                        Types          Top quartiers Multiple

## Structure des dossiers

    AirBnb_Projet/
    |
    +-- README.md                    <- Présentation du projet (affiché sur GitHub)
    +-- SOUTENANCE.md                <- Ce fichier de préparation
    +-- requirements.txt             <- Bibliothèques Python nécessaires
    +-- .gitignore                   <- Fichiers a ne PAS envoyer sur GitHub
    |
    +-- data/
    |   +-- raw/                     <- Données brutes (jamais modifiées)
    |   |   +-- lyon/listings_detail.csv      9 691 annonces x 79 colonnes
    |   |   +-- paris/listings_detail.csv    84 055 annonces x 79 colonnes
    |   |   +-- bordeaux/listings_detail.csv 12 383 annonces x 79 colonnes
    |   +-- processed/               <- Données après nettoyage
    |       +-- lyon_clean.csv       5 382 lignes x 26 colonnes
    |       +-- paris_clean.csv     52 924 lignes x 26 colonnes
    |       +-- bordeaux_clean.csv   8 195 lignes x 26 colonnes
    |       +-- *_features.csv       + colonnes encodées (room_type_code, etc.)
    |
    +-- notebooks/
    |   +-- 01_data_collection.ipynb        RÉVISION : Collecte & aperçu
    |   +-- 02_preprocessing.ipynb          RÉVISION : Nettoyage (4 étapes)
    |   +-- 03_descriptive_analysis.ipynb   RÉVISION : Stats & graphiques
    |   +-- 04_feature_engineering.ipynb    RÉVISION : Recodage & aberrants
    |   +-- 05_modeling.ipynb               RÉVISION : Régression simple & multiple
    |   +-- 06_evaluation.ipynb             RÉVISION : Métriques & prédiction
    |   +-- AirBnB_Prediction_Prix.ipynb    LIVRABLE : Notebook unique complet
    |
    +-- src/                         <- Code Python réutilisable
    |   +-- data/
    |   |   +-- collect.py           Télécharge les .gz depuis insideairbnb.com
    |   |   +-- preprocess.py        Sélection, nettoyage, typage
    |   +-- features/
    |   |   +-- build_features.py    Encodage, outliers, features ML
    |   +-- models/
    |       +-- train.py             Entraîne la régression linéaire
    |       +-- predict.py           Prédit le prix d'un nouveau logement
    |
    +-- reports/
        +-- figures/                 <- Graphiques PNG générés automatiquement

---

# PARTIE 3 — CHAQUE ÉTAPE EXPLIQUÉE EN DÉTAIL

## ÉTAPE 1 — Collecte des données

    SOURCE : https://insideairbnb.com/get-the-data/

    Inside AirBnB est un projet indépendant et open-source qui scrape
    (télécharge automatiquement) les annonces publiques d'AirBnB et les
    publie gratuitement en CSV.

    NOUVEAU PAR RAPPORT A L'ANCIENNE VERSION :
        On utilise les fichiers DÉTAILLÉS (.csv.gz) et non les fichiers
        de visualisation simplifiés.
        -> 79 colonnes au lieu de 18
        -> Beaucoup plus de variables disponibles pour le modèle

    Ce qu'on télécharge (juin 2025) :
        Lyon     -> 9 691 annonces  (15 juin 2025)
        Paris    -> 84 055 annonces (06 juin 2025)
        Bordeaux -> 12 383 annonces (15 juin 2025)

    Exemples de colonnes NOUVELLES disponibles :
        review_scores_rating       -> Note globale (/5)
        review_scores_cleanliness  -> Note propreté
        review_scores_location     -> Note localisation
        host_is_superhost          -> L'hôte est-il superhost ? (t/f)
        host_response_rate         -> Taux de réponse de l'hôte (%)
        accommodates               -> Nombre de personnes max
        bedrooms, beds, bathrooms  -> Taille du logement
        instant_bookable           -> Réservation instantanée ? (t/f)

    -> A dire en soutenance :
      "Nous avons utilisé les fichiers détaillés d'insideairbnb.com avec
      79 colonnes au lieu des 18 de la version simplifiée. Ca nous donne
      accès aux scores de reviews, aux informations hôte et aux
      caractéristiques physiques du logement — des variables bien plus
      corrélées au prix."


## ÉTAPE 2 — Prétraitement des données (4 sous-étapes)

    ---------------------------------------------------------------
    2a. SÉLECTION DES COLONNES
    ---------------------------------------------------------------

    Problème : 79 colonnes dont beaucoup inutiles (URL, texte libre,
               photo de profil de l'hôte, ID de scraping...).

    Solution : On garde 26 colonnes pertinentes pour la prédiction.

    Colonnes GARDÉES           Pourquoi ?
    -------------------------  -----------------------------------------
    id                         Identifiant (pour détecter les doublons)
    neighbourhood_cleansed     Quartier nettoyé (plus fiable que neighbourhood)
    room_type                  Type de logement = très important
    property_type              Appartement, maison, studio...
    accommodates               Nombre de personnes = corrélation forte
    bedrooms, beds, bathrooms  Taille physique du logement
    price                      VARIABLE CIBLE a prédire
    minimum_nights             Contrainte de durée
    availability_365           Disponibilité annuelle
    number_of_reviews          Proxy de popularité
    reviews_per_month          Fréquence d'utilisation
    review_scores_*            Qualité perçue (nouveau !)
    host_is_superhost          Qualité de l'hôte (nouveau !)
    host_response_rate         Réactivité de l'hôte (nouveau !)
    calculated_host_..._count  Hôte professionnel ou particulier ?
    instant_bookable           Facilité de réservation (nouveau !)
    latitude, longitude        Coordonnées géographiques

    Colonnes SUPPRIMÉES        Pourquoi ?
    -------------------------  -----------------------------------------
    listing_url, picture_url   URLs, inutiles pour le ML
    host_about, description    Texte libre, trop complexe a exploiter
    scrape_id, source          Méta-données de collecte
    name, host_name            Texte, pas pertinent pour le prix

    ---------------------------------------------------------------
    2b. SUPPRESSION DES DOUBLONS
    ---------------------------------------------------------------

    n_dup = df.duplicated(subset=['id']).sum()
    -> 0 doublon détecté dans nos 3 villes

    ---------------------------------------------------------------
    2c. GESTION DES VALEURS MANQUANTES
    ---------------------------------------------------------------

    Valeurs manquantes détectées :
        reviews_per_month    -> vide si jamais d'avis -> remplacé par 0
        review_scores_*      -> vide si pas encore d'avis -> remplacé par 0
        bedrooms / beds      -> vide pour certains studios -> remplacé par 1
        price                -> quelques annonces sans prix -> supprimées

    ---------------------------------------------------------------
    2d. CONVERSION DES TYPES DE COLONNES
    ---------------------------------------------------------------

    price             : "$150.00" -> 150.0
    bathrooms_text    : "1 bath"  -> 1.0
    host_is_superhost : "t"/"f"   -> 1/0
    instant_bookable  : "t"/"f"   -> 1/0
    host_response_rate: "95%"     -> 95.0

    Résultat final :
        Lyon     : 9 691  -> 5 382 lignes x 26 colonnes
        Paris    : 84 055 -> 52 924 lignes x 26 colonnes
        Bordeaux : 12 383 -> 8 195 lignes x 26 colonnes


## ÉTAPE 3 — Analyse descriptive des données

    Résultats clés :

        Prix médian   Lyon : 85€    Paris : 110€    Bordeaux : 80€
        Prix moyen    Lyon : 99€    Paris : 148€    Bordeaux : 95€
        Nb annonces   Lyon : 5 382  Paris : 52 924  Bordeaux : 8 195

        Superhosts    Lyon : ~25%   Paris : ~20%    Bordeaux : ~30%
        Note moyenne  Lyon : ~4.6   Paris : ~4.5    Bordeaux : ~4.7

        Type le + cher  : Entire home/apt (appartement entier)
        Type le - cher  : Shared room (chambre partagée)

        Variable la + correlée au prix : accommodates (0.45 environ)
        Puis : bedrooms, room_type_code, beds, review_scores_location

    -> A dire en soutenance :
      "Cette étape exploratoire confirme nos intuitions : la taille du
      logement (nombre de personnes) est le facteur le plus corrélé au
      prix. C'est pour ça qu'on l'utilise comme variable unique dans
      la régression simple."


## ÉTAPE 4 — Recodage des colonnes (Feature Engineering)

    3 transformations principales :

    4a. ENCODAGE ORDINAL DU TYPE DE LOGEMENT
        "Shared room"      ->  0
        "Private room"     ->  1
        "Hotel room"       ->  2
        "Entire home/apt"  ->  3
        Logique : ordre croissant de prix moyen constaté

    4b. ENCODAGE PAR FRÉQUENCE DU QUARTIER
        On remplace le nom du quartier par sa proportion dans les données.
        "Croix-Rousse" -> 0.08 (8% des annonces)
        "Vieux-Lyon"   -> 0.03 (3% des annonces)

    4c. ENCODAGE PAR FRÉQUENCE DU TYPE DE PROPRIÉTÉ
        "Entire rental unit" -> 0.42 (le type le plus fréquent)
        "Private room in..."  -> 0.28
        etc.


## ÉTAPE 5 — Gestion des valeurs aberrantes

    Seuils appliqués (percentiles 1% - 99%) :
        Lyon     : [25€  - 492€]  -> 104 lignes supprimées
        Paris    : [41€  - 1936€] -> 1 039 lignes supprimées
        Bordeaux : [24€  - 529€]  -> 150 lignes supprimées

    -> A dire en soutenance :
      "La régression linéaire est très sensible aux valeurs extrêmes.
      Sans cette étape, une annonce a 9 999€/nuit fausserait
      complètement la droite. On utilise les percentiles 99% plutôt
      que 95% pour ne retirer que les vrais cas extrêmes."


## ÉTAPE 6 — Séparation train / test + Modèles

    Séparation :
        80% -> Entraînement (le modèle apprend)
        20% -> Test (on évalue sur des données jamais vues)
        random_state=42 -> résultats reproductibles

    Lyon : 4 305 train / 1 077 test
    Paris : 42 339 train / 10 585 test
    Bordeaux : 6 556 train / 1 639 test

    RÉGRESSION SIMPLE (1 variable) :
        prix = a * accommodates + b

        Pourquoi accommodates ?
        -> C'est la variable la plus corrélée au prix (r ~ 0.45)
        -> Intuition forte : plus il y a de personnes, plus c'est cher

    RÉGRESSION MULTIPLE (18 variables) :
        prix = a1*accommodates + a2*bedrooms + a3*beds + a4*bathrooms
             + a5*minimum_nights + a6*availability_365
             + a7*number_of_reviews + a8*reviews_per_month
             + a9*review_scores_rating + a10*review_scores_cleanliness
             + a11*review_scores_location + a12*host_is_superhost
             + a13*host_response_rate + a14*calculated_host_listings_count
             + a15*instant_bookable + a16*room_type_code
             + a17*neighbourhood_freq + a18*property_type_freq
             + b


## ÉTAPE 7 — Évaluation des modèles

    Résultats obtenus (vraies valeurs calculées) :

    Ville    | Modèle    | MAE (€)  | RMSE (€) |   R²
    ---------+-----------+----------+----------+--------
    Lyon     | Simple    |  36.46   |  54.54   |  0.27
    Lyon     | Multiple  |  32.94   |  50.21   |  0.38
    Paris    | Simple    | 117.52   | 192.41   |  0.23
    Paris    | Multiple  | 105.31   | 179.34   |  0.33
    Bordeaux | Simple    |  38.08   |  56.71   |  0.51
    Bordeaux | Multiple  |  32.54   |  49.88   |  0.62

    Analyse :
        - Multiple toujours meilleur que simple
        - Bordeaux : meilleur R² (0.62) -> marché plus homogène
        - Paris : erreur plus grande car prix très dispersés
        - R² en hausse vs ancienne version grâce aux 79 colonnes

    -> A dire en soutenance :
      "Le R² de 0.62 sur Bordeaux est un bon résultat pour une
      régression linéaire sur ce type de données. Paris est plus
      difficile car le marché est extrêmement hétérogène : un studio
      dans le 20e n'a pas le même prix qu'un appartement dans le 1er.
      Avec un Random Forest, on capturerait ces relations non-linéaires
      et on améliorerait significativement le R²."

---

# PARTIE 4 — DÉMONSTRATION COMMENTÉE

## Lancer le projet

    cd C:\Users\Asus\Downloads\AirBnb_Projet
    pip install -r requirements.txt
    jupyter notebook notebooks/

    # Révision  : ouvrir 01 -> 02 -> 03 -> 04 -> 05 -> 06
    # Livrable  : ouvrir AirBnB_Prediction_Prix.ipynb

## Pipeline Python de A à Z avec vrais outputs

    +------------------------------------------------------------------+
    | ÉTAPE 1 : Téléchargement                                         |
    +------------------------------------------------------------------+

    from src.data.collect import download_all
    download_all(raw_dir='data/raw')

    OUTPUT :
        [DL]   lyon...     -> 18,927,394 octets
        [DL]   paris...    -> 173,207,599 octets
        [DL]   bordeaux... -> 25,055,604 octets
        Telechargement termine.

    +------------------------------------------------------------------+
    | ÉTAPE 2 : Prétraitement                                          |
    +------------------------------------------------------------------+

    from src.data.preprocess import preprocess, save_processed

    df_lyon = preprocess('lyon')
    save_processed(df_lyon, 'lyon')

    OUTPUT :
        === Prétraitement : LYON ===
          Charge : 9,691 lignes x 79 colonnes
          Après nettoyage : 5,382 lignes x 26 colonnes
          Sauvegarde -> data/processed/lyon_clean.csv

    +------------------------------------------------------------------+
    | ÉTAPE 3 : Feature Engineering                                    |
    +------------------------------------------------------------------+

    from src.features.build_features import build_features
    df_fe = build_features(df_lyon)

    OUTPUT :
        Aberrants prix supprimes : 104 lignes (seuil: 25€ - 492€)

    +------------------------------------------------------------------+
    | ÉTAPE 4 : Modélisation                                           |
    +------------------------------------------------------------------+

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    FEATURES = ['accommodates','bedrooms','beds','bathrooms',
                'minimum_nights','availability_365','number_of_reviews',
                'reviews_per_month','review_scores_rating',
                'review_scores_cleanliness','review_scores_location',
                'host_is_superhost','host_response_rate',
                'calculated_host_listings_count','instant_bookable',
                'room_type_code','neighbourhood_freq','property_type_freq']

    X, y = df_fe[FEATURES], df_fe['price']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)

    OUTPUT :
        Entrainement sur 4,305 annonces.
        Test sur 1,077 annonces.

    +------------------------------------------------------------------+
    | ÉTAPE 5 : Evaluation                                             |
    +------------------------------------------------------------------+

    OUTPUT (Lyon - Multiple) :
        MAE  = 32.94€
        RMSE = 50.21€
        R²   = 0.3809

    OUTPUT (Bordeaux - Multiple) :
        MAE  = 32.54€
        RMSE = 49.88€
        R²   = 0.6227

    +------------------------------------------------------------------+
    | ÉTAPE 6 : Prédiction d'un NOUVEAU logement                       |
    +------------------------------------------------------------------+

    from src.models.predict import predict_price

    prix = predict_price(
        model          = model,
        feature_names  = FEATURES,
        accommodates   = 4,     # 4 personnes
        bedrooms       = 2,     # 2 chambres
        review_scores_rating = 4.8,
        host_is_superhost    = 1,
        room_type            = "Entire home/apt",
    )

    OUTPUT :
        Prix estime a Lyon     : 130.64 euros/nuit
        Prix estime a Paris    : 305.73 euros/nuit
        Prix estime a Bordeaux : 115.36 euros/nuit

## Ce que montrent les graphiques

    prix_distribution.png
        -> Histogramme des prix sur les 3 villes.
           Lyon et Bordeaux concentres entre 60-150€.
           Paris beaucoup plus étalé (80-400€), longue queue a droite.

    top_quartiers.png
        -> Top 10 quartiers les plus chers par ville.
           Lyon : Presqu'île, Vieux-Lyon, Confluence.
           Paris : 1er, 6e, 7e arrondissements.
           Bordeaux : Centre-ville, Chartrons, Saint-Pierre.

    regression_simple.png
        -> Nuage de points prix vs accommodates + droite de régression.
           On voit clairement que plus il y a de personnes, plus c'est cher.

    regression_multiple.png
        -> Graphique Réel vs Prédit.
           Ligne diagonale = prédiction parfaite.
           Plus les points sont proches de la diagonale -> meilleur modèle.

    coefficients.png
        -> Barres des coefficients de la régression multiple.
           Bleu = variable qui augmente le prix.
           Rouge = variable qui baisse le prix.
           La barre la plus longue = variable la plus influente.

    residus.png
        -> Nuage de points des résidus (erreur = réel - prédit).
           Un bon modèle a des résidus centrés autour de 0
           sans structure particulière.

---

# PARTIE 5 — QUESTIONS FRÉQUENTES EN SOUTENANCE

    Q : Pourquoi 3 villes au lieu de 2 ?
    --------------------------------------
    R : Le prof a changé la consigne. Bordeaux apporte une comparaison
        intéressante : c'est une ville de taille intermédiaire entre
        Lyon (grande ville régionale) et Paris (capitale). On voit que
        le marché y est plus homogène -> notre modèle y est le meilleur
        avec un R² de 0.62.

    Q : Pourquoi les fichiers détaillés (.gz) et pas les CSV simples ?
    -------------------------------------------------------------------
    R : Les fichiers détaillés ont 79 colonnes vs 18 pour les fichiers
        de visualisation. Les colonnes supplémentaires (scores de reviews,
        capacité du logement, infos hôte) sont bien plus corrélées au prix
        que les colonnes basiques. Notre R² est passé de 0.10 a 0.38 sur Lyon
        et 0.04 a 0.62 sur Bordeaux grâce a ces données enrichies.

    Q : Pourquoi le R² est faible pour Paris ?
    -------------------------------------------
    R : Paris est un marché extrêmement hétérogène : un studio dans le 20e
        coûte 60€/nuit, un appartement de luxe dans le 1er peut coûter
        2 000€/nuit. La régression linéaire ne capture pas bien ces
        différences non-linéaires. Un Random Forest ou XGBoost segmenterait
        automatiquement par zone géographique et type de bien.

    Q : Pourquoi la régression linéaire et pas Random Forest directement ?
    -----------------------------------------------------------------------
    R : En Data Science, on commence TOUJOURS par le modèle le plus simple
        pour établir une baseline. Ca permet de :
        1) Vérifier que les données sont correctement préparées
        2) Avoir un point de comparaison pour les modèles plus complexes
        3) Interpréter facilement les coefficients (chaque variable a un
           impact direct et lisible sur le prix)

    Q : C'est quoi la différence entre MAE et RMSE ?
    --------------------------------------------------
    R : MAE = erreur moyenne en euros. Simple et intuitive.
        RMSE = penalise DAVANTAGE les grosses erreurs (mise au carré).
        Si RMSE >> MAE : il y a des cas où on se plante vraiment beaucoup.
        On préfère surveiller les deux ensemble.

    Q : Pourquoi le percentile 99% et pas 95% pour les outliers ?
    -------------------------------------------------------------
    R : 95% serait trop agressif (on perdrait des annonces légitimes
        un peu chères). 99% est un compromis standard : on ne retire que
        les 1% les plus extrêmes de chaque côté.

    Q : Le modèle marche pour d'autres villes ?
    --------------------------------------------
    R : Non directement. Chaque modèle est entraîné sur UNE ville.
        Les patterns de prix a Lyon ne correspondent pas a Bordeaux.
        Il faudrait télécharger les données de la ville cible et entraîner
        un nouveau modèle spécifique.

    Q : C'est quoi __init__.py et __pycache__ dans le dossier src/ ?
    ----------------------------------------------------------------
    R : __init__.py est un fichier vide qui dit a Python "ce dossier
        est un package importable". Sans lui, on ne peut pas faire
        "from src.data.preprocess import preprocess".
        __pycache__ est créé automatiquement par Python : il contient
        les fichiers compilés (.pyc) pour accélérer les prochaines
        exécutions. On le met dans .gitignore car il est inutile sur GitHub.

---

# RÉSUMÉ EN 5 POINTS POUR LA CONCLUSION

    1. PROBLÈME RÉEL
       Aider un propriétaire a fixer le bon prix sur AirBnB pour
       maximiser ses revenus sans décourager les réservations.

    2. DONNÉES RÉELLES ET RICHES
       9 691 + 84 055 + 12 383 annonces téléchargées directement
       depuis insideairbnb.com en juin 2025.
       Fichiers détaillés : 79 colonnes (vs 18 habituellement).

    3. PIPELINE COMPLET ET REPRODUCTIBLE
       Collecte -> Nettoyage (4 étapes) -> Analyse -> Recodage
       -> Aberrants -> Modélisation -> Evaluation -> Prédiction.
       Chaque étape est justifiée, codée proprement, et réutilisable
       via le package src/.

    4. RÉSULTATS HONNÊTES ET EXPLIQUÉS
       R² de 0.38 (Lyon), 0.33 (Paris), 0.62 (Bordeaux).
       La régression linéaire est notre baseline.
       Les résultats faibles sur Paris s'expliquent par l'hétérogénéité
       du marché, pas par une erreur de traitement.

    5. PERSPECTIVES CLAIRES
       Random Forest ou XGBoost pour capturer les non-linéarités.
       Ajout de données géographiques (distance au centre, transports).
       Analyse de la saisonnalité (données calendrier d'insideairbnb).
