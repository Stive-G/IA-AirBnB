import re
from pathlib import Path

import pandas as pd


COLUMNS = [
    "id",
    "neighbourhood_cleansed",
    "room_type",
    "property_type",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms_text",
    "price",
    "minimum_nights",
    "maximum_nights",
    "availability_365",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "reviews_per_month",
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_location",
    "review_scores_value",
    "host_is_superhost",
    "host_response_rate",
    "host_acceptance_rate",
    "calculated_host_listings_count",
    "instant_bookable",
    "latitude",
    "longitude",
]


def load_raw(city, raw_dir="data/raw"):
    path = Path(raw_dir) / city / "listings_detail.csv"
    return pd.read_csv(path, low_memory=False)


def select_columns(df):
    # On garde seulement les colonnes utiles pour le projet
    existing_columns = []
    for column in COLUMNS:
        if column in df.columns:
            existing_columns.append(column)

    return df[existing_columns].copy()


def remove_duplicates(df):
    return df.drop_duplicates(subset=["id"])


def clean_price(df):
    # $120.00 devient 120.0.
    df["price"] = df["price"].astype(str)
    df["price"] = df["price"].str.replace("$", "", regex=False)
    df["price"] = df["price"].str.replace(",", "", regex=False)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df


def clean_percent(column):
    column = column.astype(str)
    column = column.str.replace("%", "", regex=False)
    return pd.to_numeric(column, errors="coerce").fillna(0)


def get_bathroom_number(value):
    if pd.isna(value):
        return 1.0

    numbers = re.findall(r"[\d.]+", str(value))
    if len(numbers) == 0:
        return 1.0

    return float(numbers[0])


def handle_missing(df):
    # Valeurs simples pour ne pas bloquer le modèle avec des NaN
    values = {
        "reviews_per_month": 0,
        "number_of_reviews": 0,
        "number_of_reviews_ltm": 0,
        "review_scores_rating": 0,
        "review_scores_cleanliness": 0,
        "review_scores_location": 0,
        "review_scores_value": 0,
        "bedrooms": 1,
        "beds": 1,
    }

    for column, value in values.items():
        if column in df.columns:
            df[column] = df[column].fillna(value)

    df = df.dropna(subset=["price"])
    return df


def convert_types(df):
    df = clean_price(df)

    if "bathrooms_text" in df.columns:
        # On transforme "1 bath" ou "1.5 baths" en nombre
        df["bathrooms"] = df["bathrooms_text"].apply(get_bathroom_number)
        df = df.drop(columns=["bathrooms_text"])

    for column in ["host_response_rate", "host_acceptance_rate"]:
        if column in df.columns:
            df[column] = clean_percent(df[column])

    for column in ["host_is_superhost", "instant_bookable"]:
        if column in df.columns:
            # Dans les données Airbnb, t = vrai et f = faux.
            df[column] = df[column].map({"t": 1, "f": 0}).fillna(0).astype(int)

    int_columns = [
        "minimum_nights",
        "maximum_nights",
        "availability_365",
        "number_of_reviews",
        "number_of_reviews_ltm",
        "calculated_host_listings_count",
        "accommodates",
    ]

    for column in int_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)

    return df


def preprocess(city, raw_dir="data/raw"):
    print("Nettoyage :", city)

    # Ordre du nettoyage utilisé dans le notebook
    df = load_raw(city, raw_dir)
    df = select_columns(df)
    df = remove_duplicates(df)
    df = handle_missing(df)
    df = convert_types(df)

    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    return df.reset_index(drop=True)


def save_processed(df, city, out_dir="data/processed"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{city}_clean.csv"
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        print("Impossible d'écrire le fichier :", path)
        print("Ferme le CSV s'il est ouvert dans Excel ou VS Code, puis relance.")
        raise
    print("Fichier créé :", path)


if __name__ == "__main__":
    for city in ["paris", "lyon", "bordeaux"]:
        data = preprocess(city)
        save_processed(data, city)
