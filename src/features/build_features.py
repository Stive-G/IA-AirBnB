ROOM_TYPE_MAP = {
    "Shared room": 0,
    "Private room": 1,
    "Hotel room": 2,
    "Entire home/apt": 3,
}


FEATURE_COLS_SIMPLE = ["accommodates"]


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


TARGET = "price"


def add_room_type_code(df):
    # Le modèle travaille avec des nombres, pas avec du texte
    df["room_type_code"] = df["room_type"].map(ROOM_TYPE_MAP).fillna(0).astype(int)
    return df


def add_neighbourhood_freq(df):
    # Fréquence du quartier dans la ville
    counts = df["neighbourhood_cleansed"].value_counts(normalize=True)
    df["neighbourhood_freq"] = df["neighbourhood_cleansed"].map(counts)
    return df


def add_property_type_freq(df):
    # Même idée pour le type de logement
    counts = df["property_type"].value_counts(normalize=True)
    df["property_type_freq"] = df["property_type"].map(counts)
    return df


def remove_price_outliers(df):
    # On retire les prix extrêmes pour ne pas trop influencer la régression
    low = df["price"].quantile(0.01)
    high = df["price"].quantile(0.99)
    df = df[(df["price"] >= low) & (df["price"] <= high)]
    return df.reset_index(drop=True)


def build_features(df):
    df = df.copy()
    df = add_room_type_code(df)
    df = add_neighbourhood_freq(df)
    df = add_property_type_freq(df)
    df = remove_price_outliers(df)
    return df


def get_X_y(df, features):
    # X contient les variables explicatives, y contient le prix
    columns = []

    for feature in features:
        if feature in df.columns:
            columns.append(feature)

    df_model = df[columns + [TARGET]].dropna()
    X = df_model[columns]
    y = df_model[TARGET]

    return X, y
