import pandas as pd


ROOM_TYPE_MAP = {
    "Shared room": 0,
    "Private room": 1,
    "Hotel room": 2,
    "Entire home/apt": 3,
}


def predict_price(model, feature_names, values):
    values["room_type_code"] = ROOM_TYPE_MAP.get(values.get("room_type"), 1)

    # On met les valeurs dans le même ordre que les colonnes du modèle.
    row = []
    for feature in feature_names:
        if feature in values:
            row.append(values[feature])
        else:
            row.append(0)

    X = pd.DataFrame([row], columns=feature_names)
    price = model.predict(X)[0]

    return round(float(price), 2)
