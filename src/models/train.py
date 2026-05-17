import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features.build_features import get_X_y


def split_data(X, y):
    # 80% pour entraîner, 20% pour tester
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_linear(X_train, y_train):
    # Création puis entraînement du modèle
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    # Prédictions sur les données de test
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "y_pred": y_pred,
    }


def run_pipeline(df, features, label="Modèle"):
    # Fonction pratique pour lancer toutes les étapes d'un modèle
    X, y = get_X_y(df, features)

    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_linear(X_train, y_train)
    results = evaluate(model, X_test, y_test)

    print(label)
    print("MAE :", results["MAE"])
    print("RMSE :", results["RMSE"])
    print("R2 :", results["R2"])

    results["model"] = model
    results["X_test"] = X_test
    results["y_test"] = y_test

    return results
