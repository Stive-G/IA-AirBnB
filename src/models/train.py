"""
Entraînement des modèles : régression linéaire simple et multiple.

Ce module contient les fonctions bas niveau (split, train, evaluate)
et la fonction haut niveau `run_pipeline` qui les enchaîne.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Divise les données en ensemble d'entraînement (80%) et de test (20%).

    Paramètres :
    - test_size=0.2   → 20% des données réservées pour l'évaluation finale
    - random_state=42 → graine aléatoire fixe pour obtenir des résultats reproductibles
                        (si on relance le code, on obtient exactement le même split)

    Retourne : X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_linear(X_train, y_train) -> LinearRegression:
    """
    Entraîne un modèle de régression linéaire sur les données d'entraînement.

    LinearRegression de scikit-learn minimise la somme des erreurs au carré (moindres carrés).
    model.fit(X_train, y_train) calcule les coefficients optimaux et l'intercept.

    Retourne le modèle entraîné, prêt pour predict().
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate(model: LinearRegression, X_test, y_test) -> dict:
    """
    Calcule les métriques d'évaluation sur l'ensemble de test.

    Métriques retournées :
    - MAE  (Mean Absolute Error)       : erreur moyenne en euros — facile à interpréter
    - RMSE (Root Mean Squared Error)   : comme MAE mais pénalise davantage les grosses erreurs
    - R²   (coefficient de détermination) : proportion de variance expliquée
                                           0 = le modèle ne prédit rien
                                           1 = prédiction parfaite
                                           0.38 = le modèle explique 38% de la variabilité des prix

    On évalue sur X_test (jamais vu pendant l'entraînement) pour éviter le surapprentissage.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4), "y_pred": y_pred}


def run_pipeline(df: pd.DataFrame, features: list[str], label: str = "") -> dict:
    """
    Pipeline complet : extraction → split → entraînement → évaluation.

    C'est la fonction principale à appeler depuis un notebook ou un script.
    Elle enchaîne get_X_y → split_data → train_linear → evaluate.

    Paramètre `label` : texte affiché dans le print (ex: "simple", "multiple")
    pour identifier facilement quel modèle correspond à quel résultat.

    Retourne un dictionnaire avec : le modèle, X_test, y_test et les 3 métriques.
    """
    from src.features.build_features import get_X_y, TARGET
    X, y = get_X_y(df, features)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_linear(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    print(f"  [{label}] MAE={metrics['MAE']}€  RMSE={metrics['RMSE']}€  R²={metrics['R2']}")
    return {"model": model, "X_test": X_test, "y_test": y_test, **metrics}
