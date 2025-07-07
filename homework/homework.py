# flake8: noqa: E501
#
# Este script construye un modelo de regresión para pronosticar el precio de vehículos usados.
# El dataset original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehículo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual (objetivo).
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_Type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya está dividido en conjuntos de entrenamiento y prueba en "files/input/".

import os
import gzip
import json
import pickle
from glob import glob
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error


def cargar_data():
    train = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
    test = pd.read_csv("files/input/test_data.csv.zip", compression="zip")
    return train, test


def preprocess(df):
    df = df.copy()
    df["Age"] = 2021 - df["Year"]
    return df.drop(columns=["Year", "Car_Name"])


def split_features_target(df):
    X = df.drop(columns=["Present_Price"])
    y = df["Present_Price"]
    return X, y


def build_pipeline(X):
    cat_features = ["Fuel_Type", "Selling_type", "Transmission"]
    num_features = [col for col in X.columns if col not in cat_features]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(), cat_features),
        ("num", MinMaxScaler(), num_features)
    ])

    return Pipeline([
        ("preprocessing", preprocessor),
        ("select", SelectKBest(f_regression)),
        ("regressor", LinearRegression())
    ])


def optimize_model(pipeline):
    param_grid = {
        "select__k": range(1, 25),
        "regressor__fit_intercept": [True, False],
        "regressor__positive": [True, False],
    }

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=10,
        n_jobs=-1,
        verbose=1
    )


def save_model(model, path="files/models/model.pkl.gz"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(model, f)


def compute_metrics(y_true, y_pred, dataset):
    return {
        "type": "metrics",
        "dataset": dataset,
        "r2": r2_score(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "mad": median_absolute_error(y_true, y_pred)
    }


def save_metrics(train_metrics, test_metrics, path="files/output/metrics.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(train_metrics) + "\n")
        f.write(json.dumps(test_metrics) + "\n")


def main():
    train_df, test_df = cargar_data()
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    pipeline = build_pipeline(X_train)
    model = optimize_model(pipeline)
    model.fit(X_train, y_train)

    save_model(model)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = compute_metrics(y_train, y_train_pred, "train")
    test_metrics = compute_metrics(y_test, y_test_pred, "test")

    save_metrics(train_metrics, test_metrics)


if __name__ == "__main__":
    main()
