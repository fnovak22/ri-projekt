import os
import joblib
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

TRAIN_DATOTEKA = "data/train.csv"
VAL_DATOTEKA = "data/val.csv"

MODEL_DIR = "models"
MODEL_DATOTEKA = "models/model_backpropagation.joblib"

ULAZNI_STUPCI = [
    "age",
    "gender",
    "height_cm",
    "weight_kg",
    "sit and bend forward_cm",
    "sit-ups counts",
]

IZLAZNI_STUPAC = "broad jump_cm"


def izracunaj_metrike(y_stvarno, y_predikcija):
    mse = mean_squared_error(y_stvarno, y_predikcija)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_stvarno, y_predikcija)

    return mse, rmse, mae


def ucitaj_i_pripremi_podatke():
    train_df = pd.read_csv(TRAIN_DATOTEKA)
    val_df = pd.read_csv(VAL_DATOTEKA)

    X_train = train_df[ULAZNI_STUPCI].values
    y_train = train_df[IZLAZNI_STUPAC].values

    X_val = val_df[ULAZNI_STUPCI].values
    y_val = val_df[IZLAZNI_STUPAC].values

    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)

    X_std[X_std == 0] = 1

    y_mean = y_train.mean()
    y_std = y_train.std()

    if y_std == 0:
        y_std = 1

    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std

    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std

    return (
        X_train_norm,
        y_train_norm,
        X_val_norm,
        y_val_norm,
        y_train,
        y_val,
        X_mean,
        X_std,
        y_mean,
        y_std,
    )


def treniraj_backpropagation(X_train, y_train):
    model = MLPRegressor(
        hidden_layer_sizes=(10,),
        activation="relu",
        solver="sgd",
        learning_rate_init=0.01,
        learning_rate="adaptive",
        momentum=0.9,
        max_iter=2000,
        batch_size=64,
        tol=1e-6,
        n_iter_no_change=50,
        random_state=42,
        verbose=True
    )

    model.fit(X_train, y_train)

    return model

def main():
    print("Učitavanje i priprema podataka...")

    (
        X_train,
        y_train,
        X_val,
        y_val,
        y_train_original,
        y_val_original,
        X_mean,
        X_std,
        y_mean,
        y_std,
    ) = ucitaj_i_pripremi_podatke()

    print("Treniranje neuronske mreže backpropagation metodom...")

    model = treniraj_backpropagation(X_train, y_train)

    print("\nEvaluacija modela...")

    y_train_pred_norm = model.predict(X_train)
    y_val_pred_norm = model.predict(X_val)

    y_train_pred = y_train_pred_norm * y_std + y_mean
    y_val_pred = y_val_pred_norm * y_std + y_mean

    train_mse, train_rmse, train_mae = izracunaj_metrike(
        y_train_original,
        y_train_pred
    )

    val_mse, val_rmse, val_mae = izracunaj_metrike(
        y_val_original,
        y_val_pred
    )

    print("\nRezultati backpropagation modela:")
    print("--------------------------------")
    print(f"Train MSE:  {train_mse:.4f}")
    print(f"Train RMSE: {train_rmse:.4f} cm")
    print(f"Train MAE:  {train_mae:.4f} cm")

    print()

    print(f"Validation MSE:  {val_mse:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f} cm")
    print(f"Validation MAE:  {val_mae:.4f} cm")

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "X_mean": X_mean,
            "X_std": X_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "ulazni_stupci": ULAZNI_STUPCI,
            "izlazni_stupac": IZLAZNI_STUPAC,
            "arhitektura": "6-10-1",
            "metoda_treniranja": "backpropagation_sgd",
            "train_mse": train_mse,
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "val_mse": val_mse,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
        },
        MODEL_DATOTEKA
    )

    print()
    print(f"Model je spremljen u: {MODEL_DATOTEKA}")


if __name__ == "__main__":
    main()