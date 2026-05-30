from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

TEST_FILE = DATA_DIR / "test.csv"
MODEL_FILE = MODEL_DIR / "model_pso.npz"

ULAZNI_STUPCI = [
    "age",
    "gender",
    "height_cm",
    "weight_kg",
    "sit and bend forward_cm",
    "sit-ups counts",
]

IZLAZNI_STUPAC = "broad jump_cm"


def relu(x):
    return np.maximum(0, x)


def mse(y_stvarni, y_pred):
    return np.mean((y_stvarni - y_pred) ** 2)


def rmse(y_stvarni, y_pred):
    return np.sqrt(mse(y_stvarni, y_pred))


def mae(y_stvarni, y_pred):
    return np.mean(np.abs(y_stvarni - y_pred))


def ispisi_metrike(naziv, y_stvarni, y_pred):
    print(f"{naziv} MSE  = {mse(y_stvarni, y_pred):.4f} cm^2")
    print(f"{naziv} RMSE = {rmse(y_stvarni, y_pred):.4f} cm")
    print(f"{naziv} MAE  = {mae(y_stvarni, y_pred):.4f} cm")


def ucitaj_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Nedostaje models/model_pso.npz. Prvo pokreni: python train_pso.py"
        )

    return np.load(MODEL_FILE, allow_pickle=True)


def predvidi(X, model):
    W1 = model["W1"]
    b1 = model["b1"]
    W2 = model["W2"]
    b2 = model["b2"]

    X_mean = model["X_mean"]
    X_std = model["X_std"]
    y_mean = model["y_mean"]
    y_std = model["y_std"]

    X_norm = (X - X_mean) / X_std

    skriveni = relu(X_norm @ W1 + b1)
    y_norm = skriveni @ W2 + b2

    y_cm = y_norm.ravel() * y_std + y_mean

    return y_cm


def main():
    if not TEST_FILE.exists():
        raise FileNotFoundError(
            "Nedostaje data/test.csv. Prvo pokreni: python prepare_data.py"
        )

    model = ucitaj_model()

    test_df = pd.read_csv(TEST_FILE)

    X_test = test_df[ULAZNI_STUPCI].to_numpy(dtype=float)
    y_test = test_df[IZLAZNI_STUPAC].to_numpy(dtype=float)

    y_pred = predvidi(X_test, model)

    print("Evaluacija najboljeg PSO modela na test skupu")
    print("Broj test uzoraka:", len(X_test))
    print("------------------------------------------")
    ispisi_metrike("Test", y_test, y_pred)
    print("------------------------------------------")


if __name__ == "__main__":
    main()