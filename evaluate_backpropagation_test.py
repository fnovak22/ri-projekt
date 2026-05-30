import json
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error


# -----------------------------
# Postavke
# -----------------------------

TEST_DATOTEKA = "data/test.csv"
MODEL_DATOTEKA = "models/model_backpropagation.joblib"

RESULTS_DIR = "results"
RESULTS_DATOTEKA = "results/backpropagation_test_results.json"


# -----------------------------
# Funkcija za metrike
# -----------------------------

def izracunaj_metrike(y_stvarno, y_predikcija):
    mse = mean_squared_error(y_stvarno, y_predikcija)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_stvarno, y_predikcija)

    return mse, rmse, mae


# -----------------------------
# Učitavanje modela
# -----------------------------

def ucitaj_model():
    if not os.path.exists(MODEL_DATOTEKA):
        raise FileNotFoundError(
            f"Model nije pronađen: {MODEL_DATOTEKA}. "
            "Prvo pokreni train_backpropagation.py."
        )

    spremljeno = joblib.load(MODEL_DATOTEKA)

    return spremljeno


# -----------------------------
# Evaluacija na test skupu
# -----------------------------

def evaluiraj_na_test_skupu(spremljeno):
    test_df = pd.read_csv(TEST_DATOTEKA)

    model = spremljeno["model"]
    X_mean = spremljeno["X_mean"]
    X_std = spremljeno["X_std"]
    y_mean = spremljeno["y_mean"]
    y_std = spremljeno["y_std"]

    ulazni_stupci = spremljeno["ulazni_stupci"]
    izlazni_stupac = spremljeno["izlazni_stupac"]

    X_test = test_df[ulazni_stupci].values
    y_test = test_df[izlazni_stupac].values

    # Test podaci se normaliziraju pomoću prosjeka i std-a iz TRAIN skupa
    X_test_norm = (X_test - X_mean) / X_std

    # Predikcija je prvo u normaliziranom obliku
    y_pred_norm = model.predict(X_test_norm)

    # Vraćanje predikcije u centimetre
    y_pred = y_pred_norm * y_std + y_mean

    mse, rmse, mae = izracunaj_metrike(y_test, y_pred)

    return mse, rmse, mae


# -----------------------------
# Glavni program
# -----------------------------

def main():
    print("Učitavanje backpropagation modela...")

    spremljeno = ucitaj_model()

    print("Evaluacija backpropagation modela na test skupu...")

    test_mse, test_rmse, test_mae = evaluiraj_na_test_skupu(spremljeno)

    print("\nRezultati backpropagation modela na test skupu:")
    print("------------------------------------------------")
    print(f"Test MSE:  {test_mse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f} cm")
    print(f"Test MAE:  {test_mae:.4f} cm")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    rezultati = {
        "model": "backpropagation",
        "arhitektura": spremljeno.get("arhitektura", "6-10-1"),
        "metoda_treniranja": spremljeno.get("metoda_treniranja", "backpropagation"),
        "test_mse": float(test_mse),
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
    }

    with open(RESULTS_DATOTEKA, "w", encoding="utf-8") as f:
        json.dump(rezultati, f, indent=4, ensure_ascii=False)

    print()
    print(f"Rezultati su spremljeni u: {RESULTS_DATOTEKA}")


if __name__ == "__main__":
    main()