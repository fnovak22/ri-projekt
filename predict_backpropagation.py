from pathlib import Path
import json

import joblib
import numpy as np


MODEL_FILE = Path("models") / "model_backpropagation.joblib"
PERSON_CONFIG_FILE = Path("prediction-data") / "person.json"

# gender kodiranje u projektu:
# F = 1, M = 0


def ucitaj_podatke_osobe(config_file):
    if not config_file.exists():
        raise FileNotFoundError(
            f"Nedostaje {config_file}. "
            "Kopiraj prediction-data/person.example.json u prediction-data/person.json "
            "i upiši podatke osobe."
        )

    with open(config_file, "r", encoding="utf-8") as f:
        osoba = json.load(f)

    obavezni_kljucevi = [
        "age",
        "gender",
        "height_cm",
        "weight_kg",
        "sit_and_bend_forward_cm",
        "sit_ups_counts",
    ]

    for kljuc in obavezni_kljucevi:
        if kljuc not in osoba:
            raise KeyError(f"U {config_file} nedostaje ključ: {kljuc}")

    # Dozvoljeno je upisati gender kao broj ili kao tekst.
    # Interno koristimo: F = 1, M = 0.
    if isinstance(osoba["gender"], str):
        gender = osoba["gender"].strip().upper()

        if gender == "F":
            osoba["gender"] = 1
        elif gender == "M":
            osoba["gender"] = 0
        else:
            raise ValueError("gender mora biti 'F', 'M', 1 ili 0.")
    else:
        osoba["gender"] = int(osoba["gender"])

        if osoba["gender"] not in (0, 1):
            raise ValueError("gender mora biti 1 za F ili 0 za M.")

    return osoba


def napravi_ulazni_vektor(osoba):
    return np.array([
        osoba["age"],
        osoba["gender"],
        osoba["height_cm"],
        osoba["weight_kg"],
        osoba["sit_and_bend_forward_cm"],
        osoba["sit_ups_counts"],
    ], dtype=float)


def predvidi(spremljeno, X):
    model = spremljeno["model"]

    X_mean = spremljeno["X_mean"]
    X_std = spremljeno["X_std"]

    y_mean = float(spremljeno["y_mean"])
    y_std = float(spremljeno["y_std"])

    # Normalizacija ulaza na isti način kao kod treniranja
    X_norm = (X - X_mean) / X_std

    # MLPRegressor očekuje 2D oblik: jedan redak, više stupaca
    X_norm = X_norm.reshape(1, -1)

    # Predikcija je prvo u normaliziranom obliku
    y_norm = model.predict(X_norm)

    # Vraćanje predikcije u centimetre
    y_cm = y_norm * y_std + y_mean

    return float(y_cm[0])


def ispisi_podatke_osobe(osoba):
    spol = "F" if osoba["gender"] == 1 else "M"

    print("Podaci nove osobe:")
    print(f"Dob: {osoba['age']}")
    print(f"Spol: {spol}")
    print(f"Visina: {osoba['height_cm']} cm")
    print(f"Masa: {osoba['weight_kg']} kg")
    print(f"Sit and bend forward: {osoba['sit_and_bend_forward_cm']} cm")
    print(f"Broj trbušnjaka: {osoba['sit_ups_counts']}")


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Nedostaje models/model_backpropagation.joblib. "
            "Prvo pokreni: python train_backpropagation.py"
        )

    osoba = ucitaj_podatke_osobe(PERSON_CONFIG_FILE)
    spremljeno = joblib.load(MODEL_FILE)

    X = napravi_ulazni_vektor(osoba)
    predikcija = predvidi(spremljeno, X)

    print("------------------------------------------")
    print("Predikcija pomoću backpropagation modela")
    print("------------------------------------------")
    ispisi_podatke_osobe(osoba)
    print("------------------------------------------")
    print(f"Predviđeni skok u dalj iz mjesta: {predikcija:.2f} cm")


if __name__ == "__main__":
    main()