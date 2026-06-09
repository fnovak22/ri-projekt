from pathlib import Path
import json

import numpy as np

MODEL_FILE = Path("models") / "model_pso.npz"
PERSON_CONFIG_FILE = Path("prediction-data") / "person.json"

# gender kodiranje u projektu:
# F = 1, M = 0


def relu(x):
    return np.maximum(0, x)


def ucitaj_podatke_osobe(config_file):
    if not config_file.exists():
        raise FileNotFoundError(
            f"Nedostaje {config_file}. "
            "Kopiraj config/person.example.json u config/person.json i upiši podatke osobe."
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


def predvidi(model, X):
    W1 = model["W1"]
    b1 = model["b1"]
    W2 = model["W2"]
    b2 = model["b2"]

    X_mean = model["X_mean"]
    X_std = model["X_std"]
    y_mean = float(model["y_mean"])
    y_std = float(model["y_std"])

    X_norm = (X - X_mean) / X_std

    skriveni = relu(X_norm @ W1 + b1)
    y_norm = skriveni @ W2 + b2

    y_cm = y_norm * y_std + y_mean

    return float(y_cm[0])


def napravi_ulazni_vektor(osoba):
    return np.array([
        osoba["age"],
        osoba["gender"],
        osoba["height_cm"],
        osoba["weight_kg"],
        osoba["sit_and_bend_forward_cm"],
        osoba["sit_ups_counts"],
    ], dtype=float)


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
            "Nedostaje models/model_pso.npz. "
        )

    osoba = ucitaj_podatke_osobe(PERSON_CONFIG_FILE)
    model = np.load(MODEL_FILE, allow_pickle=True)

    X = napravi_ulazni_vektor(osoba)
    predikcija = predvidi(model, X)

    print("------------------------------------------")
    ispisi_podatke_osobe(osoba)
    print("------------------------------------------")
    print(f"Predviđeni skok u dalj iz mjesta: {predikcija:.2f} cm")


if __name__ == "__main__":
    main()
