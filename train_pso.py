from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "val.csv"
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

BROJ_ULAZNIH_NEURONA = 6
BROJ_SKRIVENIH_NEURONA = 10
BROJ_IZLAZNIH_NEURONA = 1

BROJ_CESTICA = 40
BROJ_ITERACIJA = 1000

INERCIJA = 0.7
C1 = 1.5
C2 = 1.5

DONJA_GRANICA = -1.0
GORNJA_GRANICA = 1.0
V_MIN = -0.2
V_MAX = 0.2

RANDOM_SEED = 67


def relu(x):
    return np.maximum(0, x)

def mse(y_stvarni, y_pred):
    return np.mean((y_stvarni - y_pred) ** 2)

def rmse(y_stvarni, y_pred):
    return np.sqrt(mse(y_stvarni, y_pred))

def mae(y_stvarni, y_pred):
    return np.mean(np.abs(y_stvarni - y_pred))

def broj_parametara():
    return (
        BROJ_ULAZNIH_NEURONA * BROJ_SKRIVENIH_NEURONA
        + BROJ_SKRIVENIH_NEURONA
        + BROJ_SKRIVENIH_NEURONA * BROJ_IZLAZNIH_NEURONA
        + BROJ_IZLAZNIH_NEURONA
    )


def raspakiraj_cesticu(cestica):
    indeks = 0

    velicina_w1 = BROJ_ULAZNIH_NEURONA * BROJ_SKRIVENIH_NEURONA
    W1 = cestica[indeks:indeks + velicina_w1].reshape(BROJ_ULAZNIH_NEURONA, BROJ_SKRIVENIH_NEURONA)
    indeks += velicina_w1

    b1 = cestica[indeks:indeks + BROJ_SKRIVENIH_NEURONA]
    indeks += BROJ_SKRIVENIH_NEURONA

    velicina_w2 = BROJ_SKRIVENIH_NEURONA * BROJ_IZLAZNIH_NEURONA
    W2 = cestica[indeks:indeks + velicina_w2].reshape(BROJ_SKRIVENIH_NEURONA, BROJ_IZLAZNIH_NEURONA)
    indeks += velicina_w2

    b2 = cestica[indeks:indeks + BROJ_IZLAZNIH_NEURONA]

    return W1, b1, W2, b2


def izracunaj_izlaz_mreze(X, cestica):
    W1, b1, W2, b2 = raspakiraj_cesticu(cestica)

    skriveni_net = X @ W1 + b1

    skriveni_izlaz = relu(skriveni_net)

    izlaz = skriveni_izlaz @ W2 + b2

    return izlaz.ravel()


def fitness(cestica, X_train, y_train):
    y_pred = izracunaj_izlaz_mreze(X_train, cestica)

    return mse(y_train, y_pred)


def ucitaj_splitove():
    if not TRAIN_FILE.exists() or not VAL_FILE.exists():
        raise FileNotFoundError(
            "Nedostaju data/train.csv ili data/val.csv. "
        )

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    X_train = train_df[ULAZNI_STUPCI].to_numpy(dtype=float)
    y_train = train_df[IZLAZNI_STUPAC].to_numpy(dtype=float)

    X_val = val_df[ULAZNI_STUPCI].to_numpy(dtype=float)
    y_val = val_df[IZLAZNI_STUPAC].to_numpy(dtype=float)

    X_prosjek = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0] = 1

    y_prosjek = y_train.mean()
    y_std = y_train.std()
    if y_std == 0:
        y_std = 1

    X_train_normalizirano = (X_train - X_prosjek) / X_std
    X_val_normalizirano = (X_val - X_prosjek) / X_std

    y_train_normalizirano = (y_train - y_prosjek) / y_std

    return (
        X_train_normalizirano, y_train_normalizirano,
        X_val_normalizirano,
        y_train, y_val,
        X_prosjek, X_std, y_prosjek, y_std,
    )


def treniraj_pso(X_train, y_train):
    #rng = np.random.default_rng(RANDOM_SEED)
    rng = np.random.default_rng()

    dimenzija = broj_parametara()

    pozicije = rng.uniform(
        DONJA_GRANICA,
        GORNJA_GRANICA,
        size=(BROJ_CESTICA, dimenzija),
    )

    brzine = rng.uniform(
        V_MIN,
        V_MAX,
        size=(BROJ_CESTICA, dimenzija),
    )

    personal_best_pozicije = pozicije.copy()
    personal_best_vrijednosti = np.array([
        fitness(pozicije[i], X_train, y_train)
        for i in range(BROJ_CESTICA)
    ])

    najbolji_indeks = np.argmin(personal_best_vrijednosti)
    global_best_pozicija = personal_best_pozicije[najbolji_indeks].copy()
    global_best_vrijednost = personal_best_vrijednosti[najbolji_indeks]

    for iteracija in range(BROJ_ITERACIJA):
        for i in range(BROJ_CESTICA):
            r1 = rng.random(dimenzija)
            r2 = rng.random(dimenzija)

            brzine[i] = (
                INERCIJA * brzine[i]
                + C1 * r1 * (personal_best_pozicije[i] - pozicije[i])
                + C2 * r2 * (global_best_pozicija - pozicije[i])
            )

            brzine[i] = np.clip(brzine[i], V_MIN, V_MAX)

            pozicije[i] = pozicije[i] + brzine[i]

            vrijednost = fitness(pozicije[i], X_train, y_train)

            if vrijednost < personal_best_vrijednosti[i]:
                personal_best_vrijednosti[i] = vrijednost
                personal_best_pozicije[i] = pozicije[i].copy()

                if vrijednost < global_best_vrijednost:
                    global_best_vrijednost = vrijednost
                    global_best_pozicija = pozicije[i].copy()

        if iteracija % 10 == 0 or iteracija == BROJ_ITERACIJA - 1:
            print(f"Iteracija {iteracija + 1}/{BROJ_ITERACIJA}, train MSE normalizirano = {global_best_vrijednost:.6f}")

    return global_best_pozicija, global_best_vrijednost


def denormaliziraj_y(y_norm, y_mean, y_std):
    return y_norm * y_std + y_mean


def ispisi_metrike(naziv, y_stvarni, y_pred):
    print(f"{naziv} Prosjecno kvadratno odstupanje = {mse(y_stvarni, y_pred):.4f} cm^2")
    print(f"{naziv} Korijen prosjecnog kvadratnog odstupanje = {rmse(y_stvarni, y_pred):.4f} cm")
    print(f"{naziv} Prosjecno apsolutno odstupanje  = {mae(y_stvarni, y_pred):.4f} cm")


def main():

    (
        X_train_normalizirano, y_train_normalizirano,
        X_val_normalizirano,
        y_train_original, y_val_original,
        X_mean, X_std, y_mean, y_std,
    ) = ucitaj_splitove()

    print("Broj train uzoraka:", len(X_train_normalizirano))
    print("Broj val uzoraka:", len(X_val_normalizirano))
    print("Broj parametara mreže:", broj_parametara())
    print("------------------------------------------")

    najbolja_cestica, _ = treniraj_pso(X_train_normalizirano, y_train_normalizirano)

    train_predvidene_normalizirane_vrijednosti = izracunaj_izlaz_mreze(X_train_normalizirano, najbolja_cestica)
    val_predvidene_normalizirane_vrijednosti = izracunaj_izlaz_mreze(X_val_normalizirano, najbolja_cestica)

    train_predvidene_vrijednosti = denormaliziraj_y(train_predvidene_normalizirane_vrijednosti, y_mean, y_std)
    val_predvidene_vrijednosti = denormaliziraj_y(val_predvidene_normalizirane_vrijednosti, y_mean, y_std)

    print("------------------------------------------")
    ispisi_metrike("Train", y_train_original, train_predvidene_vrijednosti)
    print("------------------------------------------")
    ispisi_metrike("Val", y_val_original, val_predvidene_vrijednosti)
    print("------------------------------------------")

    W1, b1, W2, b2 = raspakiraj_cesticu(najbolja_cestica)

    MODEL_DIR.mkdir(exist_ok=True)
    np.savez(
        MODEL_FILE,
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        X_mean=X_mean,
        X_std=X_std,
        y_mean=y_mean,
        y_std=y_std,
        ulazni_stupci=np.array(ULAZNI_STUPCI),
    )

    print(f"Model je spremljen u datoteku: {MODEL_FILE}")

if __name__ == "__main__":
    main()
