from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class PripremljeniPodaci:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    y_train_original: np.ndarray
    y_val_original: np.ndarray
    y_test_original: np.ndarray
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    feature_names: list[str]
    statistika: dict[str, Any]


def ucitaj_i_pripremi_podatke(config: dict[str, Any], korijenski_folder: Path) -> PripremljeniPodaci:
    """Učitava već transformirani CSV, uklanja nevaljane zapise i radi train/validation/test split.

    Važno: skaleri se prilagođavaju samo na trening skupu kako ne bi došlo do curenja
    informacija iz testnog skupa u treniranje modela.
    """
    putanja = korijenski_folder / config["data_path"]
    if not putanja.exists():
        raise FileNotFoundError(f"CSV nije pronađen: {putanja}")

    df = pd.read_csv(putanja)
    target = config["target_column"]
    if target not in df.columns:
        raise ValueError(f"CSV mora sadržavati ciljnu kolonu '{target}'.")

    pocetni_broj = len(df)
    broj_nula = int((df[target] <= 0).sum())
    broj_missing = int(df.isna().sum().sum())
    df = df.dropna().copy()

    if config.get("remove_zero_prices", True):
        df = df[df[target] > 0].copy()

    granica_outliera = None
    izbaceni_outlieri = 0
    if config.get("remove_price_outliers", True):
        kvantil = float(config.get("price_outlier_quantile", 0.99))
        granica_outliera = float(df[target].quantile(kvantil))
        prije = len(df)
        df = df[df[target] <= granica_outliera].copy()
        izbaceni_outlieri = prije - len(df)

    X = df.drop(columns=[target])
    y = df[[target]].to_numpy(dtype=float)

    random_state = int(config.get("random_state", 42))
    test_size = float(config.get("test_size", 0.15))
    validation_size = float(config.get("validation_size", 0.15))
    if test_size + validation_size >= 1:
        raise ValueError("Zbroj test_size i validation_size mora biti manji od 1.")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    val_relativno = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relativno, random_state=random_state
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)
    X_test_scaled = x_scaler.transform(X_test)
    y_train_scaled = y_scaler.fit_transform(y_train).ravel()
    y_val_scaled = y_scaler.transform(y_val).ravel()
    y_test_scaled = y_scaler.transform(y_test).ravel()

    statistika = {
        "pocetni_broj_zapisa": pocetni_broj,
        "nedostajuce_vrijednosti": broj_missing,
        "cijena_nula_ili_manja": broj_nula,
        "uklonjeni_outlieri": izbaceni_outlieri,
        "granica_outliera": granica_outliera,
        "broj_zapisa_nakon_obrade": len(df),
        "broj_atributa": X.shape[1],
        "train_zapisi": len(X_train),
        "validation_zapisi": len(X_val),
        "test_zapisi": len(X_test),
        "minimalna_cijena": float(df[target].min()),
        "maksimalna_cijena": float(df[target].max()),
        "prosjecna_cijena": float(df[target].mean()),
        "medijan_cijene": float(df[target].median()),
    }

    return PripremljeniPodaci(
        X_train=X_train_scaled.astype(float),
        X_val=X_val_scaled.astype(float),
        X_test=X_test_scaled.astype(float),
        y_train=y_train_scaled.astype(float),
        y_val=y_val_scaled.astype(float),
        y_test=y_test_scaled.astype(float),
        y_train_original=y_train.ravel(),
        y_val_original=y_val.ravel(),
        y_test_original=y_test.ravel(),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_names=X.columns.tolist(),
        statistika=statistika,
    )
