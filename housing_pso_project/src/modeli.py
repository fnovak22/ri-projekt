from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

from .neuronska_mreza import JednostavnaNeuronskaMreza
from .pso_optimizator import PSOOptimizator
from .priprema_podataka import PripremljeniPodaci


@dataclass
class RezultatModela:
    metoda: str
    mse: float
    rmse: float
    mae: float
    r2: float
    vrijeme_sekunde: float
    predikcije: np.ndarray
    povijest: list[float]
    detalji: dict[str, Any]


def izracunaj_metrike(y_stvarno: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_stvarno, y_pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_stvarno, y_pred)),
        "r2": float(r2_score(y_stvarno, y_pred)),
    }


def izdvoji_zajednicki_uzorak(podaci: PripremljeniPodaci, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Vraća isti training uzorak za oba modela radi poštene usporedbe."""
    velicina = int(config.get("comparison_training_sample_size", len(podaci.X_train)))
    if velicina <= 0 or velicina >= len(podaci.X_train):
        return podaci.X_train, podaci.y_train
    rng = np.random.default_rng(int(config["random_state"]))
    indeksi = rng.choice(len(podaci.X_train), size=velicina, replace=False)
    return podaci.X_train[indeksi], podaci.y_train[indeksi]


def treniraj_backpropagation(podaci: PripremljeniPodaci, config: dict[str, Any]) -> RezultatModela:
    broj_skrivenih = int(config["hidden_neurons"])
    bp = config["backprop"]
    X_train, y_train = izdvoji_zajednicki_uzorak(podaci, config)
    model = MLPRegressor(
        hidden_layer_sizes=(broj_skrivenih,),
        activation="relu",
        solver="adam",
        max_iter=int(bp["max_iter"]),
        learning_rate_init=float(bp["learning_rate_init"]),
        early_stopping=bool(bp["early_stopping"]),
        n_iter_no_change=int(bp["n_iter_no_change"]),
        validation_fraction=0.15,
        random_state=int(config["random_state"]),
    )
    pocetak = perf_counter()
    model.fit(X_train, y_train)
    trajanje = perf_counter() - pocetak
    y_pred_scaled = model.predict(podaci.X_test)
    y_pred = podaci.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    metrike = izracunaj_metrike(podaci.y_test_original, y_pred)
    return RezultatModela(
        metoda="Backpropagation (Adam)",
        vrijeme_sekunde=trajanje,
        predikcije=y_pred,
        povijest=[float(x) for x in model.loss_curve_],
        detalji={
            "broj_epoha": int(model.n_iter_),
            "broj_skrivenih_neurona": broj_skrivenih,
            "broj_train_zapisa": len(X_train),
        },
        **metrike,
    )


def treniraj_pso(podaci: PripremljeniPodaci, config: dict[str, Any]) -> RezultatModela:
    pso_cfg = config["pso"]
    broj_skrivenih = int(config["hidden_neurons"])
    mreza = JednostavnaNeuronskaMreza(podaci.X_train.shape[1], broj_skrivenih)
    X_train, y_train = izdvoji_zajednicki_uzorak(podaci, config)
    optimizator = PSOOptimizator(
        broj_cestica=int(pso_cfg["particles"]),
        broj_dimenzija=mreza.broj_parametara,
        broj_iteracija=int(pso_cfg["iterations"]),
        w_pocetak=float(pso_cfg["inertia_start"]),
        w_kraj=float(pso_cfg["inertia_end"]),
        c1=float(pso_cfg["cognitive_coefficient"]),
        c2=float(pso_cfg["social_coefficient"]),
        donja_granica=float(pso_cfg["position_min"]),
        gornja_granica=float(pso_cfg["position_max"]),
        v_min=float(pso_cfg["velocity_min"]),
        v_max=float(pso_cfg["velocity_max"]),
        random_state=int(config["random_state"]),
    )
    pocetak = perf_counter()
    rezultat = optimizator.optimiziraj(lambda cestica: mreza.mse(X_train, y_train, cestica))
    trajanje = perf_counter() - pocetak
    y_pred_scaled = mreza.predvidi(podaci.X_test, rezultat.najbolje_pozicije)
    y_pred = podaci.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    metrike = izracunaj_metrike(podaci.y_test_original, y_pred)
    return RezultatModela(
        metoda="PSO + MLP",
        vrijeme_sekunde=trajanje,
        predikcije=y_pred,
        povijest=[float(x) for x in rezultat.povijest_fitnessa],
        detalji={
            "broj_skrivenih_neurona": broj_skrivenih,
            "broj_parametara_mreze": mreza.broj_parametara,
            "broj_cestica": int(pso_cfg["particles"]),
            "broj_iteracija": int(pso_cfg["iterations"]),
            "broj_train_zapisa": len(X_train),
            "najbolji_scaled_train_mse": rezultat.najbolja_vrijednost,
        },
        **metrike,
    )
