from __future__ import annotations

import copy
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.modeli import treniraj_pso
from src.priprema_podataka import ucitaj_i_pripremi_podatke


def main() -> None:
    korijen = Path(__file__).resolve().parent
    with open(korijen / "config.json", "r", encoding="utf-8") as datoteka:
        osnovni_config = json.load(datoteka)
    podaci = ucitaj_i_pripremi_podatke(osnovni_config, korijen)

    konfiguracije = [
        {"particles": 10, "iterations": 40},
        {"particles": 20, "iterations": 80},
        {"particles": 30, "iterations": 100},
    ]
    rezultati = []
    for postavke in konfiguracije:
        config = copy.deepcopy(osnovni_config)
        config["pso"].update(postavke)
        print(f"Pokrećem PSO: {postavke}")
        pocetak = perf_counter()
        rezultat = treniraj_pso(podaci, config)
        ukupno = perf_counter() - pocetak
        rezultati.append({
            **postavke,
            "MSE": rezultat.mse,
            "RMSE": rezultat.rmse,
            "MAE": rezultat.mae,
            "R2": rezultat.r2,
            "vrijeme_sekunde": ukupno,
        })

    izlaz = pd.DataFrame(rezultati)
    izlazna_putanja = korijen / "results" / "tablice" / "eksperimenti_pso.csv"
    izlazna_putanja.parent.mkdir(parents=True, exist_ok=True)
    izlaz.to_csv(izlazna_putanja, index=False)
    print(izlaz.to_string(index=False))
    print(f"Spremljeno: {izlazna_putanja}")


if __name__ == "__main__":
    main()
