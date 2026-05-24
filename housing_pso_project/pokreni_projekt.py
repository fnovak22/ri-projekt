from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.grafovi import spremi_grafove, spremi_rezultate
from src.modeli import treniraj_backpropagation, treniraj_pso
from src.priprema_podataka import ucitaj_i_pripremi_podatke


def main() -> None:
    parser = argparse.ArgumentParser(description="Predikcija cijena nekretnina: PSO nasuprot backpropagationu")
    parser.add_argument("--config", default="config.json", help="Konfiguracijska JSON datoteka")
    args = parser.parse_args()

    korijen = Path(__file__).resolve().parent
    config_putanja = korijen / args.config
    with open(config_putanja, "r", encoding="utf-8") as datoteka:
        config = json.load(datoteka)

    podaci = ucitaj_i_pripremi_podatke(config, korijen)
    print(f"Konfiguracija: {config_putanja.name}")
    print("=== PODACI ===")
    for kljuc, vrijednost in podaci.statistika.items():
        print(f"{kljuc}: {vrijednost}")

    print("\n=== BACKPROPAGATION ===")
    bp = treniraj_backpropagation(podaci, config)
    print(f"MAE: {bp.mae:,.2f} | RMSE: {bp.rmse:,.2f} | R2: {bp.r2:.4f} | vrijeme: {bp.vrijeme_sekunde:.2f}s")
    print(f"Detalji: {bp.detalji}")

    print("\n=== PSO ===")
    pso = treniraj_pso(podaci, config)
    print(f"MAE: {pso.mae:,.2f} | RMSE: {pso.rmse:,.2f} | R2: {pso.r2:.4f} | vrijeme: {pso.vrijeme_sekunde:.2f}s")
    print(f"Detalji: {pso.detalji}")

    spremi_rezultate(podaci, bp, pso, korijen / "results" / "tablice")
    spremi_grafove(podaci, bp, pso, korijen / "results" / "grafovi")
    print("\nRezultati su spremljeni u results/tablice, a grafovi u results/grafovi.")


if __name__ == "__main__":
    main()
