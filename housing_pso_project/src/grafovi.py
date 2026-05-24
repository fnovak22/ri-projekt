from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .modeli import RezultatModela
from .priprema_podataka import PripremljeniPodaci


def spremi_grafove(
    podaci: PripremljeniPodaci,
    bp: RezultatModela,
    pso: RezultatModela,
    folder: Path,
) -> None:
    folder.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(bp.povijest)
    plt.title("Backpropagation: gubitak kroz epohe")
    plt.xlabel("Epoha")
    plt.ylabel("MSE nad skaliranom cijenom")
    plt.tight_layout()
    plt.savefig(folder / "backpropagation_loss.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(pso.povijest)
    plt.title("PSO: najbolje rješenje kroz iteracije")
    plt.xlabel("Iteracija")
    plt.ylabel("Fitness (MSE nad skaliranom cijenom)")
    plt.tight_layout()
    plt.savefig(folder / "pso_konvergencija.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(podaci.y_test_original, bp.predikcije, alpha=0.5, label="Backpropagation")
    plt.scatter(podaci.y_test_original, pso.predikcije, alpha=0.5, label="PSO")
    granica_min = min(podaci.y_test_original.min(), bp.predikcije.min(), pso.predikcije.min())
    granica_max = max(podaci.y_test_original.max(), bp.predikcije.max(), pso.predikcije.max())
    plt.plot([granica_min, granica_max], [granica_min, granica_max], linestyle="--")
    plt.title("Stvarna cijena i predikcija modela")
    plt.xlabel("Stvarna cijena")
    plt.ylabel("Predviđena cijena")
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder / "stvarno_vs_predvideno.png", dpi=180)
    plt.close()


def spremi_rezultate(
    podaci: PripremljeniPodaci,
    bp: RezultatModela,
    pso: RezultatModela,
    folder: Path,
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rezultati = pd.DataFrame([
        {
            "metoda": bp.metoda,
            "MSE": bp.mse,
            "RMSE": bp.rmse,
            "MAE": bp.mae,
            "R2": bp.r2,
            "vrijeme_sekunde": bp.vrijeme_sekunde,
        },
        {
            "metoda": pso.metoda,
            "MSE": pso.mse,
            "RMSE": pso.rmse,
            "MAE": pso.mae,
            "R2": pso.r2,
            "vrijeme_sekunde": pso.vrijeme_sekunde,
        },
    ])
    rezultati.to_csv(folder / "usporedba_modela.csv", index=False)
    pd.DataFrame({"iteracija": range(len(pso.povijest)), "best_scaled_mse": pso.povijest}).to_csv(
        folder / "pso_konvergencija.csv", index=False
    )
    pd.DataFrame({
        "stvarna_cijena": podaci.y_test_original,
        "predikcija_backpropagation": bp.predikcije,
        "predikcija_pso": pso.predikcije,
    }).to_csv(folder / "predikcije_test_skup.csv", index=False)
