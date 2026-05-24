from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class PSORezultat:
    najbolje_pozicije: np.ndarray
    najbolja_vrijednost: float
    povijest_fitnessa: list[float]


class PSOOptimizator:
    """Klasični PSO za minimizaciju funkcije pogreške.

    Logika se nadovezuje na vaš prethodni PSO zadatak: čestice imaju poziciju,
    brzinu, osobno najbolje rješenje i globalno najbolje rješenje. Ovdje pozicija
    čestice više nije nekoliko varijabli funkcije, nego sve težine neuronske mreže.
    """

    def __init__(
        self,
        broj_cestica: int,
        broj_dimenzija: int,
        broj_iteracija: int,
        w_pocetak: float = 0.9,
        w_kraj: float = 0.4,
        c1: float = 1.49445,
        c2: float = 1.49445,
        donja_granica: float = -2.0,
        gornja_granica: float = 2.0,
        v_min: float = -0.3,
        v_max: float = 0.3,
        random_state: int = 42,
    ) -> None:
        self.broj_cestica = broj_cestica
        self.broj_dimenzija = broj_dimenzija
        self.broj_iteracija = broj_iteracija
        self.w_pocetak = w_pocetak
        self.w_kraj = w_kraj
        self.c1 = c1
        self.c2 = c2
        self.donja_granica = donja_granica
        self.gornja_granica = gornja_granica
        self.v_min = v_min
        self.v_max = v_max
        self.rng = np.random.default_rng(random_state)

    def optimiziraj(self, fitness_funkcija: Callable[[np.ndarray], float]) -> PSORezultat:
        x = self.rng.uniform(
            self.donja_granica,
            self.gornja_granica,
            size=(self.broj_cestica, self.broj_dimenzija),
        )
        v = self.rng.uniform(
            self.v_min, self.v_max, size=(self.broj_cestica, self.broj_dimenzija)
        )

        vrijednosti = np.array([fitness_funkcija(c) for c in x])
        osobno_najbolje = x.copy()
        osobno_najbolje_vrijednosti = vrijednosti.copy()
        indeks = int(np.argmin(vrijednosti))
        globalno_najbolje = x[indeks].copy()
        globalno_najbolja_vrijednost = float(vrijednosti[indeks])
        povijest = [globalno_najbolja_vrijednost]

        for iteracija in range(self.broj_iteracija):
            udio = iteracija / max(1, self.broj_iteracija - 1)
            w = self.w_pocetak + udio * (self.w_kraj - self.w_pocetak)
            r1 = self.rng.random(size=x.shape)
            r2 = self.rng.random(size=x.shape)
            v = (
                w * v
                + self.c1 * r1 * (osobno_najbolje - x)
                + self.c2 * r2 * (globalno_najbolje - x)
            )
            v = np.clip(v, self.v_min, self.v_max)
            x = np.clip(x + v, self.donja_granica, self.gornja_granica)
            vrijednosti = np.array([fitness_funkcija(c) for c in x])

            poboljsanje = vrijednosti < osobno_najbolje_vrijednosti
            osobno_najbolje[poboljsanje] = x[poboljsanje]
            osobno_najbolje_vrijednosti[poboljsanje] = vrijednosti[poboljsanje]

            indeks = int(np.argmin(osobno_najbolje_vrijednosti))
            if osobno_najbolje_vrijednosti[indeks] < globalno_najbolja_vrijednost:
                globalno_najbolja_vrijednost = float(osobno_najbolje_vrijednosti[indeks])
                globalno_najbolje = osobno_najbolje[indeks].copy()
            povijest.append(globalno_najbolja_vrijednost)

        return PSORezultat(globalno_najbolje, globalno_najbolja_vrijednost, povijest)
