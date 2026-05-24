from __future__ import annotations

import numpy as np


class JednostavnaNeuronskaMreza:
    """MLP mreža s jednim skrivenim slojem.

    Kod PSO treniranja parametri mreže moraju biti zapisani u jednom vektoru jer
    jedna čestica predstavlja jedan kompletan skup težina i bias vrijednosti.
    """

    def __init__(self, broj_ulaza: int, broj_skrivenih: int = 6) -> None:
        self.broj_ulaza = broj_ulaza
        self.broj_skrivenih = broj_skrivenih
        self.broj_parametara = (
            broj_ulaza * broj_skrivenih
            + broj_skrivenih
            + broj_skrivenih * 1
            + 1
        )

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def raspakiraj_parametre(self, cestica: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if cestica.shape[0] != self.broj_parametara:
            raise ValueError("Čestica nema očekivani broj parametara mreže.")
        i = 0
        n_w1 = self.broj_ulaza * self.broj_skrivenih
        W1 = cestica[i:i + n_w1].reshape(self.broj_ulaza, self.broj_skrivenih)
        i += n_w1
        b1 = cestica[i:i + self.broj_skrivenih]
        i += self.broj_skrivenih
        W2 = cestica[i:i + self.broj_skrivenih].reshape(self.broj_skrivenih, 1)
        i += self.broj_skrivenih
        b2 = float(cestica[i])
        return W1, b1, W2, b2

    def predvidi(self, X: np.ndarray, cestica: np.ndarray) -> np.ndarray:
        W1, b1, W2, b2 = self.raspakiraj_parametre(cestica)
        skriveni = self._relu(X @ W1 + b1)
        izlaz = skriveni @ W2 + b2
        return izlaz.ravel()

    def mse(self, X: np.ndarray, y: np.ndarray, cestica: np.ndarray) -> float:
        predikcije = self.predvidi(X, cestica)
        return float(np.mean((y - predikcije) ** 2))
