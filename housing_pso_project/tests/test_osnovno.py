import unittest

import numpy as np

from src.neuronska_mreza import JednostavnaNeuronskaMreza
from src.pso_optimizator import PSOOptimizator


class TestNeuronskaMreza(unittest.TestCase):
    def test_broj_parametara_i_predikcija(self):
        mreza = JednostavnaNeuronskaMreza(broj_ulaza=3, broj_skrivenih=2)
        self.assertEqual(mreza.broj_parametara, 11)
        cestica = np.zeros(mreza.broj_parametara)
        X = np.ones((4, 3))
        pred = mreza.predvidi(X, cestica)
        self.assertEqual(pred.shape, (4,))
        self.assertTrue(np.allclose(pred, 0))


class TestPSO(unittest.TestCase):
    def test_pso_pronalazi_nizu_vrijednost(self):
        optimizator = PSOOptimizator(
            broj_cestica=15,
            broj_dimenzija=2,
            broj_iteracija=30,
            random_state=42,
        )
        rezultat = optimizator.optimiziraj(lambda x: float(np.sum(x ** 2)))
        self.assertLessEqual(rezultat.povijest_fitnessa[-1], rezultat.povijest_fitnessa[0])
        self.assertEqual(len(rezultat.povijest_fitnessa), 31)


if __name__ == "__main__":
    unittest.main()
