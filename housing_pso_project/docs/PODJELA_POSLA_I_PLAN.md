# Podjela posla i redoslijed rada — tim od 4 osobe

## Osoba 1 — Podaci i priprema

- provjera i opis CSV skupa;
- obrazloženje one-hot kodiranja gradova;
- analiza nultih cijena i outliera;
- priprema tablice atributa i početnih grafova o datasetu;
- pisanje dijela dokumentacije „Skup podataka” i „Priprema podataka”.

## Osoba 2 — Neuronska mreža i backpropagation

- razumijevanje koda `src/neuronska_mreza.py` i baseline implementacije u `src/modeli.py`;
- pokretanje backpropagation modela;
- analiza loss grafa i baseline metrika;
- pisanje dijela dokumentacije „Neuronska mreža” i „Referentno treniranje”.

## Osoba 3 — PSO algoritam

- razumijevanje i dorada `src/pso_optimizator.py`;
- povezivanje s logikom prethodnog PSO zadatka;
- analiza značenja čestice kao vektora težina;
- provođenje PSO pokretanja i spremanje konvergencije;
- pisanje dijela dokumentacije „Treniranje PSO algoritmom”.

## Osoba 4 — Eksperimenti, rezultati i predaja

- pokretanje `eksperimenti_pso.py`;
- izrada završnih usporednih tablica i umetanje grafova u dokumentaciju;
- pisanje rasprave rezultata i zaključka;
- priprema prezentacije i prijave teme na forum.

## Redoslijed izvedbe

1. Svi pregledaju `README.md` i potvrde temu te dataset.
2. Pokrenuti `py pokreni_projekt.py` s početnom konfiguracijom.
3. Provjeriti dobivene tablice i grafove.
4. Pokrenuti `py eksperimenti_pso.py` i odabrati bolju PSO konfiguraciju.
5. Za završni pokušaj koristiti `py pokreni_projekt.py --config config_final.json`.
6. Rezultate završnog izvođenja unijeti u `docs/DOKUMENTACIJA.md`.
7. Iz dokumentacije pripremiti kratku prezentaciju.
