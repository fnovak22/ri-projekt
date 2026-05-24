# Predikcija cijena nekretnina neuronskom mrežom treniranom PSO algoritmom

## Cilj projekta

Projekt predviđa prodajnu cijenu nekretnine na temelju njezinih karakteristika i lokacije. Uspoređuju se dvije metode treniranja iste vrste neuronske mreže:

1. **Backpropagation / Adam** kao standardni referentni model.
2. **Particle Swarm Optimization (PSO)** kao glavna metoda računalne inteligencije, gdje jedna čestica predstavlja sve težine i bias vrijednosti neuronske mreže.

## Što je već pripremljeno

- `data/data_housing_cleaned_transformed.csv` — dostavljeni očišćeni skup s one-hot gradovima.
- `pokreni_projekt.py` — cjelovito pokretanje: obrada, split, treniranje, evaluacija i grafovi.
- `eksperimenti_pso.py` — usporedba više PSO konfiguracija.
- `docs/DOKUMENTACIJA.md` — predložak dokumentacije s opisom praktičnog dijela.
- `stari_zadatak/` — originalni PSO zadatak za usporedbu logike.

## Zašto se još obrađuju podaci

U CSV-u postoje cijene jednake nuli i ekstremne vrijednosti. U `config.json` je zadano:

- uklanjanje zapisa s cijenom `<= 0`
- uklanjanje gornjih 1 % ekstremnih cijena
- podjela na **70 % train**, **15 % validation**, **15 % test**
- skaliranje ulaza i cilja samo na temelju trening skupa
- isti reprezentativni trening podskup za PSO i backpropagation u osnovnoj usporedbi

Postavke se mogu promijeniti u `config.json`.

## Instalacija i pokretanje u Windows PowerShellu

```powershell
cd D:\RI\PROJEKT\housing_pso_project
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py pokreni_projekt.py
```

Za dodatne PSO eksperimente:

```powershell
py eksperimenti_pso.py
```

## Struktura projekta

```text
housing_pso_project/
├── data/
│   └── data_housing_cleaned_transformed.csv
├── docs/
│   └── DOKUMENTACIJA.md
├── results/
│   ├── grafovi/
│   └── tablice/
├── src/
│   ├── priprema_podataka.py
│   ├── neuronska_mreza.py
│   ├── pso_optimizator.py
│   ├── modeli.py
│   └── grafovi.py
├── stari_zadatak/
├── config.json
├── eksperimenti_pso.py
├── pokreni_projekt.py
└── requirements.txt
```

## Kako je stari PSO zadatak iskorišten

U starom zadatku čestica je predstavljala varijable optimizacijske funkcije. U ovom projektu ista ideja se proširuje: **jedna čestica predstavlja kompletan vektor težina neuronske mreže**. Fitness više nije vrijednost zadane funkcije, nego srednja kvadratna pogreška predikcije cijene.

## Rezultati koji se automatski stvaraju

Nakon pokretanja dobivate:

- `results/tablice/usporedba_modela.csv`
- `results/tablice/pso_konvergencija.csv`
- `results/tablice/predikcije_test_skup.csv`
- `results/grafovi/backpropagation_loss.png`
- `results/grafovi/pso_konvergencija.png`
- `results/grafovi/stvarno_vs_predvideno.png`

## Važna napomena za konačnu predaju

Prije predaje pokrenite završne eksperimente na računalu jednog člana tima uz dogovorene parametre i u dokumentaciju unesite rezultate iz generiranih tablica i grafova. U radu jasno navedite da je PSO glavna metoda, a backpropagation referentna metoda za usporedbu.

## Brzi pokušaj i završni pokušaj

Početna konfiguracija brzo potvrđuje da sve radi:

```powershell
py pokreni_projekt.py
```

Za ozbiljnije završno izvođenje, koje može trajati dulje:

```powershell
py pokreni_projekt.py --config config_final.json
```

## Provjera ispravnosti osnovnih dijelova koda

```powershell
py -m unittest discover -s tests
```
