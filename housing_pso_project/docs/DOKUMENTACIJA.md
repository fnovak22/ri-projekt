# Usporedba treniranja neuronske mreže PSO algoritmom i backpropagation metodom na problemu predikcije cijena nekretnina

## 1. Definicija praktičnog problema i cilj implementacije

Cilj implementacije je izraditi model koji na temelju atributa nekretnine predviđa njezinu prodajnu cijenu. Problem je formuliran kao regresijski problem. Glavni istraživački cilj je provjeriti može li optimizacija rojem čestica (PSO) uspješno pronaći težine neuronske mreže te usporediti dobivene rezultate sa standardnim treniranjem metodom backpropagation uz Adam optimizator.

## 2. Skup podataka

Korišten je transformirani CSV skup `data_housing_cleaned_transformed.csv`. Ciljna varijabla je `price`, a ulazne varijable uključuju broj soba i kupaonica, kvadraturu stambenog i zemljišnog prostora, broj katova, pogled, stanje, godinu izgradnje i renovacije te one-hot kodirane gradove.

Nakon učitavanja potrebno je u tablicu ispod unijeti automatski ispisanu statistiku iz programa:

| Podatak | Vrijednost |
|---|---:|
| Početni broj zapisa | 4 600 |
| Broj atributa bez ciljne varijable | 56 |
| Uklonjeni zapisi cijene <= 0 | 49 |
| Uklonjeni outlieri | 46 |
| Konačni broj zapisa | 4 505 |

## 3. Priprema podataka

CSV je već sadržavao numeričke vrijednosti i one-hot kodiranje grada. U implementaciji se zatim provode sljedeći koraci:

1. uklanjanje zapisa čija je cijena manja ili jednaka nuli;
2. uklanjanje gornjih 1 % ekstremnih cijena radi smanjenja utjecaja izrazitih outliera;
3. podjela podataka na trening, validacijski i testni skup u omjeru 70 : 15 : 15;
4. standardizacija ulaznih atributa i ciljne varijable, pri čemu se skaleri prilagođavaju isključivo na trening skupu.

Testni skup nije korišten za treniranje ni izbor parametara, nego isključivo za konačnu usporedbu modela.

## 4. Implementirana neuronska mreža

Korištena je višeslojna perceptronska neuronska mreža s jednim skrivenim slojem. Ulazni sloj ima onoliko neurona koliko skup ima ulaznih atributa, skriveni sloj koristi šest neurona i ReLU aktivacijsku funkciju, a izlazni sloj ima jedan neuron za predikciju cijene.

Mala arhitektura odabrana je namjerno zato što PSO optimizira svaki parametar mreže kao zasebnu dimenziju čestice. Povećavanje broja skrivenih neurona izravno povećava dimenzionalnost pretraživanja i računalno vrijeme izvođenja.

## 5. Referentno treniranje backpropagation metodom

Referentni model implementiran je klasom `MLPRegressor` iz biblioteke scikit-learn. Koristi istu veličinu skrivenog sloja kao PSO model, ReLU aktivaciju i Adam optimizator. Rezultat ovog modela služi kao baseline za usporedbu s PSO pristupom.

## 6. Treniranje neuronske mreže PSO algoritmom

Kod PSO modela jedna čestica predstavlja cjelokupan skup težina i bias vrijednosti mreže. Pozicija čestice raspakira se u matrice težina između ulaznog i skrivenog sloja te između skrivenog i izlaznog sloja. Za svaku česticu mreža izračunava predikcije nad trening uzorkom, a fitness je srednja kvadratna pogreška na skaliranoj ciljnoj varijabli.

Korišteni parametri zadani su u datoteci `config.json`:

| Parametar | Vrijednost |
|---|---:|
| Broj čestica | 20 |
| Broj iteracija | 80 |
| Početna inercija | 0.9 |
| Završna inercija | 0.4 |
| Kognitivni koeficijent | 1.49445 |
| Socijalni koeficijent | 1.49445 |
| Broj trening zapisa korištenih u usporedbi | 1200 |

Zbog višestrukog izračuna predikcija za svaku česticu u svakoj iteraciji, za osnovnu usporedbu oba modela treniraju se na istom reprezentativnom podskupu trening podataka. Konačna evaluacija obavlja se na jednakom, prethodno neviđenom testnom skupu.

## 7. Plan eksperimenata

Provest će se sljedeći eksperimenti:

| Eksperiment | Što se mijenja | Cilj |
|---|---|---|
| E1 | Backpropagation nasuprot PSO-u | Osnovna usporedba točnosti i vremena |
| E2 | Broj čestica: 10, 20, 30 | Provjera utjecaja veličine roja |
| E3 | Broj iteracija: 40, 80, 100 | Provjera konvergencije |
| E4 | Skriveni neuroni: 4, 6, 8 | Provjera odnosa složenosti mreže i kvalitete |
| E5 | Više pokretanja s različitim seedovima | Procjena stabilnosti rezultata |

## 8. Rezultati početnog izvođenja

U nastavku su rezultati početnog pokretanja s konfiguracijom `config.json`. Prije konačne predaje tablicu treba osvježiti rezultatima završnog izvođenja i više ponavljanja eksperimenta:

| Model | MSE | RMSE | MAE | R² | Vrijeme izvođenja |
|---|---:|---:|---:|---:|---:|
| Backpropagation / Adam | 38 153 396 418,24 | 195 328,94 | 104 367,34 | 0,5172 | 0,29 s |
| PSO + MLP | 39 970 589 303,50 | 199 926,46 | 144 860,69 | 0,4942 | 0,09 s |

U dokumentaciju umetnite grafove:

- `pso_konvergencija.png` — prikazuje smanjenje najbolje PSO pogreške kroz iteracije;
- `backpropagation_loss.png` — prikazuje smanjenje pogreške baseline modela;
- `stvarno_vs_predvideno.png` — prikazuje odstupanja predikcija od stvarnih cijena.

## 9. Rasprava rezultata

U ovom dijelu obrazložite:

- koja metoda ostvaruje nižu pogrešku na testnom skupu;
- koliko je PSO sporiji ili brži od backpropagationa;
- je li povećanje broja čestica ili iteracija donijelo vidljivo poboljšanje;
- kako outlieri i veličina arhitekture utječu na rezultate;
- je li PSO prihvatljiva alternativa za treniranje male neuronske mreže u promatranom regresijskom problemu.

## 10. Zaključak

Zaključak treba temeljiti na provedenim eksperimentima, bez općenitog teorijskog pregleda. Navedite najbolju dobivenu konfiguraciju PSO-a, usporedite je s baselineom te istaknite moguća poboljšanja, primjerice optimizaciju hiperparametara, testiranje alternativnih PSO varijanti ili korištenje drugog skupa podataka.
