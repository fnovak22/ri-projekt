import numpy as np
import pandas as pd
import csv

CONFIG_FILE = "config1a.txt"
DONJE_GRANICE = np.array([-10.0, -7.0])
GORNJE_GRANICE = np.array([5.0, 11.0])
BROJ_DIMENZIJA = 2
V_MAX = 1
V_MIN = -1

def UcitajKonfiguraciju(putanja=CONFIG_FILE):
    cfg = {}
    with open(putanja, "r", encoding="utf-8") as datoteka:
        for linija in datoteka:
            linija = linija.strip()
            if not linija or linija.startswith("#"):
                continue
            if "=" in linija:
                kljuc, crijednost = linija.split("=", 1)
                cfg[kljuc.strip()] = crijednost.strip()

    cfg["w"] = float(cfg["w"])
    cfg["w_smanjenje"] = float(cfg["w_smanjenje"])
    cfg["C"] = float(cfg["C"])
    cfg["broj_cestica"] = int(cfg["broj_cestica"])
    cfg["broj_iteracija"] = int(cfg["broj_iteracija"])
    cfg["refreshing_gap"] = int(cfg["refreshing_gap"])
    return cfg

def FunkcijaIzlaz(u1, u2):
    if u1 > u2:
        return np.sin(u1 + u2 ** 2) * np.exp(np.abs(1 - np.sqrt(u1 ** 2 + 2.5 * u2 ** 2)))
    elif u1 - 2 * u2 <= 0.5:
        return u2 * np.cos(u1 * u2 + 2.3 * u1) - 0.6
    else:
        return 2.0 / (3.0 + np.sin(u2))

def IzracunajVrijednostFunkcijeZaCesticu(x):
    return FunkcijaIzlaz(x[0], x[1])

def IzracunajXb(i, personalBests, personalBestsVrijednost, Pc_vrijednosti):
    xb = np.zeros_like(personalBests[i])

    for d in range(BROJ_DIMENZIJA):
        r = np.random.rand()

        if r < Pc_vrijednosti[i]:
            xb[d] = personalBests[i][d]

        else:
            ostaleCestice = []
            for pbs in range(len(personalBests)):
                if pbs != i:
                    ostaleCestice.append(pbs)

            j = np.random.choice(ostaleCestice)
            ostaleCestice.remove(j)
            k = np.random.choice(ostaleCestice)

            if personalBestsVrijednost[j] < personalBestsVrijednost[k]:
                xb[d] = personalBests[j][d]
            else:
                xb[d] = personalBests[k][d]
    return xb

def main():
    cfg = UcitajKonfiguraciju()

    w = float(cfg["w"])
    wSmanjenje = float(cfg["w_smanjenje"])
    c = float(cfg["C"])
    brojCestica = int(cfg["broj_cestica"])
    brojIteracija = int(cfg["broj_iteracija"])
    refreshing_gap = int(cfg["refreshing_gap"])

    redniBrojevi = np.arange(1, brojCestica + 1)
    Pc_vrijednosti = 0.05 + 0.45 * (
        (np.exp(10 * (redniBrojevi - 1) / (brojCestica - 1)) - 1) / (np.exp(10) - 1)
    )

    x = np.zeros((brojCestica, BROJ_DIMENZIJA))
    for i in range(brojCestica):
        for d in range(BROJ_DIMENZIJA):
            x[i][d] = np.random.uniform(DONJE_GRANICE[d], GORNJE_GRANICE[d])

    v = np.zeros((brojCestica, BROJ_DIMENZIJA))
    for i in range(brojCestica):
        for d in range(BROJ_DIMENZIJA):
            v[i][d] = np.random.uniform(V_MIN, V_MAX)

    personalBests = x.copy()
    personalBestsVrijednost = np.zeros(brojCestica)
    for i in range(brojCestica):
        personalBestsVrijednost[i] = IzracunajVrijednostFunkcijeZaCesticu(x[i])

    xb = np.zeros_like(x)
    for i in range(brojCestica):
        xb[i] = IzracunajXb(i, personalBests, personalBestsVrijednost, Pc_vrijednosti)

    brojaciOsvjezenjaXb = []
    for i in range(brojCestica):
        brojaciOsvjezenjaXb.append(refreshing_gap)

    indeksNajbolje = np.argmin(personalBestsVrijednost)
    globalnoNajboljePozicije = personalBests[indeksNajbolje].copy()
    globalnoNajboljaVrijednost = personalBestsVrijednost[indeksNajbolje]

    results = []

    for iteracija in range(brojIteracija):
        for i in range(brojCestica):
            if brojaciOsvjezenjaXb[i] <= 0:
                xb[i] = IzracunajXb(i, personalBests, personalBestsVrijednost, Pc_vrijednosti)
                brojaciOsvjezenjaXb[i] = refreshing_gap

            r = np.random.rand(BROJ_DIMENZIJA)
            for d in range(BROJ_DIMENZIJA):
                v[i][d] = w * v[i][d] + c * r[d] * (xb[i][d] - x[i][d])
                x[i][d] = x[i][d] + v[i][d]

            for d in range(BROJ_DIMENZIJA):
                if x[i][d] < DONJE_GRANICE[d]:
                    x[i][d] = DONJE_GRANICE[d]
                elif x[i][d] > GORNJE_GRANICE[d]:
                    x[i][d] = GORNJE_GRANICE[d]

            novaVrijednost = IzracunajVrijednostFunkcijeZaCesticu(x[i])

            if novaVrijednost < personalBestsVrijednost[i]:
                personalBests[i] = x[i].copy()
                personalBestsVrijednost[i] = novaVrijednost
            else:
                brojaciOsvjezenjaXb[i] = brojaciOsvjezenjaXb[i] - 1

        najboljiIndeksIteracija = np.argmin(personalBestsVrijednost)
        najboljaCesticaIteracija = personalBests[najboljiIndeksIteracija].copy()
        najboljaVrijednostIteracija = personalBestsVrijednost[najboljiIndeksIteracija]

        if najboljaVrijednostIteracija < globalnoNajboljaVrijednost:
            globalnoNajboljePozicije = najboljaCesticaIteracija.copy()
            globalnoNajboljaVrijednost = najboljaVrijednostIteracija

        results.append(najboljaVrijednostIteracija)
        w = w - wSmanjenje

    df = pd.DataFrame(results)
    df.to_csv("rezultat1a.csv", index=False, header=False, sep=';', decimal=',', quoting=csv.QUOTE_NONE, escapechar='\\')

    print("Najbolje pronedeno rjesenje:")
    print(f"Vrijednost funkcije  = {globalnoNajboljaVrijednost:.10f}")
    print(f"u1 = {globalnoNajboljePozicije[0]:.10f}")
    print(f"u2 = {globalnoNajboljePozicije[1]:.10f}")
    
        
if __name__ == "__main__":
    main()