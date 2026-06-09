# Upute za pokretanje aplikacije

## 1. Instalacija Pythona 3.14.3

Prvo instalirati Python verziju **3.14.3** sa službene stranice:

https://www.python.org/downloads/

Tijekom instalacije obavezno označiti opciju:

```text
Add Python to PATH
```

Nakon instalacije provjeriti verziju u terminalu:

```bash
python --version
```


---

## 2. Otvaranje foldera projekta

U terminalu se prebaciti u glavni folder projekta.

Primjer:

```bash
cd putanja/do/projekta
```

---

## 3. Kreiranje virtualnog okruženja

U glavnom folderu projekta pokrenuti:

```bash
python -m venv venv
```

---

## 4. Aktivacija virtualnog okruženja

Na Windowsu pokrenuti:

```bash
.venv\Scripts\activate
```

Na Linuxu ili macOS-u pokrenuti:

```bash
source .venv/bin/activate
```

Nakon aktivacije u terminalu bi se trebalo pojaviti:

```text
(.venv)
```

To znači da je virtualno okruženje aktivno.

---

## 5. Instalacija potrebnih biblioteka

Dok je virtualno okruženje aktivirano, pokrenuti:

```bash
pip install -r requirements.txt
```

Ova naredba instalira sve biblioteke koje su potrebne za pokretanje aplikacije.

---

## 6. Pokretanje aplikacije

Nakon instalacije biblioteka u virtualno okruženje, aplikacija se pokreće naredbom:

```bash
python app.py
```

---

## 7. Ponovno pokretanje aplikacije kasnije

Kod sljedećeg pokretanja nije potrebno ponovno instalirati biblioteke.

Potrebno je samo otvoriti glavni folder projekta, aktivirati virtualno okruženje:

```bash
.venv\Scripts\activate
```

i pokrenuti aplikaciju:

```bash
python app.py
```