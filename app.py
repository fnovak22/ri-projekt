import tkinter as tk
import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from predict import predvidi, napravi_ulazni_vektor


# -----------------------------
# Putanje projekta
# -----------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"

TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "val.csv"
TEST_FILE = DATA_DIR / "test.csv"

MODEL_PSO_FILE = MODEL_DIR / "model_pso.npz"
MODEL_BACKPROP_FILE = MODEL_DIR / "model_backpropagation.npz"

SCRIPT_PREPARE_DATA = PROJECT_DIR / "prepare_data.py"
SCRIPT_TRAIN_PSO = PROJECT_DIR / "train_pso.py"
SCRIPT_EVALUATE_TEST = PROJECT_DIR / "evaluate_test.py"
PERSON_FILE = PROJECT_DIR / "prediction-data" / "person.json"


# -----------------------------
# Boje sučelja
# -----------------------------

BOJA_POZADINE = "#f3f6fb"
BOJA_KARTICE = "#ffffff"
BOJA_TEKSTA = "#1f2937"
BOJA_MUTNOG_TEKSTA = "#6b7280"

BOJA_PSO_GUMB = "#2563eb"
BOJA_PSO_GUMB_HOVER = "#1d4ed8"

BOJA_OKOLINA_GUMB = "#7c3aed"
BOJA_OKOLINA_GUMB_HOVER = "#6d28d9"

BOJA_DISABLED = "#9ca3af"

BOJA_REZULTAT = "#ecfdf5"
BOJA_REZULTAT_RUB = "#10b981"

BOJA_GRESKA = "#fef2f2"
BOJA_GRESKA_RUB = "#ef4444"

BOJA_INFO = "#eff6ff"
BOJA_INFO_RUB = "#3b82f6"


# -----------------------------
# Pomoćne funkcije za sučelje
# -----------------------------

def postavi_ikonu(root):
    ico_path = Path("assets") / "app.ico"
    png_path = Path("assets") / "app.png"

    try:
        if ico_path.exists():
            root.iconbitmap(str(ico_path))
        elif png_path.exists():
            icon_image = tk.PhotoImage(file=str(png_path))
            root.iconphoto(True, icon_image)

            # Zadržavamo referencu da Python ne obriše sliku iz memorije.
            root.icon_image = icon_image
    except Exception:
        # Ikona nije kritična za rad aplikacije.
        pass


def napravi_labelu(parent, tekst):
    return tk.Label(
        parent,
        text=tekst,
        bg=BOJA_KARTICE,
        fg=BOJA_TEKSTA,
        font=("Segoe UI", 10, "bold"),
        anchor="w"
    )


def napravi_unos(parent):
    return tk.Entry(
        parent,
        font=("Segoe UI", 11),
        bg="#f9fafb",
        fg=BOJA_TEKSTA,
        relief="flat",
        highlightthickness=1,
        highlightbackground="#d1d5db",
        highlightcolor="#2563eb",
        insertbackground=BOJA_TEKSTA
    )


def napravi_polje(parent, label_text):
    field_frame = tk.Frame(parent, bg=BOJA_KARTICE)
    field_frame.pack(fill="x", pady=(0, 14))

    label = napravi_labelu(field_frame, label_text)
    label.pack(fill="x", pady=(0, 5))

    entry = napravi_unos(field_frame)
    entry.pack(fill="x", ipady=8)

    return entry


def ocisti_rezultat():
    rezultat_frame.config(bg=BOJA_KARTICE, highlightbackground=BOJA_KARTICE)

    rezultat_naslov.config(
        text="Rezultat",
        bg=BOJA_KARTICE,
        fg=BOJA_MUTNOG_TEKSTA
    )

    rezultat_tekst.config(
        text="Unesi podatke osobe i odaberi model za predikciju.",
        bg=BOJA_KARTICE,
        fg=BOJA_MUTNOG_TEKSTA
    )


def prikazi_rezultat(tekst):
    rezultat_frame.config(bg=BOJA_REZULTAT, highlightbackground=BOJA_REZULTAT_RUB)

    rezultat_naslov.config(
        text="Uspješno",
        bg=BOJA_REZULTAT,
        fg="#047857"
    )

    rezultat_tekst.config(
        text=tekst,
        bg=BOJA_REZULTAT,
        fg="#065f46"
    )

    root.after(100, osvjezi_scroll_region)


def prikazi_gresku(tekst):
    rezultat_frame.config(bg=BOJA_GRESKA, highlightbackground=BOJA_GRESKA_RUB)

    rezultat_naslov.config(
        text="Greška",
        bg=BOJA_GRESKA,
        fg="#b91c1c"
    )

    rezultat_tekst.config(
        text=tekst,
        bg=BOJA_GRESKA,
        fg="#7f1d1d"
    )

    root.after(100, osvjezi_scroll_region)


def prikazi_info(tekst):
    rezultat_frame.config(bg=BOJA_INFO, highlightbackground=BOJA_INFO_RUB)

    rezultat_naslov.config(
        text="Priprema okoline",
        bg=BOJA_INFO,
        fg="#1d4ed8"
    )

    rezultat_tekst.config(
        text=tekst,
        bg=BOJA_INFO,
        fg="#1e3a8a"
    )

    root.after(100, osvjezi_scroll_region)

def azuriraj_status_pripreme(tekst):
    rezultat_tekst.config(
        text=tekst,
        bg=BOJA_INFO,
        fg="#1e3a8a"
    )

    root.after(100, osvjezi_scroll_region)


def hover_gumb(gumb, normalna_boja, hover_boja):
    def on_enter(event):
        if str(gumb["state"]) == "normal":
            gumb.config(bg=hover_boja)

    def on_leave(event):
        if str(gumb["state"]) == "normal":
            gumb.config(bg=normalna_boja)

    gumb.bind("<Enter>", on_enter)
    gumb.bind("<Leave>", on_leave)


# -----------------------------
# Logika predikcije
# -----------------------------

def procitaj_podatke_iz_sucelja():
    if spol_var.get() == "Žensko":
        gender = 1
    elif spol_var.get() == "Muško":
        gender = 0
    else:
        raise ValueError("Odaberi spol osobe.")

    try:
        osoba = {
            "age": float(age_entry.get()),
            "gender": gender,
            "height_cm": float(height_entry.get()),
            "weight_kg": float(weight_entry.get()),
            "sit_and_bend_forward_cm": float(bend_entry.get()),
            "sit_ups_counts": float(situps_entry.get()),
        }
    except ValueError:
        raise ValueError("Sva numerička polja moraju biti ispunjena brojevima.")

    if osoba["age"] <= 0:
        raise ValueError("Dob mora biti veća od 0.")

    if osoba["height_cm"] <= 0:
        raise ValueError("Visina mora biti veća od 0.")

    if osoba["weight_kg"] <= 0:
        raise ValueError("Masa mora biti veća od 0.")

    return osoba


def pokreni_predikciju(model_file, naziv_modela):
    try:
        if not model_file.exists():
            raise FileNotFoundError(
                f"Nedostaje datoteka modela:\n{model_file}\n\n"
                "Prvo istreniraj model ili klikni gumb 'Pripremi okolinu'."
            )

        osoba = procitaj_podatke_iz_sucelja()

        model = np.load(model_file, allow_pickle=True)
        X = napravi_ulazni_vektor(osoba)

        predikcija = predvidi(model, X)

        prikazi_rezultat(
            f"{naziv_modela}\n\n"
            f"Predviđeni skok u dalj iz mjesta: {predikcija:.2f} cm"
        )

    except Exception as e:
        prikazi_gresku(str(e))


def resetiraj_polja():
    age_entry.delete(0, tk.END)
    height_entry.delete(0, tk.END)
    weight_entry.delete(0, tk.END)
    bend_entry.delete(0, tk.END)
    situps_entry.delete(0, tk.END)

    spol_var.set("Žensko")

    ocisti_rezultat()

def postavi_vrijednost(entry, vrijednost):
    entry.delete(0, tk.END)

    if vrijednost is not None:
        entry.insert(0, str(vrijednost))


def postavi_spol(vrijednost):
    """
    Podržava više mogućih formata:
    - 1 ili "1" ili "F" ili "female" -> Žensko
    - 0 ili "0" ili "M" ili "male" -> Muško
    """

    vrijednost = str(vrijednost).strip().lower()

    if vrijednost in ["1", "f", "female", "žensko", "zensko"]:
        spol_var.set("Žensko")
    elif vrijednost in ["0", "m", "male", "muško", "musko"]:
        spol_var.set("Muško")
    else:
        spol_var.set("Žensko")


def ucitaj_person_json_ako_postoji():
    if not PERSON_FILE.exists():
        return

    try:
        with open(PERSON_FILE, "r", encoding="utf-8") as file:
            osoba = json.load(file)

        postavi_vrijednost(age_entry, osoba.get("age"))
        postavi_spol(osoba.get("gender", "Žensko"))
        postavi_vrijednost(height_entry, osoba.get("height_cm"))
        postavi_vrijednost(weight_entry, osoba.get("weight_kg"))
        postavi_vrijednost(bend_entry, osoba.get("sit_and_bend_forward_cm"))
        postavi_vrijednost(situps_entry, osoba.get("sit_ups_counts"))

        prikazi_info(
            "Učitani su početni podaci iz datoteke:\n"
            "prediction-data/person.json"
        )

    except Exception as e:
        prikazi_gresku(
            "Datoteka prediction-data/person.json postoji, "
            "ali je nije moguće učitati.\n\n"
            f"Detalji greške:\n{e}"
        )


# -----------------------------
# Logika pripreme okoline
# -----------------------------

def postoje_pripremljeni_podaci():
    return TRAIN_FILE.exists() and VAL_FILE.exists() and TEST_FILE.exists()


def skrati_izlaz(tekst, max_broj_redaka=45):
    redci = tekst.splitlines()

    if len(redci) <= max_broj_redaka:
        return tekst

    pocetak = redci[:10]
    kraj = redci[-25:]

    return "\n".join(
        pocetak
        + [
            "",
            f"... izlaz je skraćen, ukupno redaka: {len(redci)} ...",
            "",
        ]
        + kraj
    )


def formatiraj_izlaz_skripte(naziv_skripte, rezultat):
    return f"✓ {naziv_skripte} je uspješno završila."


def pokreni_skriptu(script_path):
    if not script_path.exists():
        raise FileNotFoundError(f"Nedostaje skripta: {script_path.name}")

    rezultat = subprocess.run(
        [sys.executable, script_path.name],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        errors="replace"
    )

    if rezultat.returncode != 0:
        izlaz = ""

        if rezultat.stdout.strip():
            izlaz += rezultat.stdout.strip()

        if rezultat.stderr.strip():
            if izlaz:
                izlaz += "\n\n"
            izlaz += rezultat.stderr.strip()

        if not izlaz:
            izlaz = "Skripta nije vratila dodatni opis greške."

        raise RuntimeError(
            f"Skripta {script_path.name} nije uspješno završila.\n\n{izlaz}"
        )

    return rezultat


def postavi_gumbe_zauzeto(zauzeto):
    if zauzeto:
        okolina_button.config(state="disabled", cursor="")
        pso_button.config(state="disabled", cursor="")
        reset_button.config(state="disabled", cursor="")
    else:
        okolina_button.config(state="normal", cursor="hand2")
        pso_button.config(state="normal", cursor="hand2")
        reset_button.config(state="normal", cursor="hand2")

    # Backpropagation ostaje onemogućen dok ne implementiraš i ne istreniraš taj model.
    backprop_button.config(state="disabled")


def priprema_okoline_worker():
    log = []

    try:
        root.after(
            0,
            lambda: azuriraj_status_pripreme(
                "Priprema okoline je pokrenuta...\n\n"
                "[1/3] Provjera podataka..."
            )
        )

        if postoje_pripremljeni_podaci():
            log.append("✓ Podaci su već pripremljeni.")
        else:
            pokreni_skriptu(SCRIPT_PREPARE_DATA)
            log.append("✓ Podaci su pripremljeni.")

        root.after(
            0,
            lambda: azuriraj_status_pripreme(
                "Priprema okoline je pokrenuta...\n\n"
                "[1/3] Podaci su spremni.\n"
                "[2/3] Provjera PSO modela..."
            )
        )

        if MODEL_PSO_FILE.exists():
            log.append("✓ PSO model već postoji.")
        else:
            pokreni_skriptu(SCRIPT_TRAIN_PSO)
            log.append("✓ PSO model je istreniran.")

        root.after(
            0,
            lambda: azuriraj_status_pripreme(
                "Priprema okoline je pokrenuta...\n\n"
                "[1/3] Podaci su spremni.\n"
                "[2/3] PSO model je spreman.\n"
                "[3/3] Evaluacija modela..."
            )
        )

        if TEST_FILE.exists() and MODEL_PSO_FILE.exists():
            pokreni_skriptu(SCRIPT_EVALUATE_TEST)
            log.append("✓ Evaluacija modela je završena.")
        else:
            log.append("⚠ Evaluacija je preskočena jer nedostaje test skup ili PSO model.")

        zavrsni_tekst = "\n".join(log)

        root.after(
            0,
            lambda: prikazi_rezultat(
                "Okolina je spremna.\n\n" + zavrsni_tekst
            )
        )

    except Exception as e:
        zavrsni_tekst = "\n".join(log)

        if zavrsni_tekst:
            zavrsni_tekst += "\n\n"

        zavrsni_tekst += str(e)

        root.after(0, lambda: prikazi_gresku(zavrsni_tekst))

    finally:
        root.after(0, lambda: postavi_gumbe_zauzeto(False))

def pokreni_pripremu_okoline():
    postavi_gumbe_zauzeto(True)

    prikazi_info(
        "Priprema okoline je pokrenuta...\n\n"
        "[1/3] Provjera / priprema podataka..."
    )

    thread = threading.Thread(target=priprema_okoline_worker, daemon=True)
    thread.start()

# -----------------------------
# Scroll funkcije
# -----------------------------

def scroll_misem(event):
    if event.num == 4:
        canvas.yview_scroll(-1, "units")
    elif event.num == 5:
        canvas.yview_scroll(1, "units")
    elif event.delta > 0:
        canvas.yview_scroll(-1, "units")
    else:
        canvas.yview_scroll(1, "units")


def osvjezi_scroll_region(event=None):
    canvas.configure(scrollregion=canvas.bbox("all"))


def prilagodi_sirinu(event):
    canvas.itemconfig(canvas_window, width=event.width)

    dostupna_sirina = max(event.width - 90, 240)

    if "subtitle_label" in globals():
        subtitle_label.config(wraplength=dostupna_sirina)

    if "rezultat_tekst" in globals():
        rezultat_tekst.config(wraplength=dostupna_sirina)


# -----------------------------
# Glavni prozor
# -----------------------------

root = tk.Tk()
root.title("Predikcija skoka u dalj")

root.geometry("620x720")
root.minsize(390, 420)

root.configure(bg=BOJA_POZADINE)

postavi_ikonu(root)


# -----------------------------
# Scrollable glavni layout
# -----------------------------

main_container = tk.Frame(root, bg=BOJA_POZADINE)
main_container.pack(fill="both", expand=True)

canvas = tk.Canvas(
    main_container,
    bg=BOJA_POZADINE,
    highlightthickness=0
)
canvas.pack(side="left", fill="both", expand=True)

scrollbar = tk.Scrollbar(
    main_container,
    orient="vertical",
    command=canvas.yview
)
scrollbar.pack(side="right", fill="y")

canvas.configure(yscrollcommand=scrollbar.set)

scrollable_frame = tk.Frame(canvas, bg=BOJA_POZADINE)

canvas_window = canvas.create_window(
    (0, 0),
    window=scrollable_frame,
    anchor="nw"
)

scrollable_frame.bind("<Configure>", osvjezi_scroll_region)
canvas.bind("<Configure>", prilagodi_sirinu)

canvas.bind_all("<MouseWheel>", scroll_misem)
canvas.bind_all("<Button-4>", scroll_misem)
canvas.bind_all("<Button-5>", scroll_misem)


# -----------------------------
# Sadržaj
# -----------------------------

outer_frame = tk.Frame(scrollable_frame, bg=BOJA_POZADINE)
outer_frame.pack(fill="both", expand=True, padx=22, pady=22)

header_frame = tk.Frame(outer_frame, bg=BOJA_POZADINE)
header_frame.pack(fill="x", pady=(0, 16))

title_label = tk.Label(
    header_frame,
    text="Predikcija skoka u dalj",
    bg=BOJA_POZADINE,
    fg=BOJA_TEKSTA,
    font=("Segoe UI", 22, "bold"),
    anchor="w"
)
title_label.pack(fill="x")

subtitle_label = tk.Label(
    header_frame,
    text="Unesi podatke osobe i odaberi istrenirani model neuronske mreže.",
    bg=BOJA_POZADINE,
    fg=BOJA_MUTNOG_TEKSTA,
    font=("Segoe UI", 11),
    anchor="w",
    justify="left"
)
subtitle_label.pack(fill="x", pady=(6, 0))


# -----------------------------
# Kartica
# -----------------------------

card = tk.Frame(
    outer_frame,
    bg=BOJA_KARTICE,
    highlightthickness=1,
    highlightbackground="#e5e7eb"
)
card.pack(fill="x", expand=True)

form_frame = tk.Frame(card, bg=BOJA_KARTICE)
form_frame.pack(fill="x", padx=24, pady=24)

section_label = tk.Label(
    form_frame,
    text="Podaci osobe",
    bg=BOJA_KARTICE,
    fg=BOJA_TEKSTA,
    font=("Segoe UI", 15, "bold"),
    anchor="w"
)
section_label.pack(fill="x", pady=(0, 18))


# -----------------------------
# Forma
# -----------------------------

age_entry = napravi_polje(form_frame, "Dob")

spol_field_frame = tk.Frame(form_frame, bg=BOJA_KARTICE)
spol_field_frame.pack(fill="x", pady=(0, 14))

napravi_labelu(spol_field_frame, "Spol").pack(fill="x", pady=(0, 5))

spol_var = tk.StringVar(value="Žensko")

spol_frame = tk.Frame(spol_field_frame, bg=BOJA_KARTICE)
spol_frame.pack(fill="x")

zensko_radio = tk.Radiobutton(
    spol_frame,
    text="Žensko",
    variable=spol_var,
    value="Žensko",
    bg=BOJA_KARTICE,
    fg=BOJA_TEKSTA,
    activebackground=BOJA_KARTICE,
    activeforeground=BOJA_TEKSTA,
    selectcolor=BOJA_KARTICE,
    font=("Segoe UI", 10)
)
zensko_radio.pack(side="left", padx=(0, 18))

musko_radio = tk.Radiobutton(
    spol_frame,
    text="Muško",
    variable=spol_var,
    value="Muško",
    bg=BOJA_KARTICE,
    fg=BOJA_TEKSTA,
    activebackground=BOJA_KARTICE,
    activeforeground=BOJA_TEKSTA,
    selectcolor=BOJA_KARTICE,
    font=("Segoe UI", 10)
)
musko_radio.pack(side="left")

height_entry = napravi_polje(form_frame, "Visina (cm)")
weight_entry = napravi_polje(form_frame, "Masa (kg)")
bend_entry = napravi_polje(form_frame, "Pretklon u sjedu (cm)")
situps_entry = napravi_polje(form_frame, "Broj trbušnjaka")


# -----------------------------
# Gumbi
# -----------------------------

button_frame = tk.Frame(card, bg=BOJA_KARTICE)
button_frame.pack(fill="x", padx=24, pady=(0, 22))

okolina_button = tk.Button(
    button_frame,
    text="Pripremi okolinu",
    command=pokreni_pripremu_okoline,
    bg=BOJA_OKOLINA_GUMB,
    fg="white",
    activebackground=BOJA_OKOLINA_GUMB_HOVER,
    activeforeground="white",
    disabledforeground="white",
    relief="flat",
    bd=0,
    font=("Segoe UI", 11, "bold"),
    cursor="hand2"
)
okolina_button.pack(fill="x", ipady=11, pady=(0, 10))
hover_gumb(okolina_button, BOJA_OKOLINA_GUMB, BOJA_OKOLINA_GUMB_HOVER)

pso_button = tk.Button(
    button_frame,
    text="Predikcija PSO modelom",
    command=lambda: pokreni_predikciju(MODEL_PSO_FILE, "PSO model"),
    bg=BOJA_PSO_GUMB,
    fg="white",
    activebackground=BOJA_PSO_GUMB_HOVER,
    activeforeground="white",
    disabledforeground="white",
    relief="flat",
    bd=0,
    font=("Segoe UI", 11, "bold"),
    cursor="hand2"
)
pso_button.pack(fill="x", ipady=11, pady=(0, 10))
hover_gumb(pso_button, BOJA_PSO_GUMB, BOJA_PSO_GUMB_HOVER)

backprop_button = tk.Button(
    button_frame,
    text="Predikcija Backpropagation modelom",
    command=lambda: pokreni_predikciju(MODEL_BACKPROP_FILE, "Backpropagation model"),
    bg=BOJA_DISABLED,
    fg="white",
    disabledforeground="white",
    relief="flat",
    bd=0,
    font=("Segoe UI", 11, "bold"),
    state="disabled"
)
backprop_button.pack(fill="x", ipady=11, pady=(0, 10))

reset_button = tk.Button(
    button_frame,
    text="Očisti unos",
    command=resetiraj_polja,
    bg="#e5e7eb",
    fg=BOJA_TEKSTA,
    activebackground="#d1d5db",
    activeforeground=BOJA_TEKSTA,
    disabledforeground=BOJA_MUTNOG_TEKSTA,
    relief="flat",
    bd=0,
    font=("Segoe UI", 10, "bold"),
    cursor="hand2"
)
reset_button.pack(fill="x", ipady=9)


# -----------------------------
# Rezultat
# -----------------------------

rezultat_frame = tk.Frame(
    card,
    bg=BOJA_KARTICE,
    highlightthickness=1,
    highlightbackground=BOJA_KARTICE
)
rezultat_frame.pack(fill="x", padx=24, pady=(0, 24))

rezultat_naslov = tk.Label(
    rezultat_frame,
    text="Rezultat",
    bg=BOJA_KARTICE,
    fg=BOJA_MUTNOG_TEKSTA,
    font=("Segoe UI", 11, "bold"),
    anchor="w"
)
rezultat_naslov.pack(fill="x", padx=16, pady=(14, 4))

rezultat_tekst = tk.Label(
    rezultat_frame,
    text="Unesi podatke osobe i odaberi model za predikciju.",
    bg=BOJA_KARTICE,
    fg=BOJA_MUTNOG_TEKSTA,
    font=("Segoe UI", 12),
    anchor="w",
    justify="left"
)
rezultat_tekst.pack(fill="x", padx=16, pady=(0, 16))


# -----------------------------
# Pokretanje aplikacije
# -----------------------------

ucitaj_person_json_ako_postoji()

root.after(100, osvjezi_scroll_region)

root.mainloop()