import tkinter as tk
from tkinter import ttk
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
import ctypes

import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from predict import predvidi, napravi_ulazni_vektor
from predict_backpropagation import predvidi as predvidi_backpropagation

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"

TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "val.csv"
TEST_FILE = DATA_DIR / "test.csv"

MODEL_PSO_FILE = MODEL_DIR / "model_pso.npz"
MODEL_BACKPROP_FILE = MODEL_DIR / "model_backpropagation.joblib"

SCRIPT_PREPARE_DATA = PROJECT_DIR / "prepare_data.py"

SCRIPT_TRAIN_PSO = PROJECT_DIR / "train_pso.py"
SCRIPT_EVALUATE_TEST = PROJECT_DIR / "evaluate_test.py"

SCRIPT_TRAIN_BACKPROP = PROJECT_DIR / "train_backpropagation.py"
SCRIPT_EVALUATE_BACKPROP = PROJECT_DIR / "evaluate_backpropagation_test.py"

PERSON_FILE = PROJECT_DIR / "prediction-data" / "person.json"

ULAZNI_STUPCI = [
    "age",
    "gender",
    "height_cm",
    "weight_kg",
    "sit and bend forward_cm",
    "sit-ups counts",
]

IZLAZNI_STUPAC = "broad jump_cm"

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

def postavi_ikonu(root):
    ico_path = PROJECT_DIR / "assets" / "app.ico"
    png_path = PROJECT_DIR / "assets" / "app.png"

    try:
        if ico_path.exists():
            root.iconbitmap(str(ico_path))
        elif png_path.exists():
            icon_image = tk.PhotoImage(file=str(png_path))
            root.iconphoto(True, icon_image)
            root.icon_image = icon_image
    except Exception:
        pass


def postavi_windows_app_id():
    try:
        app_id = "ri.projekt.predikcija.skoka"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
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


def prikazi_info_pripreme(tekst):
    priprema_status_frame.config(bg=BOJA_INFO, highlightbackground=BOJA_INFO_RUB)

    priprema_status_naslov.config(
        text="Priprema okoline",
        bg=BOJA_INFO,
        fg="#1d4ed8"
    )

    priprema_status_tekst.config(
        text=tekst,
        bg=BOJA_INFO,
        fg="#1e3a8a"
    )

    root.after(100, osvjezi_scroll_region)


def azuriraj_status_pripreme(tekst):
    priprema_status_tekst.config(
        text=tekst,
        bg=BOJA_INFO,
        fg="#1e3a8a"
    )

    root.after(100, osvjezi_scroll_region)


def prikazi_uspjeh_pripreme(tekst):
    priprema_status_frame.config(bg=BOJA_REZULTAT, highlightbackground=BOJA_REZULTAT_RUB)

    priprema_status_naslov.config(
        text="Uspješno",
        bg=BOJA_REZULTAT,
        fg="#047857"
    )

    priprema_status_tekst.config(
        text=tekst,
        bg=BOJA_REZULTAT,
        fg="#065f46"
    )

    root.after(100, osvjezi_scroll_region)


def prikazi_gresku_pripreme(tekst):
    priprema_status_frame.config(bg=BOJA_GRESKA, highlightbackground=BOJA_GRESKA_RUB)

    priprema_status_naslov.config(
        text="Greška",
        bg=BOJA_GRESKA,
        fg="#b91c1c"
    )

    priprema_status_tekst.config(
        text=tekst,
        bg=BOJA_GRESKA,
        fg="#7f1d1d"
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


def pokreni_predikciju_backpropagation():
    try:
        if not MODEL_BACKPROP_FILE.exists():
            raise FileNotFoundError(
                f"Nedostaje datoteka modela:\n{MODEL_BACKPROP_FILE}\n\n"
                "Prvo istreniraj backpropagation model ili klikni gumb 'Pripremi okolinu'."
            )

        osoba = procitaj_podatke_iz_sucelja()

        spremljeno = joblib.load(MODEL_BACKPROP_FILE)
        X = napravi_ulazni_vektor(osoba)

        predikcija = predvidi_backpropagation(spremljeno, X)

        prikazi_rezultat(
            "Backpropagation model\n\n"
            f"Predviđeni skok u dalj iz mjesta: {predikcija:.2f} cm"
        )

    except Exception as e:
        prikazi_gresku(str(e))

# GRAF: Usporedba predikcija modela za unesenu osobu
def prikazi_graf_usporedbe_predikcija(pso_predikcija, backprop_predikcija):
    for widget in graf_predikcija_frame.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    modeli = ["PSO", "Backpropagation"]
    vrijednosti = [pso_predikcija, backprop_predikcija]

    ax.bar(modeli, vrijednosti)
    ax.set_title("Usporedba predikcija modela")
    ax.set_ylabel("Predviđeni skok (cm)")
    ax.set_xlabel("Model")

    for i, vrijednost in enumerate(vrijednosti):
        ax.text(i, vrijednost, f"{vrijednost:.2f} cm", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()

    canvas_graf = FigureCanvasTkAgg(fig, master=graf_predikcija_frame)
    canvas_graf.draw()
    canvas_graf.get_tk_widget().pack(fill="both", expand=True)

    plt.close(fig)


def prikazi_usporedbu_predikcija(pso_predikcija, backprop_predikcija):
    pso_vrijednost_label.config(
        text=f"{pso_predikcija:.2f} cm",
        fg="#1d4ed8"
    )

    backprop_vrijednost_label.config(
        text=f"{backprop_predikcija:.2f} cm",
        fg="#047857"
    )

    razlika = abs(pso_predikcija - backprop_predikcija)

    rezultat_frame.config(
        bg=BOJA_REZULTAT,
        highlightbackground=BOJA_REZULTAT_RUB
    )

    rezultat_naslov.config(
        text="Usporedba modela",
        bg=BOJA_REZULTAT,
        fg="#047857"
    )

    rezultat_tekst.config(
        text=(
            f"PSO model: {pso_predikcija:.2f} cm\n"
            f"Backpropagation model: {backprop_predikcija:.2f} cm\n"
            f"Razlika između modela: {razlika:.2f} cm"
        ),
        bg=BOJA_REZULTAT,
        fg="#065f46"
    )

    prikazi_graf_usporedbe_predikcija(pso_predikcija, backprop_predikcija)

    root.after(100, osvjezi_scroll_region)


def pokreni_usporedbu_modela():
    try:
        if not MODEL_PSO_FILE.exists():
            raise FileNotFoundError(
                f"Nedostaje datoteka modela:\n{MODEL_PSO_FILE}\n\n"
                "Prvo klikni gumb 'Pripremi okolinu'."
            )

        if not MODEL_BACKPROP_FILE.exists():
            raise FileNotFoundError(
                f"Nedostaje datoteka modela:\n{MODEL_BACKPROP_FILE}\n\n"
                "Prvo klikni gumb 'Pripremi okolinu'."
            )

        osoba = procitaj_podatke_iz_sucelja()
        X = napravi_ulazni_vektor(osoba)

        pso_model = np.load(MODEL_PSO_FILE, allow_pickle=True)
        pso_predikcija = predvidi(pso_model, X)

        backprop_model = joblib.load(MODEL_BACKPROP_FILE)
        backprop_predikcija = predvidi_backpropagation(backprop_model, X)

        prikazi_usporedbu_predikcija(pso_predikcija, backprop_predikcija)

    except Exception as e:
        prikazi_gresku(str(e))


def resetiraj_polja():
    age_entry.delete(0, tk.END)
    height_entry.delete(0, tk.END)
    weight_entry.delete(0, tk.END)
    bend_entry.delete(0, tk.END)
    situps_entry.delete(0, tk.END)

    spol_var.set("Žensko")

    pso_vrijednost_label.config(text="-")
    backprop_vrijednost_label.config(text="-")

    for widget in graf_predikcija_frame.winfo_children():
        widget.destroy()

    ocisti_rezultat()


def postavi_vrijednost(entry, vrijednost):
    entry.delete(0, tk.END)

    if vrijednost is not None:
        entry.insert(0, str(vrijednost))


def postavi_spol(vrijednost):
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


def ocisti_frame(frame):
    for widget in frame.winfo_children():
        widget.destroy()


def prikazi_gresku_evaluacije(tekst):
    evaluacija_status_frame.config(bg=BOJA_GRESKA, highlightbackground=BOJA_GRESKA_RUB)
    evaluacija_status_naslov.config(
        text="Greška",
        bg=BOJA_GRESKA,
        fg="#b91c1c"
    )
    evaluacija_status_tekst.config(
        text=tekst,
        bg=BOJA_GRESKA,
        fg="#7f1d1d"
    )
    root.after(100, osvjezi_scroll_region)


def prikazi_info_evaluacije(tekst):
    evaluacija_status_frame.config(bg=BOJA_INFO, highlightbackground=BOJA_INFO_RUB)
    evaluacija_status_naslov.config(
        text="Evaluacija modela",
        bg=BOJA_INFO,
        fg="#1d4ed8"
    )
    evaluacija_status_tekst.config(
        text=tekst,
        bg=BOJA_INFO,
        fg="#1e3a8a"
    )
    root.after(100, osvjezi_scroll_region)


def nacrtaj_figure_u_frame(fig, frame):
    ocisti_frame(frame)
    canvas_graf = FigureCanvasTkAgg(fig, master=frame)
    canvas_graf.draw()
    canvas_graf.get_tk_widget().pack(fill="both", expand=True)
    plt.close(fig)
    root.after(100, osvjezi_scroll_region)

# GRAF: Podjela podataka na train, validation i test skup
def prikazi_graf_podjele_podataka():
    ocisti_frame(split_graf_frame)

    if not postoje_pripremljeni_podaci():
        poruka = tk.Label(
            split_graf_frame,
            text="Podaci još nisu pripremljeni. Klikni 'Pripremi okolinu' za izradu train, validation i test datoteka.",
            bg=BOJA_KARTICE,
            fg=BOJA_MUTNOG_TEKSTA,
            font=("Segoe UI", 10),
            anchor="w",
            justify="left",
            wraplength=620
        )
        poruka.pack(fill="x", pady=12)
        return

    try:
        broj_train = len(pd.read_csv(TRAIN_FILE))
        broj_val = len(pd.read_csv(VAL_FILE))
        broj_test = len(pd.read_csv(TEST_FILE))

        vrijednosti = [broj_train, broj_val, broj_test]
        oznake = ["Train", "Validation", "Test"]

        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        ax.pie(
            vrijednosti,
            labels=oznake,
            autopct="%1.1f%%",
            startangle=90
        )
        ax.set_title("Podjela podataka po skupovima")
        ax.axis("equal")

        nacrtaj_figure_u_frame(fig, split_graf_frame)

    except Exception as e:
        poruka = tk.Label(
            split_graf_frame,
            text=f"Nije moguće prikazati graf podjele podataka.\n\nDetalji: {e}",
            bg=BOJA_KARTICE,
            fg="#b91c1c",
            font=("Segoe UI", 10),
            anchor="w",
            justify="left",
            wraplength=620
        )
        poruka.pack(fill="x", pady=12)


def provjeri_evaluacijske_datoteke():
    if not TEST_FILE.exists():
        raise FileNotFoundError("Nedostaje data/test.csv. Prvo klikni 'Pripremi okolinu'.")

    if not MODEL_PSO_FILE.exists():
        raise FileNotFoundError("Nedostaje models/model_pso.npz. Prvo klikni 'Pripremi okolinu'.")

    if not MODEL_BACKPROP_FILE.exists():
        raise FileNotFoundError("Nedostaje models/model_backpropagation.joblib. Prvo klikni 'Pripremi okolinu'.")


def ucitaj_test_podatke():
    test_df = pd.read_csv(TEST_FILE)
    X_test = test_df[ULAZNI_STUPCI].to_numpy(dtype=float)
    y_test = test_df[IZLAZNI_STUPAC].to_numpy(dtype=float)
    return X_test, y_test


def predikcije_pso_test():
    provjeri_evaluacijske_datoteke()
    X_test, y_test = ucitaj_test_podatke()

    model = np.load(MODEL_PSO_FILE, allow_pickle=True)

    W1 = model["W1"]
    b1 = model["b1"]
    W2 = model["W2"]
    b2 = model["b2"]
    X_mean = model["X_mean"]
    X_std = model["X_std"]
    y_mean = model["y_mean"]
    y_std = model["y_std"]

    X_norm = (X_test - X_mean) / X_std
    skriveni = np.maximum(0, X_norm @ W1 + b1)
    y_norm = skriveni @ W2 + b2
    y_pred = y_norm.ravel() * y_std + y_mean

    return y_test, y_pred


def predikcije_backpropagation_test():
    provjeri_evaluacijske_datoteke()
    test_df = pd.read_csv(TEST_FILE)

    spremljeno = joblib.load(MODEL_BACKPROP_FILE)
    model = spremljeno["model"]
    X_mean = spremljeno["X_mean"]
    X_std = spremljeno["X_std"]
    y_mean = spremljeno["y_mean"]
    y_std = spremljeno["y_std"]
    ulazni_stupci = spremljeno["ulazni_stupci"]
    izlazni_stupac = spremljeno["izlazni_stupac"]

    X_test = test_df[ulazni_stupci].values
    y_test = test_df[izlazni_stupac].values

    X_test_norm = (X_test - X_mean) / X_std
    y_pred_norm = model.predict(X_test_norm)
    y_pred = y_pred_norm * y_std + y_mean

    return y_test, y_pred


def izracunaj_metrike(y_stvarno, y_predikcija):
    mse = float(np.mean((y_stvarno - y_predikcija) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_stvarno - y_predikcija)))
    return mse, rmse, mae

# GRAF: Stvarne i predviđene vrijednosti na test skupu
def prikazi_graf_stvarno_predvideno(naziv_modela, funkcija_predikcija):
    try:
        y_test, y_pred = funkcija_predikcija()

        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.plot(y_test, label="Stvarne vrijednosti")
        ax.plot(y_pred, label="Predviđene vrijednosti")
        ax.set_title(f"{naziv_modela}: stvarno vs predviđeno")
        ax.set_xlabel("Redni broj testnog primjera")
        ax.set_ylabel("Skok u dalj (cm)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        nacrtaj_figure_u_frame(fig, evaluacija_graf_frame)

        mse, rmse, mae = izracunaj_metrike(y_test, y_pred)
        prikazi_info_evaluacije(
            f"Prikazan je graf stvarnih i predviđenih vrijednosti za {naziv_modela}.\n\n"
            f"MSE: {mse:.4f} cm²\n"
            f"RMSE: {rmse:.4f} cm\n"
            f"MAE: {mae:.4f} cm\n\n"
            "Objašnjenje metrika:\n"
            "MSE označava srednju kvadratnu pogrešku. Veće pogreške jače kažnjava jer se razlike kvadriraju.\n"
            "RMSE je korijen srednje kvadratne pogreške i izražava se u istoj mjernoj jedinici kao ciljna vrijednost, ovdje u centimetrima.\n"
            "MAE označava srednju apsolutnu pogrešku i pokazuje prosječno odstupanje predikcije od stvarne vrijednosti u centimetrima.\n\n"
            "Kod svih ovih metrika manja vrijednost znači bolji model."
        )

    except Exception as e:
        prikazi_gresku_evaluacije(str(e))

# GRAF: Reziduali modela na test skupu
def prikazi_graf_reziduala(naziv_modela, funkcija_predikcija):
    try:
        y_test, y_pred = funkcija_predikcija()
        reziduali = y_test - y_pred

        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.axhline(0, linewidth=1)
        ax.scatter(range(len(reziduali)), reziduali, s=18)
        ax.set_title(f"Reziduali: {naziv_modela}")
        ax.set_xlabel("Redni broj testnog primjera")
        ax.set_ylabel("Greška predikcije (cm)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        nacrtaj_figure_u_frame(fig, evaluacija_graf_frame)
        prikazi_info_evaluacije(
            f"Prikazan je graf reziduala za {naziv_modela}.\n\n"
            "Rezidual je razlika između stvarne i predviđene vrijednosti:\n"
            "rezidual = stvarna vrijednost - predviđena vrijednost.\n\n"
            "Ako su točke blizu nule, model ima manju pogrešku. "
            "Pozitivna vrijednost znači da je stvarni skok veći od predviđenog, "
            "a negativna vrijednost znači da je model predvidio veći skok od stvarnog."
        )

    except Exception as e:
        prikazi_gresku_evaluacije(str(e))

# GRAF: Usporedba metrika PSO i Backpropagation modela
def prikazi_usporedbu_metrika():
    try:
        y_pso, pred_pso = predikcije_pso_test()
        y_back, pred_back = predikcije_backpropagation_test()

        pso_mse, pso_rmse, pso_mae = izracunaj_metrike(y_pso, pred_pso)
        back_mse, back_rmse, back_mae = izracunaj_metrike(y_back, pred_back)

        metrike = ["RMSE", "MAE"]
        pso_vrijednosti = [pso_rmse, pso_mae]
        back_vrijednosti = [back_rmse, back_mae]

        x = np.arange(len(metrike))
        sirina = 0.35

        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.bar(x - sirina / 2, pso_vrijednosti, sirina, label="PSO")
        ax.bar(x + sirina / 2, back_vrijednosti, sirina, label="Backpropagation")
        ax.set_title("Usporedba modela prema grešci")
        ax.set_xlabel("Metrika")
        ax.set_ylabel("Greška (cm)")
        ax.set_xticks(x)
        ax.set_xticklabels(metrike)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        nacrtaj_figure_u_frame(fig, evaluacija_graf_frame)
        prikazi_info_evaluacije(
            "Prikazana je usporedba modela prema RMSE i MAE metrikama.\n\n"
            f"PSO - MSE: {pso_mse:.4f} cm², RMSE: {pso_rmse:.4f} cm, MAE: {pso_mae:.4f} cm\n"
            f"Backpropagation - MSE: {back_mse:.4f} cm², RMSE: {back_rmse:.4f} cm, MAE: {back_mae:.4f} cm\n\n"
            "Objašnjenje metrika:\n"
            "MSE označava srednju kvadratnu pogrešku. Budući da kvadrira razlike, veće pogreške imaju veći utjecaj na rezultat.\n"
            "RMSE je korijen MSE vrijednosti i zato je izražen u centimetrima, kao i predviđeni skok.\n"
            "MAE prikazuje prosječnu apsolutnu razliku između stvarne i predviđene vrijednosti.\n\n"
            "Manja vrijednost MSE, RMSE i MAE označava precizniji model."
        )

    except Exception as e:
        prikazi_gresku_evaluacije(str(e))


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

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    rezultat = subprocess.run(
        [sys.executable, "-X", "utf8", script_path.name],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env
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
        usporedi_button.config(state="disabled", cursor="")
        reset_button.config(state="disabled", cursor="")
    else:
        okolina_button.config(state="normal", cursor="hand2")
        usporedi_button.config(state="normal", cursor="hand2")
        reset_button.config(state="normal", cursor="hand2")


def priprema_okoline_worker():
    log = []

    try:
        root.after(
            0,
            lambda: azuriraj_status_pripreme(
                "Priprema okoline je pokrenuta...\n\n"
                "[1/5] Provjera podataka..."
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
                "[1/5] Podaci su spremni.\n"
                "[2/5] Provjera PSO modela..."
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
                "[1/5] Podaci su spremni.\n"
                "[2/5] PSO model je spreman.\n"
                "[3/5] Evaluacija PSO modela..."
            )
        )

        if TEST_FILE.exists() and MODEL_PSO_FILE.exists():
            pokreni_skriptu(SCRIPT_EVALUATE_TEST)
            log.append("✓ Evaluacija PSO modela je završena.")
        else:
            log.append("⚠ Evaluacija PSO modela je preskočena.")

        root.after(
            0,
            lambda: azuriraj_status_pripreme(
                "Priprema okoline je pokrenuta...\n\n"
                "[1/5] Podaci su spremni.\n"
                "[2/5] PSO model je spreman.\n"
                "[3/5] PSO evaluacija je završena.\n"
                "[4/5] Provjera Backpropagation modela..."
            )
        )

        if MODEL_BACKPROP_FILE.exists():
            log.append("✓ Backpropagation model već postoji.")
        else:
            pokreni_skriptu(SCRIPT_TRAIN_BACKPROP)
            log.append("✓ Backpropagation model je istreniran.")

        root.after(
            0,
            lambda: azuriraj_status_pripreme(
                "Priprema okoline je pokrenuta...\n\n"
                "[1/5] Podaci su spremni.\n"
                "[2/5] PSO model je spreman.\n"
                "[3/5] PSO evaluacija je završena.\n"
                "[4/5] Backpropagation model je spreman.\n"
                "[5/5] Evaluacija Backpropagation modela..."
            )
        )

        if TEST_FILE.exists() and MODEL_BACKPROP_FILE.exists():
            pokreni_skriptu(SCRIPT_EVALUATE_BACKPROP)
            log.append("✓ Evaluacija Backpropagation modela je završena.")
        else:
            log.append("⚠ Evaluacija Backpropagation modela je preskočena.")

        zavrsni_tekst = "\n".join(log)

        def zavrsi_pripremu():
            prikazi_uspjeh_pripreme("Okolina je spremna.\n\n" + zavrsni_tekst)
            prikazi_graf_podjele_podataka()

        root.after(0, zavrsi_pripremu)

    except Exception as e:
        zavrsni_tekst = "\n".join(log)

        if zavrsni_tekst:
            zavrsni_tekst += "\n\n"

        zavrsni_tekst += str(e)

        root.after(0, lambda: prikazi_gresku_pripreme(zavrsni_tekst))

    finally:
        root.after(0, lambda: postavi_gumbe_zauzeto(False))


def pokreni_pripremu_okoline():
    postavi_gumbe_zauzeto(True)

    prikazi_info_pripreme(
        "Priprema okoline je pokrenuta...\n\n"
        "[1/5] Provjera podataka..."
    )

    thread = threading.Thread(target=priprema_okoline_worker, daemon=True)
    thread.start()


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

    if "priprema_opis" in globals():
        priprema_opis.config(wraplength=dostupna_sirina)

    if "evaluacija_opis" in globals():
        evaluacija_opis.config(wraplength=dostupna_sirina)



postavi_windows_app_id()

root = tk.Tk()
root.title("Predikcija skoka u dalj")

root.geometry("760x760")
root.minsize(520, 500)

root.configure(bg=BOJA_POZADINE)

postavi_ikonu(root)



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


notebook = ttk.Notebook(outer_frame)
notebook.pack(fill="x", expand=False)

priprema_tab = tk.Frame(notebook, bg=BOJA_POZADINE)
predikcija_tab = tk.Frame(notebook, bg=BOJA_POZADINE)
evaluacija_tab = tk.Frame(notebook, bg=BOJA_POZADINE)

notebook.add(priprema_tab, text="Priprema i podaci")
notebook.add(evaluacija_tab, text="Evaluacija")
notebook.add(predikcija_tab, text="Predikcija")



priprema_card = tk.Frame(
    priprema_tab,
    bg=BOJA_KARTICE,
    highlightthickness=1,
    highlightbackground="#e5e7eb"
)
priprema_card.pack(fill="x", padx=0, pady=14)

priprema_content = tk.Frame(priprema_card, bg=BOJA_KARTICE)
priprema_content.pack(fill="x", padx=24, pady=24)

priprema_naslov = tk.Label(
    priprema_content,
    text="Priprema okoline i podataka",
    bg=BOJA_KARTICE,
    fg=BOJA_TEKSTA,
    font=("Segoe UI", 15, "bold"),
    anchor="w"
)
priprema_naslov.pack(fill="x", pady=(0, 12))

priprema_opis = tk.Label(
    priprema_content,
    text=(
        "Prije treniranja neuronskih mreža podaci se pripremaju i dijele "
        "na tri odvojena skupa podataka. Svaki skup ima posebnu ulogu u razvoju i provjeri modela.\n\n"
        "Train skup koristi se za učenje modela. Na temelju tih podataka neuronska mreža "
        "prilagođava svoje težine i uči odnos između ulaznih značajki osobe i rezultata skoka.\n\n"
        "Validation skup koristi se tijekom razvoja modela. Pomoću njega se provjerava ponašanje "
        "modela i mogu se podešavati parametri, bez korištenja testnog skupa.\n\n"
        "Test skup koristi se tek na kraju za završnu evaluaciju. On predstavlja nove podatke koje "
        "model nije koristio tijekom treniranja, pa daje objektivniju procjenu točnosti modela."
    ),
    bg=BOJA_KARTICE,
    fg=BOJA_TEKSTA,
    font=("Segoe UI", 11),
    anchor="w",
    justify="left",
    wraplength=650
)
priprema_opis.pack(fill="x", pady=(0, 16))

okolina_button = tk.Button(
    priprema_content,
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

priprema_status_frame = tk.Frame(
    priprema_content,
    bg=BOJA_KARTICE,
    highlightthickness=1,
    highlightbackground=BOJA_KARTICE
)
priprema_status_frame.pack(fill="x", pady=(16, 0))

priprema_status_naslov = tk.Label(
    priprema_status_frame,
    text="Status",
    bg=BOJA_KARTICE,
    fg=BOJA_MUTNOG_TEKSTA,
    font=("Segoe UI", 11, "bold"),
    anchor="w"
)
priprema_status_naslov.pack(fill="x", padx=16, pady=(14, 4))

priprema_status_tekst = tk.Label(
    priprema_status_frame,
    bg=BOJA_KARTICE,
    fg=BOJA_MUTNOG_TEKSTA,
    font=("Segoe UI", 12),
    anchor="w",
    justify="left"
)
priprema_status_tekst.pack(fill="x", padx=16, pady=(0, 16))

split_graf_kartica = tk.Frame(
    priprema_content,
    bg=BOJA_KARTICE,
    highlightthickness=1,
    highlightbackground="#e5e7eb"
)
split_graf_kartica.pack(fill="x", pady=(16, 0))

split_graf_naslov = tk.Label(
    split_graf_kartica,
    text="Kružni graf podjele podataka",
    bg=BOJA_KARTICE,
    fg=BOJA_TEKSTA,
    font=("Segoe UI", 11, "bold"),
    anchor="w"
)
split_graf_naslov.pack(fill="x", padx=16, pady=(14, 4))

split_graf_opis = tk.Label(
    split_graf_kartica,
    bg=BOJA_KARTICE,
    fg=BOJA_MUTNOG_TEKSTA,
    font=("Segoe UI", 10),
    anchor="w",
    justify="left",
    wraplength=650
)
split_graf_opis.pack(fill="x", padx=16, pady=(0, 10))

split_graf_frame = tk.Frame(split_graf_kartica, bg=BOJA_KARTICE)
split_graf_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))


card = tk.Frame(
    predikcija_tab,
    bg=BOJA_KARTICE,
    highlightthickness=1,
    highlightbackground="#e5e7eb"
)
card.pack(fill="x", padx=0, pady=14)

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


button_frame = tk.Frame(card, bg=BOJA_KARTICE)
button_frame.pack(fill="x", padx=24, pady=(0, 22))

usporedi_button = tk.Button(
    button_frame,
    text="Usporedi modele",
    command=pokreni_usporedbu_modela,
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
usporedi_button.pack(fill="x", ipady=11, pady=(0, 10))
hover_gumb(usporedi_button, BOJA_PSO_GUMB, BOJA_PSO_GUMB_HOVER)

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


usporedba_frame = tk.Frame(card, bg=BOJA_KARTICE)
usporedba_frame.pack(fill="x", padx=24, pady=(0, 22))

pso_kartica = tk.Frame(
    usporedba_frame,
    bg="#eff6ff",
    highlightthickness=1,
    highlightbackground="#3b82f6"
)
pso_kartica.pack(side="left", fill="both", expand=True, padx=(0, 8))

pso_naslov_label = tk.Label(
    pso_kartica,
    text="PSO model",
    bg="#eff6ff",
    fg="#1d4ed8",
    font=("Segoe UI", 11, "bold"),
    anchor="center"
)
pso_naslov_label.pack(fill="x", padx=12, pady=(14, 4))

pso_vrijednost_label = tk.Label(
    pso_kartica,
    text="-",
    bg="#eff6ff",
    fg="#1d4ed8",
    font=("Segoe UI", 16, "bold"),
    anchor="center"
)
pso_vrijednost_label.pack(fill="x", padx=12, pady=(0, 14))

backprop_kartica = tk.Frame(
    usporedba_frame,
    bg="#ecfdf5",
    highlightthickness=1,
    highlightbackground="#10b981"
)
backprop_kartica.pack(side="left", fill="both", expand=True, padx=(8, 0))

backprop_naslov_label = tk.Label(
    backprop_kartica,
    text="Backpropagation",
    bg="#ecfdf5",
    fg="#047857",
    font=("Segoe UI", 11, "bold"),
    anchor="center"
)
backprop_naslov_label.pack(fill="x", padx=12, pady=(14, 4))

backprop_vrijednost_label = tk.Label(
    backprop_kartica,
    text="-",
    bg="#ecfdf5",
    fg="#047857",
    font=("Segoe UI", 16, "bold"),
    anchor="center"
)
backprop_vrijednost_label.pack(fill="x", padx=12, pady=(0, 14))

graf_predikcija_kartica = tk.Frame(
    card,
    bg=BOJA_KARTICE,
    highlightthickness=1,
    highlightbackground="#e5e7eb"
)
graf_predikcija_kartica.pack(fill="x", padx=24, pady=(0, 22))

graf_predikcija_naslov = tk.Label(
    graf_predikcija_kartica,
    text="Graf usporedbe predikcija",
    bg=BOJA_KARTICE,
    fg=BOJA_TEKSTA,
    font=("Segoe UI", 11, "bold"),
    anchor="w"
)
graf_predikcija_naslov.pack(fill="x", padx=16, pady=(14, 4))

graf_predikcija_opis = tk.Label(
    graf_predikcija_kartica,
    bg=BOJA_KARTICE,
    fg=BOJA_MUTNOG_TEKSTA,
    font=("Segoe UI", 10),
    anchor="w",
    justify="left",
    wraplength=620
)
graf_predikcija_opis.pack(fill="x", padx=16, pady=(0, 10))

graf_predikcija_frame = tk.Frame(graf_predikcija_kartica, bg=BOJA_KARTICE)
graf_predikcija_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))


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
    bg=BOJA_KARTICE,
    fg=BOJA_MUTNOG_TEKSTA,
    font=("Segoe UI", 12),
    anchor="w",
    justify="left"
)
rezultat_tekst.pack(fill="x", padx=16, pady=(0, 16))

evaluacija_card = tk.Frame(
    evaluacija_tab,
    bg=BOJA_KARTICE,
    highlightthickness=1,
    highlightbackground="#e5e7eb"
)
evaluacija_card.pack(fill="x", padx=0, pady=14)

evaluacija_content = tk.Frame(evaluacija_card, bg=BOJA_KARTICE)
evaluacija_content.pack(fill="x", padx=24, pady=24)

evaluacija_naslov = tk.Label(
    evaluacija_content,
    text="Evaluacija modela",
    bg=BOJA_KARTICE,
    fg=BOJA_TEKSTA,
    font=("Segoe UI", 15, "bold"),
    anchor="w"
)
evaluacija_naslov.pack(fill="x", pady=(0, 12))

evaluacija_opis = tk.Label(
    evaluacija_content,
    bg=BOJA_KARTICE,
    fg=BOJA_TEKSTA,
    font=("Segoe UI", 11),
    anchor="w",
    justify="left",
    wraplength=650
)
evaluacija_opis.pack(fill="x", pady=(0, 16))

evaluacija_button_frame = tk.Frame(evaluacija_content, bg=BOJA_KARTICE)
evaluacija_button_frame.pack(fill="x", pady=(0, 16))

eval_pso_button = tk.Button(
    evaluacija_button_frame,
    text="PSO: stvarno vs predviđeno",
    command=lambda: prikazi_graf_stvarno_predvideno("PSO model", predikcije_pso_test),
    bg=BOJA_PSO_GUMB,
    fg="white",
    activebackground=BOJA_PSO_GUMB_HOVER,
    activeforeground="white",
    relief="flat",
    bd=0,
    font=("Segoe UI", 10, "bold"),
    cursor="hand2"
)
eval_pso_button.pack(fill="x", ipady=9, pady=(0, 8))
hover_gumb(eval_pso_button, BOJA_PSO_GUMB, BOJA_PSO_GUMB_HOVER)

eval_back_button = tk.Button(
    evaluacija_button_frame,
    text="Backpropagation: stvarno vs predviđeno",
    command=lambda: prikazi_graf_stvarno_predvideno("Backpropagation model", predikcije_backpropagation_test),
    bg="#059669",
    fg="white",
    activebackground="#047857",
    activeforeground="white",
    relief="flat",
    bd=0,
    font=("Segoe UI", 10, "bold"),
    cursor="hand2"
)
eval_back_button.pack(fill="x", ipady=9, pady=(0, 8))
hover_gumb(eval_back_button, "#059669", "#047857")

eval_metrike_button = tk.Button(
    evaluacija_button_frame,
    text="Usporedba metrika",
    command=prikazi_usporedbu_metrika,
    bg="#7c3aed",
    fg="white",
    activebackground="#6d28d9",
    activeforeground="white",
    relief="flat",
    bd=0,
    font=("Segoe UI", 10, "bold"),
    cursor="hand2"
)
eval_metrike_button.pack(fill="x", ipady=9, pady=(0, 8))
hover_gumb(eval_metrike_button, "#7c3aed", "#6d28d9")

eval_reziduali_pso_button = tk.Button(
    evaluacija_button_frame,
    text="Reziduali PSO modela",
    command=lambda: prikazi_graf_reziduala("PSO model", predikcije_pso_test),
    bg="#374151",
    fg="white",
    activebackground="#1f2937",
    activeforeground="white",
    relief="flat",
    bd=0,
    font=("Segoe UI", 10, "bold"),
    cursor="hand2"
)
eval_reziduali_pso_button.pack(fill="x", ipady=9, pady=(0, 8))
hover_gumb(eval_reziduali_pso_button, "#374151", "#1f2937")

eval_reziduali_back_button = tk.Button(
    evaluacija_button_frame,
    text="Reziduali Backpropagation modela",
    command=lambda: prikazi_graf_reziduala("Backpropagation model", predikcije_backpropagation_test),
    bg="#374151",
    fg="white",
    activebackground="#1f2937",
    activeforeground="white",
    relief="flat",
    bd=0,
    font=("Segoe UI", 10, "bold"),
    cursor="hand2"
)
eval_reziduali_back_button.pack(fill="x", ipady=9)
hover_gumb(eval_reziduali_back_button, "#374151", "#1f2937")

evaluacija_status_frame = tk.Frame(
    evaluacija_content,
    bg=BOJA_KARTICE,
    highlightthickness=1,
    highlightbackground=BOJA_KARTICE
)
evaluacija_status_frame.pack(fill="x", pady=(0, 16))

evaluacija_status_naslov = tk.Label(
    evaluacija_status_frame,
    text="Status",
    bg=BOJA_KARTICE,
    fg=BOJA_MUTNOG_TEKSTA,
    font=("Segoe UI", 11, "bold"),
    anchor="w"
)
evaluacija_status_naslov.pack(fill="x", padx=16, pady=(14, 4))

evaluacija_status_tekst = tk.Label(
    evaluacija_status_frame,
    bg=BOJA_KARTICE,
    fg=BOJA_MUTNOG_TEKSTA,
    font=("Segoe UI", 11),
    anchor="w",
    justify="left",
    wraplength=650
)
evaluacija_status_tekst.pack(fill="x", padx=16, pady=(0, 16))

evaluacija_graf_kartica = tk.Frame(
    evaluacija_content,
    bg=BOJA_KARTICE,
    highlightthickness=1,
    highlightbackground="#e5e7eb"
)
evaluacija_graf_kartica.pack(fill="x")

evaluacija_graf_naslov = tk.Label(
    evaluacija_graf_kartica,
    text="Prikaz grafa",
    bg=BOJA_KARTICE,
    fg=BOJA_TEKSTA,
    font=("Segoe UI", 11, "bold"),
    anchor="w"
)
evaluacija_graf_naslov.pack(fill="x", padx=16, pady=(14, 4))

evaluacija_graf_frame = tk.Frame(evaluacija_graf_kartica, bg=BOJA_KARTICE)
evaluacija_graf_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))

ucitaj_person_json_ako_postoji()

root.after(100, osvjezi_scroll_region)
root.after(200, prikazi_graf_podjele_podataka)

root.mainloop()