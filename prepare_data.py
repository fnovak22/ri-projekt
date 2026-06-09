from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
ULAZNA_DATOTEKA = DATA_DIR / "bodyPerformance.csv"
TRAIN_DATOTEKA = DATA_DIR / "train.csv"
VAL_DATOTEKA = DATA_DIR / "val.csv"
TEST_DATOTEKA = DATA_DIR / "test.csv"

ULAZNI_STUPCI = [
    "age",
    "gender",
    "height_cm",
    "weight_kg",
    "sit and bend forward_cm",
    "sit-ups counts",
]

IZLAZNI_STUPAC = "broad jump_cm"

RANDOM_STATE = 67


def main():
    df = pd.read_csv(ULAZNA_DATOTEKA)

    df = df[ULAZNI_STUPCI + [IZLAZNI_STUPAC]].copy()

    df["gender"] = df["gender"].map({"F": 1, "M": 0})

    df = df.dropna()

    train_val_df, test_df = train_test_split(
        df,
        test_size=0.15,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.15 / 0.85,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    DATA_DIR.mkdir(exist_ok=True)
    train_df.to_csv(TRAIN_DATOTEKA, index=False)
    val_df.to_csv(VAL_DATOTEKA, index=False)
    test_df.to_csv(TEST_DATOTEKA, index=False)

    ukupno = len(df)
    print("Podaci su podijeljeni i spremljeni u folder data.")
    print(f"Ukupno: {ukupno}")
    print(f"Train: {len(train_df)} ({len(train_df) / ukupno:.2%}) -> {TRAIN_DATOTEKA}")
    print(f"Val:   {len(val_df)} ({len(val_df) / ukupno:.2%}) -> {VAL_DATOTEKA}")
    print(f"Test:  {len(test_df)} ({len(test_df) / ukupno:.2%}) -> {TEST_DATOTEKA}")


if __name__ == "__main__":
    main()
