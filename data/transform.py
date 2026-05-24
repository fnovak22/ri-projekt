import pandas as pd

# Učitavanje CSV datoteke
df = pd.read_csv("data.csv")

# Micanje nepotrebnih stupaca
df = df.drop(columns=["date", "street", "country", "statezip"])

# One-Hot Encoding za gradove
df = pd.get_dummies(df, columns=["city"], dtype=int)

# Spremanje očišćenog CSV-a
df.to_csv("data_housing_cleaned_transformed.csv", index=False)

print("CSV očišćen i spremljen!")
print(df.head()) 