import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Učitavanje CSV datoteke
df = pd.read_csv("data.csv")

# Micanje nepotrebnih stupaca
df = df.drop(columns=["date", "street", "country", "statezip"])

# Transformacija tekstualnog stupca 'city' u brojeve
encoder = LabelEncoder()
df["city"] = encoder.fit_transform(df["city"])

# Spremanje očišćenog CSV-a
df.to_csv("data_housing_cleaned.csv", index=False)

print("CSV očišćen i spremljen!")
print(df.head()) 