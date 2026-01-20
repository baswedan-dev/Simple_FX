import os
import pandas as pd

PARQUET_DIR = "data/raw"   # parquet source
CSV_DIR = "data/csv"       # csv output

os.makedirs(CSV_DIR, exist_ok=True)

def normalize_timestamp(df):
    # Case 1: timestamp is index
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)

    # Case 2: timestamp column exists
    for c in df.columns:
        if c.lower() in ("timestamp", "date", "time", "dt"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
            if df[c].dtype == "datetime64[ns]":
                df.rename(columns={c: "timestamp"}, inplace=True)
                return df

    raise ValueError("No valid timestamp column or index found")

for file in os.listdir(PARQUET_DIR):
    if not file.endswith(".parquet"):
        continue

    parquet_path = os.path.join(PARQUET_DIR, file)
    csv_path = os.path.join(
        CSV_DIR,
        file.replace(".parquet", ".csv")
    )

    print(f"Converting {file}")

    df = pd.read_parquet(parquet_path)
    df = normalize_timestamp(df)

    # Enforce correct ordering
    df = df.sort_values("timestamp")

    df.to_csv(csv_path, index=False)

print("All files converted successfully.")
