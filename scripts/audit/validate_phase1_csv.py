import os
import pandas as pd

CSV_DIR = "data/csv"

EXPECTED_MIN_ROWS = 520
EXPECTED_MAX_ROWS = 700
MAX_ALLOWED_GAP_DAYS = 3

results = []

for file in sorted(os.listdir(CSV_DIR)):
    if not file.endswith(".csv"):
        continue

    path = os.path.join(CSV_DIR, file)
    df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp")

    gaps = df["timestamp"].diff().dt.days.dropna()
    max_gap = int(gaps.max()) if len(gaps) else 0

    ohlc_violations = (
        (df["high"] < df["low"]).sum()
        + (df["close"] > df["high"]).sum()
        + (df["close"] < df["low"]).sum()
    )

    status = "PASS"
    if (
        df["timestamp"].isna().any()
        or df.duplicated(subset=["timestamp"]).any()
        or ohlc_violations > 0
        or max_gap > MAX_ALLOWED_GAP_DAYS
        or not (EXPECTED_MIN_ROWS <= len(df) <= EXPECTED_MAX_ROWS)
    ):
        status = "FAIL"

    results.append({
        "file": file,
        "rows": len(df),
        "start_date": df["timestamp"].iloc[0].date(),
        "end_date": df["timestamp"].iloc[-1].date(),
        "duplicates": int(df.duplicated(subset=["timestamp"]).sum()),
        "max_gap_days": max_gap,
        "ohlc_violations": int(ohlc_violations),
        "status": status
    })

summary = pd.DataFrame(results)
summary.to_csv("phase1_csv_validation_summary.csv", index=False)
print(summary)
