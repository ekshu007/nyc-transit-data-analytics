from pathlib import Path
import pandas as pd

files = sorted(Path("data").glob("*.parquet"))

print("Files downloaded:", len(files))

if files:
    df = pd.concat(
        [pd.read_parquet(file) for file in files],
        ignore_index=True
    )

    print("Total rows:", len(df))
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head())

    print("\nDate range:")
    print(df["service_date"].min(), "to", df["service_date"].max())
    