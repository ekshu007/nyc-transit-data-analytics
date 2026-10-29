
from pathlib import Path
import urllib.request

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Download one Parquet file per month
for month in range(1, 13):
    year = 2025
    filename = f"{year}-{month:02d}.parquet"

    url = (
        f"https://data.transitlab.nyc/v1/"
        f"subway_segment_times/year={year}/"
        f"month={month:02d}/{filename}"
    )

    destination = DATA_DIR / filename

    try:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, destination)
        print(f"Saved: {destination}")
    except Exception as error:
        print(f"Could not download {filename}: {error}")

print("Download process finished.")
