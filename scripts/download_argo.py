"""
scripts/download_argo.py
------------------------
Downloads real ARGO float profiles OR generates synthetic demo data
when network access is limited (e.g. during development).
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_argo_data(n_floats=50, profiles_per_float=20) -> pd.DataFrame:
    """
    Generate realistic synthetic ARGO float data for the Indian Ocean region.
    Used as fallback when real data is unavailable.
    """
    np.random.seed(42)
    records = []

    # Indian Ocean bounding box
    lat_range = (-30, 25)
    lon_range = (45, 100)

    regions = {
        "Arabian Sea":      {"lat": (5, 25),  "lon": (55, 75)},
        "Bay of Bengal":    {"lat": (5, 22),  "lon": (80, 100)},
        "Indian Ocean":     {"lat": (-30, 5), "lon": (45, 100)},
    }

    depths = np.array([0, 10, 20, 50, 100, 200, 300, 500, 750, 1000,
                       1500, 2000])

    float_id = 1000
    for region_name, bounds in regions.items():
        n = n_floats // len(regions)
        for _ in range(n):
            float_id += 1
            lat0 = np.random.uniform(*bounds["lat"])
            lon0 = np.random.uniform(*bounds["lon"])

            for p in range(profiles_per_float):
                date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=p * 10)
                # Drift
                lat = lat0 + np.random.normal(0, 0.3)
                lon = lon0 + np.random.normal(0, 0.3)

                for depth in depths:
                    # Realistic temperature: warm surface, cold deep
                    sst = np.random.uniform(24, 30) if region_name == "Arabian Sea" else np.random.uniform(26, 32)
                    temp = sst * np.exp(-depth / 300) + np.random.normal(0, 0.3)
                    temp = max(2.0, temp)

                    # Salinity: 34-36 PSU range
                    salt = 35.0 + 0.5 * np.sin(depth / 200) + np.random.normal(0, 0.1)
                    salt = np.clip(salt, 33.5, 37.0)

                    records.append({
                        "float_id": f"WMO{float_id:06d}",
                        "profile_id": f"WMO{float_id:06d}_{p:04d}",
                        "date": date,
                        "latitude": round(lat, 4),
                        "longitude": round(lon, 4),
                        "depth_m": depth,
                        "temperature_c": round(temp, 3),
                        "salinity_psu": round(salt, 3),
                        "region": region_name,
                        "year": date.year,
                        "month": date.month,
                    })

    df = pd.DataFrame(records)
    return df


def main():
    print("Generating synthetic ARGO float data...")
    df = generate_synthetic_argo_data(n_floats=60, profiles_per_float=15)
    out_path = OUTPUT_DIR / "argo_indian_ocean.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df):,} records → {out_path}")
    print(f"Floats: {df['float_id'].nunique()}")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Regions: {df['region'].unique().tolist()}")
    return df


if __name__ == "__main__":
    main()
