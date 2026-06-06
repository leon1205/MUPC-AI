"""
Download SMART-DS dataset subset from NREL OEDI data lake.
Downloads PV (solar) generation profiles, per-unit load shapes,
and building-level load data in Parquet format.
"""
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE_URL = "https://oedi-data-lake.s3.amazonaws.com"
DATA_DIR = Path(__file__).parent
SOLAR_DIR = DATA_DIR / "smart_ds" / "solar"
PROFILE_DIR = DATA_DIR / "smart_ds" / "load_profiles"
PARQUET_DIR = DATA_DIR / "smart_ds" / "parquet"

for d in [SOLAR_DIR, PROFILE_DIR, PARQUET_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PREFIX = "SMART-DS/v1.0/2016/SFO/P1U"

def download_file(s3_key: str, dest: Path, retries: int = 3) -> bool:
    """Download a single file with retry logic."""
    url = f"{BASE_URL}/{s3_key}"
    fname = os.path.basename(s3_key)
    dest_path = dest / fname

    if dest_path.exists():
        # Check if complete (solar files ~1.7MB, profiles ~260KB-680KB, parquet varies)
        min_size = 50000  # 50KB minimum for CSV, 100KB for parquet
        if fname.endswith('.parquet'):
            min_size = 100000
        if dest_path.stat().st_size > min_size:
            print(f"  SKIP (exists): {fname}")
            return True

    for attempt in range(retries):
        try:
            print(f"  Downloading: {fname} ({attempt+1}/{retries})...", end=" ", flush=True)
            urllib.request.urlretrieve(url, dest_path)
            size_kb = dest_path.stat().st_size / 1024
            print(f"{size_kb:.0f} KB OK")
            return True
        except Exception as e:
            print(f"FAILED: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return False


# ── Solar (PV) data: full year 15-min resolution ─────────────────────
# Select 6 representative PV profiles:
# 2 locations × (south-facing 180°, east 90°, west 270°) × tilt=15°
SOLAR_FILES = [
    f"{PREFIX}/solar_data/SFO_37.7083_-122.4074_15_180_full.csv",
    f"{PREFIX}/solar_data/SFO_37.7083_-122.4074_15_90_full.csv",
    f"{PREFIX}/solar_data/SFO_37.7083_-122.4074_15_270_full.csv",
    f"{PREFIX}/solar_data/SFO_37.7452_-122.4074_15_180_full.csv",
    f"{PREFIX}/solar_data/SFO_37.7452_-122.4074_15_90_full.csv",
    f"{PREFIX}/solar_data/SFO_37.7452_-122.4074_15_270_full.csv",
]

# ── Per-unit load profiles ─────────────────────────────────────────
# Select a small representative set: commercial + residential, real + reactive
LOAD_FILES = [
    f"{PREFIX}/profiles/com_kw_1023_pu.csv",
    f"{PREFIX}/profiles/com_kvar_1023_pu.csv",
    f"{PREFIX}/profiles/com_kw_1033_pu.csv",
    f"{PREFIX}/profiles/com_kvar_1033_pu.csv",
]

# Check for residential files (may use different naming)
RES_CHECK = [
    f"{PREFIX}/profiles/res_kw_1_pu.csv",
    f"{PREFIX}/profiles/res_kvar_1_pu.csv",
]

# ── Parquet building-level load data ───────────────────────────────
PARQUET_FILES = [
    f"{PREFIX}/load_data/com_1023.parquet",
    f"{PREFIX}/load_data/com_1033.parquet",
]

print("=== Phase 1: Solar PV Data ===")
ok = 0
for f in SOLAR_FILES:
    if download_file(f, SOLAR_DIR):
        ok += 1
print(f"  Downloaded {ok}/{len(SOLAR_FILES)} solar files\n")

print("=== Phase 2: Load Profiles (per-unit) ===")
ok = 0
for f in LOAD_FILES:
    if download_file(f, PROFILE_DIR):
        ok += 1

# Try residential profiles
for f in RES_CHECK:
    try:
        urllib.request.urlopen(f"{BASE_URL}/{f}")
        ok += 1 if download_file(f, PROFILE_DIR) else 0
    except urllib.error.HTTPError:
        pass  # File doesn't exist

print(f"  Downloaded {ok} load profile files\n")

print("=== Phase 3: Parquet Building Data ===")
ok = 0
for f in PARQUET_FILES:
    if download_file(f, PARQUET_DIR):
        ok += 1
print(f"  Downloaded {ok}/{len(PARQUET_FILES)} parquet files\n")

print("=== Download Complete ===")
# Show summary
for label, d in [("Solar", SOLAR_DIR), ("Profiles", PROFILE_DIR), ("Parquet", PARQUET_DIR)]:
    files = list(d.glob("*"))
    if files:
        total_size = sum(f.stat().st_size for f in files)
        print(f"  {label}: {len(files)} files, {total_size/1024/1024:.1f} MB")
