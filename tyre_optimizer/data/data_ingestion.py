"""
Wave 1 — Data Ingestion & Unified Data Layer
============================================
Reads the per-factory CSV/Excel silos and merges them into a single,
validated, canonical DataFrame — the "Single Version of the Truth."

Key responsibilities:
  • Ingest any mix of CSV or Excel files from the /data/raw/ directory
  • Validate schema and data quality (nulls, negative quantities, date gaps)
  • Enrich with segment, regional, and promotional metadata
  • Expose a clean unified_sales(), unified_inventory(), unified_logistics(),
    and unified_energy() interface for downstream modules
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import FACTORIES, SKUS, SEGMENTS, SEASONAL_INDICES

RAW_DIR = Path(__file__).resolve().parent / "raw"


# ── Schema definitions ────────────────────────────────────────────────────────

SALES_SCHEMA = {
    "date": "datetime64[ns]",
    "factory": str,
    "sku": str,
    "segment": str,
    "channel": str,
    "units_sold": "int64",
    "promotion": str,
    "seasonal_index": float,
    "price_inr": float,
}

INVENTORY_SCHEMA = {
    "date": "datetime64[ns]",
    "factory": str,
    "sku": str,
    "segment": str,
    "opening_stock": "int64",
    "production_qty": "int64",
    "units_sold": "int64",
    "closing_stock": "int64",
    "stockout_flag": "int64",
    "days_of_stock": float,
}

LOGISTICS_SCHEMA = {
    "date": "datetime64[ns]",
    "origin": str,
    "destination": str,
    "sku": str,
    "segment": str,
    "transport_mode": str,
    "quantity": "int64",
    "weight_tonnes": float,
    "distance_km": "int64",
    "freight_cost_inr": float,
    "transit_days": "int64",
    "co2_factor": float,
}

ENERGY_SCHEMA = {
    "date": object,
    "factory": str,
    "grid_uptime_pct": float,
    "sofc_deployed": "int64",
    "power_uptime_pct": float,
    "avg_active_presses": float,
    "total_batches_scrapped": "int64",
    "total_scrap_cost_inr": float,
    "total_energy_kwh": float,
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _read_file(path: Path) -> pd.DataFrame:
    """Read CSV or Excel transparently."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file format: {suffix}")


def _validate(df: pd.DataFrame, schema: dict, source_name: str) -> pd.DataFrame:
    """Apply type coercions and basic quality checks."""
    for col, dtype in schema.items():
        if col not in df.columns:
            raise KeyError(f"[{source_name}] Missing required column: '{col}'")
        try:
            if dtype == "datetime64[ns]":
                df[col] = pd.to_datetime(df[col])
            elif dtype in ("int64",):
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
            elif dtype == float:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = df[col].astype(str)
        except Exception as exc:
            raise TypeError(f"[{source_name}] Column '{col}' coercion failed: {exc}") from exc

    # Negative quantity guard
    for col in ["units_sold", "closing_stock", "production_qty", "quantity"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg:
                print(f"  [WARN] {source_name}: {neg} negative values in '{col}' — zeroing.")
                df[col] = df[col].clip(lower=0)

    # Unknown factory guard
    if "factory" in df.columns:
        unknown = ~df["factory"].isin(FACTORIES)
        if unknown.any():
            print(f"  [WARN] {source_name}: {unknown.sum()} rows with unknown factory — dropped.")
            df = df[~unknown]

    return df


def _enrich_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns: week, month, quarter, revenue_inr."""
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    df["revenue_inr"] = (df["units_sold"] * df["price_inr"]).round(2)
    # Reattach seasonal index if missing (for externally sourced files)
    if "seasonal_index" not in df.columns or df["seasonal_index"].isna().all():
        df["seasonal_index"] = df["month"].map(SEASONAL_INDICES)
    return df


def _enrich_inventory(df: pd.DataFrame) -> pd.DataFrame:
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["fill_rate"] = (
        df["units_sold"] / (df["opening_stock"] + df["production_qty"])
    ).clip(0, 1).round(4)
    df["overstock_flag"] = (df["days_of_stock"] > 45).astype(int)
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def unified_sales(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Merge per-factory sales files into one DataFrame.
    Falls back to generating synthetic data if no raw files exist.
    """
    raw_dir = raw_dir or RAW_DIR
    files = list(raw_dir.glob("sales_*.csv")) + list(raw_dir.glob("sales_*.xlsx"))

    if not files:
        print("[INFO] No raw sales files found — generating synthetic data…")
        _ensure_synthetic(raw_dir)
        files = list(raw_dir.glob("sales_*.csv"))

    frames = []
    for f in sorted(files):
        print(f"  Loading {f.name}…")
        frames.append(_read_file(f))

    df = pd.concat(frames, ignore_index=True)
    df = _validate(df, SALES_SCHEMA, "sales")
    df = _enrich_sales(df)
    df.sort_values(["factory", "sku", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[OK] Unified sales layer: {len(df):,} records across {df['factory'].nunique()} factories.")
    return df


def unified_inventory(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    raw_dir = raw_dir or RAW_DIR
    files = list(raw_dir.glob("inventory_*.csv")) + list(raw_dir.glob("inventory_*.xlsx"))

    if not files:
        _ensure_synthetic(raw_dir)
        files = list(raw_dir.glob("inventory_*.csv"))

    frames = [_read_file(f) for f in sorted(files)]
    df = pd.concat(frames, ignore_index=True)
    df = _validate(df, INVENTORY_SCHEMA, "inventory")
    df = _enrich_inventory(df)
    df.sort_values(["factory", "sku", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[OK] Unified inventory layer: {len(df):,} records.")
    return df


def unified_logistics(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    raw_dir = raw_dir or RAW_DIR
    files = list(raw_dir.glob("logistics_*.csv")) + list(raw_dir.glob("logistics_*.xlsx"))

    if not files:
        _ensure_synthetic(raw_dir)
        files = list(raw_dir.glob("logistics_*.csv"))

    frames = [_read_file(f) for f in sorted(files)]
    df = pd.concat(frames, ignore_index=True)
    df = _validate(df, LOGISTICS_SCHEMA, "logistics")
    df["date"] = pd.to_datetime(df["date"])
    df["cost_per_unit_inr"] = (df["freight_cost_inr"] / df["quantity"].replace(0, np.nan)).round(2)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[OK] Unified logistics layer: {len(df):,} shipment records.")
    return df


def unified_energy(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    raw_dir = raw_dir or RAW_DIR
    files = list(raw_dir.glob("energy_*.csv")) + list(raw_dir.glob("energy_*.xlsx"))

    if not files:
        _ensure_synthetic(raw_dir)
        files = list(raw_dir.glob("energy_*.csv"))

    frames = [_read_file(f) for f in sorted(files)]
    df = pd.concat(frames, ignore_index=True)
    df = _validate(df, ENERGY_SCHEMA, "energy")
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["factory", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[OK] Unified energy layer: {len(df):,} daily records.")
    return df


def network_snapshot(raw_dir: Optional[Path] = None) -> dict:
    """
    Returns a dict of all four unified DataFrames — the 'Digital Twin' snapshot.
    This is the primary entry point for downstream analytics modules.
    """
    print("\n=== Building Digital Twin Snapshot ===")
    return {
        "sales": unified_sales(raw_dir),
        "inventory": unified_inventory(raw_dir),
        "logistics": unified_logistics(raw_dir),
        "energy": unified_energy(raw_dir),
    }


def _ensure_synthetic(raw_dir: Path) -> None:
    """Generate synthetic data lazily if raw_dir is empty."""
    import importlib.util
    gen_path = Path(__file__).resolve().parent / "sample_data_generator.py"
    spec = importlib.util.spec_from_file_location("gen", gen_path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    gen.save_all(str(raw_dir))


if __name__ == "__main__":
    snap = network_snapshot()
    for name, df in snap.items():
        print(f"\n{name.upper()} — shape: {df.shape}")
        print(df.head(3).to_string())
