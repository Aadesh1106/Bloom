"""
Generates realistic synthetic data for all three factory nodes.
Simulates 3 years of daily sales, inventory, production, and logistics records
across Gurgaon, Chennai, and Mumbai — replicating the Excel-silo problem
before the unified data layer is applied.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import (
    FACTORIES, SKUS, SEGMENTS, SEASONAL_INDICES, PROMOTION_LIFT,
    FINANCIALS, TYRE_WEIGHT_TONNES,
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── helpers ───────────────────────────────────────────────────────────────────

def _segment_for_sku(sku: str) -> str:
    if sku.startswith("CAR") or sku.startswith("SUV"):
        return "Car & SUV"
    if sku.startswith("VLT"):
        return "Van & Light Truck"
    return "Truck & Bus"


def _base_demand(factory: str, sku: str) -> float:
    """Mean daily units per SKU per factory (baseline, no seasonality)."""
    base_map = {
        ("Gurgaon",  "CAR-185-65R15"):  38,
        ("Gurgaon",  "CAR-205-55R16"):  32,
        ("Gurgaon",  "SUV-235-65R17"):  22,
        ("Gurgaon",  "VLT-195-80R15"):  14,
        ("Gurgaon",  "VLT-215-75R17"):  10,
        ("Gurgaon",  "TRB-295-80R22"):   8,
        ("Gurgaon",  "TRB-315-80R22"):   5,
        ("Mumbai",   "CAR-185-65R15"):  28,
        ("Mumbai",   "CAR-205-55R16"):  25,
        ("Mumbai",   "SUV-235-65R17"):  30,
        ("Mumbai",   "VLT-195-80R15"):  40,
        ("Mumbai",   "VLT-215-75R17"):  35,
        ("Mumbai",   "TRB-295-80R22"):  20,
        ("Mumbai",   "TRB-315-80R22"):  18,
        ("Chennai",  "CAR-185-65R15"):  18,
        ("Chennai",  "CAR-205-55R16"):  15,
        ("Chennai",  "SUV-235-65R17"):  12,
        ("Chennai",  "VLT-195-80R15"):  20,
        ("Chennai",  "VLT-215-75R17"):  18,
        ("Chennai",  "TRB-295-80R22"):  45,
        ("Chennai",  "TRB-315-80R22"):  38,
    }
    return base_map.get((factory, sku), 10.0)


def generate_sales_data(start: str = "2023-01-01", periods: int = 365 * 3) -> pd.DataFrame:
    """Daily sales records for all factories × SKUs."""
    dates = pd.date_range(start=start, periods=periods, freq="D")
    records = []

    promotions = _build_promotion_calendar(dates)

    for date in dates:
        seasonal = SEASONAL_INDICES[date.month]
        promo = promotions.get(date, "None")
        promo_lift = PROMOTION_LIFT[promo]

        for factory in FACTORIES:
            for sku in SKUS:
                base = _base_demand(factory, sku)
                demand = base * (seasonal + promo_lift) * np.random.lognormal(0, 0.12)
                demand = max(0, round(demand))

                records.append({
                    "date": date,
                    "factory": factory,
                    "sku": sku,
                    "segment": _segment_for_sku(sku),
                    "channel": np.random.choice(["OEM", "Replacement"], p=[0.35, 0.65]),
                    "units_sold": demand,
                    "promotion": promo,
                    "seasonal_index": seasonal,
                    "price_inr": FINANCIALS["avg_tyre_price_inr"][_segment_for_sku(sku)]
                               * np.random.uniform(0.92, 1.08),
                })

    return pd.DataFrame(records)


def generate_inventory_data(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Simulates daily closing stock at each factory (simple FIFO model)."""
    records = []
    # opening stock = 30 days of average daily demand
    stock = {}
    for factory in FACTORIES:
        for sku in SKUS:
            stock[(factory, sku)] = round(_base_demand(factory, sku) * 30)

    daily_production = {}
    for factory in FACTORIES:
        for sku in SKUS:
            daily_production[(factory, sku)] = round(_base_demand(factory, sku) * 1.05)

    dates = sorted(sales_df["date"].unique())
    for date in dates:
        day_sales = sales_df[sales_df["date"] == date]
        for factory in FACTORIES:
            for sku in SKUS:
                sold = int(
                    day_sales.loc[
                        (day_sales["factory"] == factory) & (day_sales["sku"] == sku),
                        "units_sold"
                    ].sum()
                )
                prod = daily_production[(factory, sku)]
                opening = stock[(factory, sku)]
                closing = max(0, opening + prod - sold)
                stock[(factory, sku)] = closing

                records.append({
                    "date": date,
                    "factory": factory,
                    "sku": sku,
                    "segment": _segment_for_sku(sku),
                    "opening_stock": opening,
                    "production_qty": prod,
                    "units_sold": sold,
                    "closing_stock": closing,
                    "stockout_flag": int(closing == 0),
                    "days_of_stock": round(closing / max(sold, 1), 1),
                })

    return pd.DataFrame(records)


def generate_logistics_data(start: str = "2023-01-01", periods: int = 365 * 3) -> pd.DataFrame:
    """Simulated inter-factory shipment records."""
    from config.settings import DISTANCES_KM, TRANSPORT_MODES
    dates = pd.date_range(start=start, periods=periods, freq="D")
    routes = list(DISTANCES_KM.keys())
    records = []

    for date in dates:
        n_shipments = np.random.randint(2, 6)
        for _ in range(n_shipments):
            route = routes[np.random.randint(len(routes))]
            mode = np.random.choice(
                ["Road", "Rail (DFC)"], p=[0.72, 0.28]  # current state: road heavy
            )
            sku = SKUS[np.random.randint(len(SKUS))]
            qty = np.random.randint(50, 500)
            seg = _segment_for_sku(sku)
            weight_t = qty * TYRE_WEIGHT_TONNES[seg]
            dist = DISTANCES_KM[route]
            cost = weight_t * dist * TRANSPORT_MODES[mode]["cost_per_ton_km"]
            transit_days = round(dist / (TRANSPORT_MODES[mode]["avg_speed_kmph"] * 8))

            records.append({
                "date": date,
                "origin": route[0],
                "destination": route[1],
                "sku": sku,
                "segment": seg,
                "transport_mode": mode,
                "quantity": qty,
                "weight_tonnes": round(weight_t, 3),
                "distance_km": dist,
                "freight_cost_inr": round(cost, 2),
                "transit_days": transit_days,
                "co2_factor": TRANSPORT_MODES[mode]["carbon_factor"],
            })

    return pd.DataFrame(records)


def generate_energy_data(start: str = "2023-01-01", periods: int = 365 * 3) -> pd.DataFrame:
    """Simulates hourly power events at each factory."""
    from config.settings import BLOOM_SOFC
    dates = pd.date_range(start=start, periods=periods * 24, freq="h")
    records = []

    for factory, cfg in FACTORIES.items():
        grid_rel = cfg["grid_reliability_score"]
        for ts in dates:
            grid_ok = np.random.random() < grid_rel
            sofc_deployed = factory == "Chennai"  # SOFC at most vulnerable site
            power_ok = grid_ok or sofc_deployed

            press_count = BLOOM_SOFC["presses_per_factory"]
            active_presses = press_count if power_ok else 0
            batch_scrapped = 0 if power_ok else np.random.randint(0, 3)

            records.append({
                "timestamp": ts,
                "factory": factory,
                "grid_available": int(grid_ok),
                "sofc_deployed": int(sofc_deployed),
                "power_available": int(power_ok),
                "active_presses": active_presses,
                "batches_scrapped": batch_scrapped,
                "scrap_cost_inr": batch_scrapped * BLOOM_SOFC["scrap_cost_per_batch_inr"],
                "power_kw": active_presses * BLOOM_SOFC["curing_press_power_kw"],
            })

    df = pd.DataFrame(records)
    # Downsample to daily for storage efficiency
    df_daily = (
        df.groupby(["factory", df["timestamp"].dt.date])
          .agg(
              grid_uptime_pct=("grid_available", "mean"),
              sofc_deployed=("sofc_deployed", "first"),
              power_uptime_pct=("power_available", "mean"),
              avg_active_presses=("active_presses", "mean"),
              total_batches_scrapped=("batches_scrapped", "sum"),
              total_scrap_cost_inr=("scrap_cost_inr", "sum"),
              total_energy_kwh=("power_kw", "sum"),
          )
          .reset_index()
          .rename(columns={"timestamp": "date"})
    )
    return df_daily


def _build_promotion_calendar(dates) -> dict:
    promo_map = {}
    for date in dates:
        # Diwali window: Oct 15 – Nov 15
        if date.month == 10 and date.day >= 15:
            promo_map[date] = np.random.choice(
                ["Cash Discount", "Combo Offer", "None"], p=[0.4, 0.35, 0.25]
            )
        elif date.month == 11 and date.day <= 15:
            promo_map[date] = np.random.choice(
                ["Cash Discount", "Extended Warranty", "None"], p=[0.3, 0.3, 0.4]
            )
        # Republic Day / Independence Day spikes
        elif (date.month == 1 and date.day == 26) or (date.month == 8 and date.day == 15):
            promo_map[date] = "Cash Discount"
        else:
            promo_map[date] = "None"
    return promo_map


def save_all(output_dir: str = "data/raw") -> None:
    """Generate and save all datasets as CSV files (simulating Excel silos)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Generating sales data (3 years × 3 factories × 7 SKUs)…")
    sales = generate_sales_data()
    # Save separate CSV per factory — simulating legacy Excel silos
    for factory in FACTORIES:
        sales[sales["factory"] == factory].to_csv(
            out / f"sales_{factory.lower()}.csv", index=False
        )
    print(f"  Saved {len(sales):,} sales records.")

    print("Generating inventory data…")
    inventory = generate_inventory_data(sales)
    for factory in FACTORIES:
        inventory[inventory["factory"] == factory].to_csv(
            out / f"inventory_{factory.lower()}.csv", index=False
        )
    print(f"  Saved {len(inventory):,} inventory records.")

    print("Generating logistics data…")
    logistics = generate_logistics_data()
    logistics.to_csv(out / "logistics_shipments.csv", index=False)
    print(f"  Saved {len(logistics):,} shipment records.")

    print("Generating energy/power data…")
    energy = generate_energy_data()
    energy.to_csv(out / "energy_events.csv", index=False)
    print(f"  Saved {len(energy):,} energy records.")

    print("All datasets saved to:", out.resolve())


if __name__ == "__main__":
    save_all()
