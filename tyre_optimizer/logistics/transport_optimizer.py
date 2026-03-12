"""
Wave 3 — Logistics & Transportation Optimizer
==============================================
Implements:
  • Modal shift analysis: Road vs. Rail (DFC) vs. Sea
  • Route-cost matrix for all Gurgaon ↔ Mumbai ↔ Chennai pairs
  • Load factor optimization (max payload utilization)
  • Carbon footprint calculator
  • KPI: cost per unit, empty-miles ratio, CO₂ index
  • Recommended modal split to achieve 10-20 % logistics cost reduction
"""

import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import (
    FACTORIES, TRANSPORT_MODES, DISTANCES_KM,
    TYRE_WEIGHT_TONNES, FINANCIALS, SEGMENTS,
)


# ── Route cost matrix ─────────────────────────────────────────────────────────

def route_cost_matrix() -> pd.DataFrame:
    """
    Build a full cost + transit-time matrix for every origin-destination pair
    across all three transport modes.
    """
    rows = []
    for (origin, dest), dist_km in DISTANCES_KM.items():
        for mode, params in TRANSPORT_MODES.items():
            for segment in SEGMENTS:
                weight_per_unit = TYRE_WEIGHT_TONNES[segment]
                # Assume a standard FTL consignment of 300 units
                consignment_qty = 300
                total_weight_t  = consignment_qty * weight_per_unit
                freight_cost    = total_weight_t * dist_km * params["cost_per_ton_km"]
                cost_per_unit   = freight_cost / consignment_qty
                transit_days    = max(1, round(dist_km / (params["avg_speed_kmph"] * 10)))

                rows.append({
                    "origin": origin,
                    "destination": dest,
                    "mode": mode,
                    "segment": segment,
                    "distance_km": dist_km,
                    "consignment_qty": consignment_qty,
                    "total_weight_tonnes": round(total_weight_t, 3),
                    "total_freight_inr": round(freight_cost, 0),
                    "cost_per_unit_inr": round(cost_per_unit, 2),
                    "transit_days": transit_days,
                    "reliability_score": params["reliability"],
                    "co2_relative": params["carbon_factor"],
                })

    return pd.DataFrame(rows)


def optimal_mode(route_cost_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each route × segment, select the recommended mode that minimises
    cost while respecting a minimum reliability threshold of 0.85.
    """
    filtered = route_cost_df[route_cost_df["reliability_score"] >= 0.85]
    idx = filtered.groupby(["origin", "destination", "segment"])["cost_per_unit_inr"].idxmin()
    best = filtered.loc[idx].copy()
    best["recommendation"] = "✓ Optimal"

    # Always include road as comparison baseline
    road = route_cost_df[route_cost_df["mode"] == "Road"].copy()
    road["recommendation"] = "Baseline (Road)"

    merged = pd.concat([best, road], ignore_index=True).drop_duplicates(
        subset=["origin", "destination", "segment", "mode"]
    )

    # Compute savings vs road
    road_costs = road.set_index(["origin", "destination", "segment"])["cost_per_unit_inr"]
    merged["saving_vs_road_inr"] = merged.apply(
        lambda r: road_costs.get((r["origin"], r["destination"], r["segment"]), np.nan)
                  - r["cost_per_unit_inr"],
        axis=1,
    ).round(2)
    merged["saving_pct"] = (
        merged["saving_vs_road_inr"]
        / road_costs.reindex(
            pd.MultiIndex.from_frame(
                merged[["origin", "destination", "segment"]]
            )
        ).values
        * 100
    ).round(1)

    return merged.sort_values(["origin", "destination", "segment", "cost_per_unit_inr"])


# ── Load utilisation analyser ─────────────────────────────────────────────────

MAX_PAYLOAD_TONNES = {
    "Road": 25.0,     # standard FTL truck
    "Rail (DFC)": 60.0,  # DFC wagon
    "Sea": 200.0,
}


def load_utilisation_report(logistics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify under-loaded shipments and suggest consolidation.
    Returns per-route-mode utilisation stats + "empty miles" proxy.
    """
    df = logistics_df.copy()
    df["max_payload_t"] = df["transport_mode"].map(MAX_PAYLOAD_TONNES).fillna(25.0)
    df["utilisation_pct"] = (
        df["weight_tonnes"] / df["max_payload_t"] * 100
    ).clip(0, 100).round(1)
    df["empty_miles_proxy"] = ((100 - df["utilisation_pct"]) / 100 * df["distance_km"]).round(1)

    agg = (
        df.groupby(["origin", "destination", "transport_mode"])
          .agg(
              shipment_count=("quantity", "count"),
              avg_qty_per_load=("quantity", "mean"),
              avg_weight_tonnes=("weight_tonnes", "mean"),
              avg_utilisation_pct=("utilisation_pct", "mean"),
              total_freight_inr=("freight_cost_inr", "sum"),
              avg_empty_km=("empty_miles_proxy", "mean"),
          )
          .reset_index()
    )
    agg["avg_utilisation_pct"] = agg["avg_utilisation_pct"].round(1)
    agg["under_loaded"] = agg["avg_utilisation_pct"] < 70
    agg["consolidation_opportunity"] = agg["under_loaded"] & (agg["shipment_count"] > 5)

    return agg.sort_values("avg_utilisation_pct")


# ── Modal shift savings calculator ───────────────────────────────────────────

def modal_shift_savings(logistics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate annual savings from shifting > 600 km road shipments to DFC rail.
    """
    df = logistics_df.copy()
    long_haul_road = df[
        (df["transport_mode"] == "Road") & (df["distance_km"] > 600)
    ].copy()

    if long_haul_road.empty:
        return pd.DataFrame()

    # Current road cost
    long_haul_road["rail_freight_cost"] = (
        long_haul_road["weight_tonnes"]
        * long_haul_road["distance_km"]
        * TRANSPORT_MODES["Rail (DFC)"]["cost_per_ton_km"]
    )
    long_haul_road["saving_inr"] = (
        long_haul_road["freight_cost_inr"] - long_haul_road["rail_freight_cost"]
    ).clip(lower=0)
    co2_reduction_pct = round(
        (1 - TRANSPORT_MODES["Rail (DFC)"]["carbon_factor"]
             / TRANSPORT_MODES["Road"]["carbon_factor"]) * 100,
        1,
    )
    long_haul_road["co2_reduction_pct"] = co2_reduction_pct

    summary = (
        long_haul_road.groupby(["origin", "destination"])
        .agg(
            road_shipments=("quantity", "count"),
            total_road_freight_inr=("freight_cost_inr", "sum"),
            total_rail_freight_inr=("rail_freight_cost", "sum"),
            total_saving_inr=("saving_inr", "sum"),
            co2_reduction_pct=("co2_reduction_pct", "first"),
        )
        .reset_index()
    )
    summary["saving_pct"] = (
        summary["total_saving_inr"] / summary["total_road_freight_inr"] * 100
    ).round(1)

    total_saving = summary["total_saving_inr"].sum()
    print(f"[Logistics] Modal shift saving potential: ₹{total_saving:,.0f} / year")
    print(f"[Logistics] Target: {FINANCIALS['target_logistics_saving_pct']}% reduction")

    return summary


# ── Carbon footprint report ───────────────────────────────────────────────────

def carbon_report(logistics_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise CO₂ relative index by route and current vs optimal mode."""
    df = logistics_df.copy()
    df["co2_index"] = df["co2_factor"] * df["weight_tonnes"] * df["distance_km"]

    summary = (
        df.groupby(["transport_mode"])
          .agg(
              shipments=("quantity", "count"),
              total_co2_index=("co2_index", "sum"),
              total_freight_inr=("freight_cost_inr", "sum"),
          )
          .reset_index()
    )
    summary["co2_index_pct"] = (
        summary["total_co2_index"] / summary["total_co2_index"].sum() * 100
    ).round(1)

    return summary


# ── Logistics KPIs ────────────────────────────────────────────────────────────

def logistics_kpis(logistics_df: pd.DataFrame) -> dict:
    total_freight = logistics_df["freight_cost_inr"].sum()
    total_units   = logistics_df["quantity"].sum()
    avg_cpu       = total_freight / max(total_units, 1)
    road_pct      = (
        logistics_df[logistics_df["transport_mode"] == "Road"]["quantity"].sum()
        / max(total_units, 1) * 100
    )
    savings_df    = modal_shift_savings(logistics_df)
    potential_saving = savings_df["total_saving_inr"].sum() if not savings_df.empty else 0

    return {
        "total_freight_spend_inr": round(total_freight),
        "total_units_shipped": int(total_units),
        "avg_cost_per_unit_inr": round(avg_cpu, 2),
        "road_mode_share_pct": round(road_pct, 1),
        "rail_mode_share_pct": round(100 - road_pct, 1),
        "potential_modal_shift_saving_inr": round(potential_saving),
        "target_saving_pct": FINANCIALS["target_logistics_saving_pct"],
    }


# ── Module entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data.data_ingestion import unified_logistics

    print("Loading logistics data…")
    log = unified_logistics()

    print("\nRoute Cost Matrix (best modes):")
    rcm = route_cost_matrix()
    best = optimal_mode(rcm)
    print(
        best[best["recommendation"] == "✓ Optimal"][
            ["origin", "destination", "segment", "mode",
             "cost_per_unit_inr", "transit_days", "saving_pct"]
        ].to_string(index=False)
    )

    print("\nLoad Utilisation Report:")
    util = load_utilisation_report(log)
    print(util.to_string(index=False))

    print("\nModal Shift Savings:")
    shifts = modal_shift_savings(log)
    if not shifts.empty:
        print(shifts.to_string(index=False))

    print("\nCarbon Report:")
    print(carbon_report(log).to_string(index=False))

    print("\nLogistics KPIs:")
    for k, v in logistics_kpis(log).items():
        print(f"  {k}: {v}")
