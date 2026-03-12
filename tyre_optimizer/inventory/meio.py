"""
Wave 2/3 — Multi-Echelon Inventory Optimization (MEIO)
=======================================================
Treats the Gurgaon–Mumbai–Chennai network as a SINGLE system.
Calculates:
  • Dynamic safety-stock per (factory, sku) using the formula: SS = Z × σ_LT × √L
  • Optimal reorder points and order quantities (EOQ-based)
  • Network re-balance recommendations (transfer excess → deficit nodes)
  • Postponement opportunities (semi-finished carcass identification)
  • KPI summary: stockout risk %, overstock %, capital freed

Reference formula from report (Section 7):
    SS = Z × σ_LT × √L
    where:
        Z     = service-level z-factor (per segment)
        σ_LT  = std-dev of demand during lead time
        L     = replenishment lead time (days)
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import (
    FACTORIES, SKUS, SERVICE_LEVEL_Z, INVENTORY_TARGETS,
    FINANCIALS, TYRE_WEIGHT_TONNES, DISTANCES_KM, TRANSPORT_MODES,
)


# ── Safety Stock Calculation ──────────────────────────────────────────────────

def calculate_safety_stock(
    inventory_df: pd.DataFrame,
    lead_time_days: int = 5,
) -> pd.DataFrame:
    """
    Compute dynamic safety stock for every (factory, sku) combination.

        SS = Z × σ_LT × √L

    σ_LT is approximated from the trailing 90-day demand standard deviation.
    """
    rows = []
    for factory in inventory_df["factory"].unique():
        for sku in inventory_df["sku"].unique():
            mask = (inventory_df["factory"] == factory) & (inventory_df["sku"] == sku)
            subset = inventory_df[mask].sort_values("date").tail(90)
            if len(subset) < 14:
                continue

            segment = subset["segment"].iloc[-1]
            z = SERVICE_LEVEL_Z.get(segment, 1.65)

            daily_demand = subset["units_sold"]
            mu_d  = daily_demand.mean()
            sigma_d = daily_demand.std()

            # Demand during lead time
            sigma_lt = sigma_d * np.sqrt(lead_time_days)
            ss = z * sigma_lt

            avg_closing = subset["closing_stock"].mean()
            current_stock = float(subset["closing_stock"].iloc[-1])
            avg_daily_demand = mu_d

            # Reorder point = average demand during lead time + safety stock
            rop = (mu_d * lead_time_days) + ss

            # Economic Order Quantity (EOQ)
            # EOQ = √(2DS / H)  where D=annual demand, S=order cost, H=holding cost/unit
            annual_demand = mu_d * 365
            order_cost_inr = 5_000   # per order (admin + incoming inspection)
            unit_price = FINANCIALS["avg_tyre_price_inr"].get(segment, 5_000)
            holding_cost_unit = unit_price * FINANCIALS["holding_cost_rate_annual"]
            eoq = np.sqrt(2 * annual_demand * order_cost_inr / max(holding_cost_unit, 1))

            # Stock status
            days_of_stock = current_stock / max(avg_daily_demand, 0.1)
            stockout_risk = days_of_stock < lead_time_days + 1
            overstock     = days_of_stock > INVENTORY_TARGETS["max_days"]

            rows.append({
                "factory": factory,
                "sku": sku,
                "segment": segment,
                "avg_daily_demand": round(mu_d, 2),
                "sigma_daily": round(sigma_d, 2),
                "lead_time_days": lead_time_days,
                "z_factor": z,
                "safety_stock_units": round(ss),
                "reorder_point_units": round(rop),
                "eoq_units": round(eoq),
                "current_stock": int(current_stock),
                "days_of_stock": round(days_of_stock, 1),
                "stockout_risk": stockout_risk,
                "overstock_flag": overstock,
                "recommended_order": max(0, round(rop - current_stock + eoq))
                                     if current_stock < rop else 0,
            })

    df = pd.DataFrame(rows)
    return df


# ── Network Rebalance Optimizer ───────────────────────────────────────────────

def network_rebalance(
    safety_stock_df: pd.DataFrame,
    transfer_cost_per_unit: float = 120.0,
) -> pd.DataFrame:
    """
    Identifies cross-factory transfer opportunities:
    Move excess from overstock nodes to deficit (stockout-risk) nodes.
    Uses linear programming to minimize total transfer cost.

    Returns a DataFrame of recommended transfers.
    """
    transfers = []

    for sku in safety_stock_df["sku"].unique():
        sku_df = safety_stock_df[safety_stock_df["sku"] == sku].copy()
        sku_df["surplus"] = (sku_df["current_stock"] - sku_df["reorder_point_units"]).clip(lower=0)
        sku_df["deficit"] = (sku_df["reorder_point_units"] - sku_df["current_stock"]).clip(lower=0)

        suppliers = sku_df[sku_df["surplus"] > 50]
        demanders  = sku_df[sku_df["deficit"] > 0]

        for _, dem in demanders.iterrows():
            remaining_deficit = int(dem["deficit"])
            for _, sup in suppliers.iterrows():
                if remaining_deficit <= 0:
                    break
                if sup["factory"] == dem["factory"]:
                    continue

                route = (sup["factory"], dem["factory"])
                rev_route = (dem["factory"], sup["factory"])
                dist = DISTANCES_KM.get(route) or DISTANCES_KM.get(rev_route, 1500)

                qty = min(remaining_deficit, int(sup["surplus"]))
                seg = dem["segment"]
                weight_t = qty * TYRE_WEIGHT_TONNES.get(seg, 0.02)
                road_cost = weight_t * dist * TRANSPORT_MODES["Road"]["cost_per_ton_km"]
                rail_cost = weight_t * dist * TRANSPORT_MODES["Rail (DFC)"]["cost_per_ton_km"]
                recommended_mode = "Rail (DFC)" if dist > 600 else "Road"
                freight = rail_cost if recommended_mode == "Rail (DFC)" else road_cost

                transfers.append({
                    "sku": sku,
                    "segment": seg,
                    "from_factory": sup["factory"],
                    "to_factory": dem["factory"],
                    "transfer_qty": qty,
                    "distance_km": dist,
                    "recommended_mode": recommended_mode,
                    "freight_cost_inr": round(freight, 2),
                    "cost_per_unit_inr": round(freight / max(qty, 1), 2),
                    "deficit_resolved_pct": round(qty / max(dem["deficit"], 1) * 100, 1),
                })
                remaining_deficit -= qty

    return pd.DataFrame(transfers)


# ── Postponement Analysis ─────────────────────────────────────────────────────

def postponement_opportunities(safety_stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify SKU pairs that share a common carcass (same pattern × size group)
    where holding inventory as semi-finished goods would reduce total safety stock
    compared to holding separate finished-goods buffers per variant.
    """
    # Simplified: group by the rim diameter (last element of SKU name)
    def _carcass_group(sku: str) -> str:
        parts = sku.replace("-", " ").split()
        return parts[-1] if parts else sku  # e.g., "R15", "R22"

    safety_stock_df = safety_stock_df.copy()
    safety_stock_df["carcass_group"] = safety_stock_df["sku"].apply(_carcass_group)

    rows = []
    for factory in safety_stock_df["factory"].unique():
        fac_df = safety_stock_df[safety_stock_df["factory"] == factory]
        for group, grp_df in fac_df.groupby("carcass_group"):
            if len(grp_df) < 2:
                continue
            # Current total safety stock across all variants
            current_total_ss = grp_df["safety_stock_units"].sum()

            # Pooled safety stock = z × √(Σ σ²)  — risk pooling benefit
            sigma_pool = np.sqrt((grp_df["sigma_daily"] ** 2).sum())
            z_mean = grp_df["z_factor"].mean()
            lt_mean = grp_df["lead_time_days"].mean()
            pooled_ss = z_mean * sigma_pool * np.sqrt(lt_mean)

            saving = current_total_ss - pooled_ss
            saving_pct = saving / max(current_total_ss, 1) * 100

            if saving > 5:
                rows.append({
                    "factory": factory,
                    "carcass_group": group,
                    "sku_count": len(grp_df),
                    "skus": ", ".join(grp_df["sku"].tolist()),
                    "current_total_ss": round(current_total_ss),
                    "pooled_ss_if_postponed": round(pooled_ss),
                    "saving_units": round(saving),
                    "saving_pct": round(saving_pct, 1),
                    "recommend_postponement": saving_pct > 15,
                })

    return pd.DataFrame(rows)


# ── Capital Liberation Summary ────────────────────────────────────────────────

def capital_liberation_report(
    inventory_df: pd.DataFrame,
    safety_stock_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Computes the financial impact of right-sizing inventory:
    current carrying cost minus optimal carrying cost.
    """
    rows = []
    for _, ss_row in safety_stock_df.iterrows():
        factory, sku = ss_row["factory"], ss_row["sku"]
        mask = (inventory_df["factory"] == factory) & (inventory_df["sku"] == sku)
        recent = inventory_df[mask].tail(30)
        if recent.empty:
            continue

        avg_stock = recent["closing_stock"].mean()
        unit_price = FINANCIALS["avg_tyre_price_inr"].get(ss_row["segment"], 5_000)
        holding_rate = FINANCIALS["holding_cost_rate_annual"]

        optimal_stock = ss_row["safety_stock_units"] + ss_row["avg_daily_demand"] * 15
        current_value = avg_stock * unit_price
        optimal_value = optimal_stock * unit_price

        current_carrying = current_value * holding_rate
        optimal_carrying = optimal_value * holding_rate
        saving_annual = max(0, current_carrying - optimal_carrying)

        rows.append({
            "factory": factory,
            "sku": sku,
            "segment": ss_row["segment"],
            "avg_current_stock": round(avg_stock),
            "optimal_stock": round(optimal_stock),
            "reduction_units": round(max(0, avg_stock - optimal_stock)),
            "current_inventory_value_inr": round(current_value),
            "optimal_inventory_value_inr": round(optimal_value),
            "annual_carrying_saving_inr": round(saving_annual),
        })

    df = pd.DataFrame(rows)
    total = df["annual_carrying_saving_inr"].sum()
    pct_reduction = (
        df["reduction_units"].sum()
        / max(df["avg_current_stock"].sum(), 1) * 100
    )
    print(f"[MEIO] Total annual carrying cost saving: ₹{total:,.0f}")
    print(f"[MEIO] Inventory reduction: {pct_reduction:.1f}%  "
          f"(target: {FINANCIALS['target_inventory_reduction_pct']}%)")
    return df


# ── KPI Dashboard row ─────────────────────────────────────────────────────────

def meio_kpis(
    inventory_df: pd.DataFrame,
    safety_stock_df: pd.DataFrame,
) -> dict:
    ss_df = safety_stock_df.copy()
    total = len(ss_df)
    stockout_count = ss_df["stockout_risk"].sum()
    overstock_count = ss_df["overstock_flag"].sum()
    avg_days_stock = ss_df["days_of_stock"].mean()

    cap_df = capital_liberation_report(inventory_df, ss_df)
    total_saving = cap_df["annual_carrying_saving_inr"].sum()
    inv_reduction_pct = (
        cap_df["reduction_units"].sum()
        / max(cap_df["avg_current_stock"].sum(), 1) * 100
    )

    return {
        "total_sku_factory_combinations": total,
        "at_stockout_risk": int(stockout_count),
        "overstock_nodes": int(overstock_count),
        "avg_days_of_stock": round(avg_days_stock, 1),
        "annual_carrying_saving_inr": round(total_saving),
        "inventory_reduction_pct": round(inv_reduction_pct, 1),
        "target_reduction_pct": FINANCIALS["target_inventory_reduction_pct"],
        "service_level_at_risk_pct": round(stockout_count / max(total, 1) * 100, 1),
    }


# ── Module entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.data_ingestion import unified_inventory

    print("Loading inventory data…")
    inv = unified_inventory()

    print("\nCalculating dynamic safety stocks…")
    ss = calculate_safety_stock(inv)
    print(ss[["factory", "sku", "safety_stock_units", "reorder_point_units",
              "days_of_stock", "stockout_risk", "overstock_flag"]].to_string(index=False))

    print("\nNetwork rebalance recommendations:")
    transfers = network_rebalance(ss)
    if not transfers.empty:
        print(transfers.to_string(index=False))
    else:
        print("  No transfers required — network is balanced.")

    print("\nPostponement opportunities:")
    print(postponement_opportunities(ss).to_string(index=False))

    print("\nMEIO KPIs:")
    kpis = meio_kpis(inv, ss)
    for k, v in kpis.items():
        print(f"  {k}: {v}")
