"""
Main runner — Tyre Manufacturing Supply Chain Optimizer
=======================================================
Executes the full 3-wave strategy pipeline in sequence and
prints a consolidated KPI report to the terminal.

For the interactive dashboard run:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.data_ingestion import network_snapshot
from forecasting.demand_forecasting import (
    build_features, GBMDemandForecaster,
    promotion_impact_report, accuracy_summary,
)
from inventory.meio import (
    calculate_safety_stock, network_rebalance,
    postponement_opportunities, meio_kpis,
)
from logistics.transport_optimizer import (
    route_cost_matrix, optimal_mode,
    modal_shift_savings, logistics_kpis,
)
from energy.power_reliability import (
    uptime_comparison, sustainability_metrics, sofc_roi_model,
)


def run_pipeline():
    print("=" * 70)
    print("  TYRE MANUFACTURING SUPPLY CHAIN OPTIMISER — FULL PIPELINE")
    print("  Gurgaon · Mumbai · Chennai  |  March 2026")
    print("=" * 70)

    # ── WAVE 1: Data ingestion ────────────────────────────────────────────
    print("\n▶ WAVE 1: Data Centralisation & Unified Digital Twin")
    snap = network_snapshot()
    sales, inventory, logistics, energy = (
        snap["sales"], snap["inventory"], snap["logistics"], snap["energy"]
    )

    # ── WAVE 2: Demand forecasting ────────────────────────────────────────
    print("\n▶ WAVE 2a: AI Demand Forecasting (GBM)")
    features = build_features(sales)
    forecaster = GBMDemandForecaster()
    forecaster.fit(features)
    forecasts = forecaster.predict(features, horizon=30)

    print("\n  30-day forecast summary (units) by factory × segment:")
    print(
        forecasts.groupby(["factory", "segment"])["forecast_units"]
        .sum()
        .to_string()
    )

    acc = accuracy_summary(forecaster)
    overall_mape = acc[acc["factory"] != "ALL"]["cv_mape"].mean()
    print(f"\n  Overall CV MAPE : {overall_mape:.1%}")

    promo = promotion_impact_report(sales)
    if not promo.empty:
        top = promo.iloc[0]
        print(f"  Top promo lift  : {top['promotion']} → +{top['lift_pct']:.1f}% "
              f"in {top['segment']} (Month {int(top['month'])})")

    # ── WAVE 2: MEIO ──────────────────────────────────────────────────────
    print("\n▶ WAVE 2b: Multi-Echelon Inventory Optimisation (MEIO)")
    ss = calculate_safety_stock(inventory)
    kpis = meio_kpis(inventory, ss)

    print(f"  Stockout-risk nodes : {kpis['at_stockout_risk']} "
          f"/ {kpis['total_sku_factory_combinations']}")
    print(f"  Overstock nodes     : {kpis['overstock_nodes']}")
    print(f"  Avg days of stock   : {kpis['avg_days_of_stock']:.1f} days")
    print(f"  Inventory reduction : {kpis['inventory_reduction_pct']:.1f}% "
          f"(target {kpis['target_reduction_pct']}%)")
    print(f"  Annual carrying saving : ₹{kpis['annual_carrying_saving_inr']:,.0f}")

    transfers = network_rebalance(ss)
    if not transfers.empty:
        print(f"\n  Network rebalance: {len(transfers)} transfer(s) recommended")
        print(
            transfers[["sku", "from_factory", "to_factory",
                        "transfer_qty", "recommended_mode"]].head(5).to_string(index=False)
        )

    postpone = postponement_opportunities(ss)
    if not postpone.empty:
        total_save = postpone["saving_units"].sum()
        print(f"\n  Postponement opportunities: {len(postpone)} carcass groups, "
              f"saving ~{total_save} units of finished-goods buffer")

    # ── WAVE 3: Logistics ─────────────────────────────────────────────────
    print("\n▶ WAVE 3a: Logistics & Transport Optimisation")
    log_kpis = logistics_kpis(logistics)
    print(f"  Total freight spend : ₹{log_kpis['total_freight_spend_inr']:,.0f}")
    print(f"  Road share          : {log_kpis['road_mode_share_pct']:.0f}%")
    print(f"  Rail share          : {log_kpis['rail_mode_share_pct']:.0f}%")
    print(f"  Modal shift saving  : ₹{log_kpis['potential_modal_shift_saving_inr']:,.0f} "
          f"({log_kpis['target_saving_pct']}% target)")

    rcm = route_cost_matrix()
    best = optimal_mode(rcm)
    opt = best[best["recommendation"] == "✓ Optimal"][
        ["origin", "destination", "segment", "mode", "cost_per_unit_inr", "saving_pct"]
    ]
    print("\n  Recommended modal mix:")
    print(opt.to_string(index=False))

    # ── WAVE 3: Bloom Energy ──────────────────────────────────────────────
    print("\n▶ WAVE 3b: Bloom Energy SOFC — Power Reliability")
    uptime = uptime_comparison(energy)
    print(uptime[["factory", "grid_uptime_pct", "target_uptime_pct",
                  "annual_scrap_saving_inr", "sofc_priority"]].to_string(index=False))

    roi = sofc_roi_model(energy)
    print("\n  SOFC ROI Summary:")
    print(
        roi[["factory", "capex_inr", "net_annual_benefit_inr",
             "payback_years", "recommendation"]].to_string(index=False)
    )

    sus = sustainability_metrics(energy)
    total_co2 = sus["co2_saved_kg"].sum() / 1_000
    total_water = sus["water_saved_kl"].sum()
    print(f"\n  CO₂ saving (if all SOFC) : {total_co2:,.1f} tonnes/yr")
    print(f"  Water saving             : {total_water:,.1f} kL/yr")

    # ── Final scorecard ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STRATEGIC SCORECARD")
    print("=" * 70)
    scorecard = [
        ("Forecast Accuracy Improvement", f"{(0.12 - overall_mape) / 0.12 * 100:.1f}%",
         f"Target {FINANCIALS['target_forecast_accuracy_improvement_pct']}%"),
        ("Inventory Reduction",
         f"{kpis['inventory_reduction_pct']:.1f}%",
         f"Target {kpis['target_reduction_pct']}%"),
        ("Annual Carrying Cost Saving",
         f"₹{kpis['annual_carrying_saving_inr']:,.0f}", "—"),
        ("Logistics Cost Saving Potential",
         f"₹{log_kpis['potential_modal_shift_saving_inr']:,.0f}",
         f"Target {log_kpis['target_saving_pct']}% cut"),
        ("SOFC Annual Scrap Saving",
         f"₹{uptime['annual_scrap_saving_inr'].sum():,.0f}", "—"),
        ("Production Yield (SOFC sites)",
         f"{BLOOM_SOFC['uptime_guarantee'] * 100:.2f}%",
         f"Target {FINANCIALS['target_production_yield_pct']}%"),
    ]
    for metric, value, target in scorecard:
        print(f"  {metric:<40} {value:<20} {target}")

    print("\n✅ Pipeline complete. Launch dashboard: streamlit run dashboard/app.py")
    print("=" * 70)


if __name__ == "__main__":
    from config.settings import BLOOM_SOFC, FINANCIALS
    run_pipeline()
