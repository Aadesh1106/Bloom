"""
Wave 3 — Bloom Energy SOFC Reliability & ROI Model
====================================================
Models the impact of deploying Solid Oxide Fuel Cell (SOFC) Energy Servers
at each factory node and quantifies:

  • Production yield protected (scrap batches prevented)
  • Curing-press uptime improvement (grid vs. SOFC-backed)
  • Annual scrap cost savings
  • CO₂ and water savings from SOFC vs. grid power
  • Predictive maintenance alert scoring (C3 AI Reliability Suite proxy)
  • Full SOFC ROI model: payback period, NPV, IRR

All parameters sourced from the strategy document Section 9.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import BLOOM_SOFC, FACTORIES, FINANCIALS


# ── Uptime comparison ─────────────────────────────────────────────────────────

def uptime_comparison(energy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare grid-only uptime vs. SOFC-backed uptime per factory.
    Shows the difference in production yield and scrap cost.
    """
    rows = []
    for factory, cfg in FACTORIES.items():
        fac_df = energy_df[energy_df["factory"] == factory]
        if fac_df.empty:
            continue

        grid_uptime = fac_df["grid_uptime_pct"].mean()
        power_uptime = fac_df["power_uptime_pct"].mean()
        sofc_deployed = bool(fac_df["sofc_deployed"].iloc[0])

        total_batches_scrapped = fac_df["total_batches_scrapped"].sum()
        total_scrap_cost = fac_df["total_scrap_cost_inr"].sum()

        # Theoretical yield if SOFC were deployed
        sofc_uptime = BLOOM_SOFC["uptime_guarantee"]
        prevented_outage_hrs = max(0, sofc_uptime - grid_uptime) * 24 * len(fac_df)
        batches_per_hr = BLOOM_SOFC["presses_per_factory"] / 24  # crude estimate
        batches_saved = prevented_outage_hrs * batches_per_hr
        cost_saved = batches_saved * BLOOM_SOFC["scrap_cost_per_batch_inr"]

        rows.append({
            "factory": factory,
            "grid_uptime_pct": round(grid_uptime * 100, 2),
            "power_uptime_pct": round(power_uptime * 100, 2),
            "sofc_deployed": sofc_deployed,
            "target_uptime_pct": round(BLOOM_SOFC["uptime_guarantee"] * 100, 2),
            "uptime_gap_pp": round((BLOOM_SOFC["uptime_guarantee"] - grid_uptime) * 100, 2),
            "batches_scrapped_ytd": int(total_batches_scrapped),
            "scrap_cost_inr_ytd": round(total_scrap_cost),
            "batches_saveable_by_sofc": round(batches_saved),
            "annual_scrap_saving_inr": round(cost_saved),
            "grid_reliability_score": cfg["grid_reliability_score"],
            "sofc_priority": "HIGH" if cfg["grid_reliability_score"] < 0.85 else "MEDIUM",
        })

    return pd.DataFrame(rows).sort_values("grid_reliability_score")


# ── Energy & sustainability metrics ──────────────────────────────────────────

def sustainability_metrics(energy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate CO₂ reduction and water savings from deploying SOFCs,
    aligned with Bloom Energy's ESG benchmarks from the strategy document.
    """
    rows = []
    for factory in energy_df["factory"].unique():
        fac_df = energy_df[energy_df["factory"] == factory]
        total_kwh = fac_df["total_energy_kwh"].sum()

        # CO₂ savings vs Indian grid (baseline ~0.82 kg CO₂/kWh)
        grid_co2_kg = total_kwh * 0.82
        sofc_co2_kg = total_kwh * 0.82 * (1 - BLOOM_SOFC["emission_reduction_pct"])
        co2_saved_kg = grid_co2_kg - sofc_co2_kg

        # Water savings (Bloom benchmark: 4.2 L/kWh vs thermal plants)
        water_saved_liters = total_kwh * BLOOM_SOFC["water_saved_liters_per_kwh"]

        rows.append({
            "factory": factory,
            "total_energy_kwh": round(total_kwh),
            "grid_co2_kg": round(grid_co2_kg),
            "sofc_co2_kg": round(sofc_co2_kg),
            "co2_saved_kg": round(co2_saved_kg),
            "co2_saved_pct": round(BLOOM_SOFC["emission_reduction_pct"] * 100, 1),
            "water_saved_liters": round(water_saved_liters),
            "water_saved_kl": round(water_saved_liters / 1000, 1),
        })

    return pd.DataFrame(rows)


# ── Predictive maintenance scoring (C3 AI Reliability Suite proxy) ────────────

def predictive_maintenance_alerts(energy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates the C3 AI Reliability Suite-style anomaly scoring.
    Detects presses at risk of unplanned downtime using a rolling
    z-score on energy consumption variance.
    """
    df = energy_df.copy().sort_values(["factory", "date"])

    rows = []
    for factory in df["factory"].unique():
        fac_df = df[df["factory"] == factory].copy()

        # Rolling 7-day energy consumption stats
        fac_df["roll_mean_7d"] = fac_df["total_energy_kwh"].rolling(7, min_periods=2).mean()
        fac_df["roll_std_7d"]  = fac_df["total_energy_kwh"].rolling(7, min_periods=2).std()
        fac_df["z_score"]      = (
            (fac_df["total_energy_kwh"] - fac_df["roll_mean_7d"])
            / fac_df["roll_std_7d"].replace(0, np.nan)
        ).fillna(0)

        # Alert if |z| > 2 (>2σ anomaly)
        alerts = fac_df[fac_df["z_score"].abs() > 2].copy()
        alerts["alert_type"] = alerts["z_score"].apply(
            lambda z: "OVERCONSUMPTION" if z > 0 else "UNDERCONSUMPTION"
        )
        alerts["risk_level"] = alerts["z_score"].abs().apply(
            lambda z: "CRITICAL" if z > 3 else "WARNING"
        )
        alerts["factory"] = factory
        rows.append(
            alerts[["date", "factory", "total_energy_kwh", "z_score",
                     "alert_type", "risk_level"]]
        )

    if rows:
        return pd.concat(rows, ignore_index=True).sort_values(["factory", "date"])
    return pd.DataFrame()


# ── SOFC ROI model ────────────────────────────────────────────────────────────

SOFC_CAPEX_PER_KW = 3_500          # USD/kW (Industry benchmark)
USD_TO_INR        = 83.5

def sofc_roi_model(
    energy_df: pd.DataFrame,
    discount_rate: float = 0.10,
    project_years: int = 10,
) -> pd.DataFrame:
    """
    Full discounted cash flow model for SOFC deployment at each factory.
    Returns: NPV, IRR, payback period, and recommendation.
    """
    uptime_df = uptime_comparison(energy_df)
    sus_df    = sustainability_metrics(energy_df)

    rows = []
    for factory, cfg in FACTORIES.items():
        press_kw   = BLOOM_SOFC["curing_press_power_kw"] * BLOOM_SOFC["presses_per_factory"]
        capex_usd  = press_kw * SOFC_CAPEX_PER_KW
        capex_inr  = capex_usd * USD_TO_INR

        # Annual benefits
        uptime_row = uptime_df[uptime_df["factory"] == factory]
        scrap_saving = float(uptime_row["annual_scrap_saving_inr"].iloc[0]) if not uptime_row.empty else 0
        opex_inr   = capex_inr * 0.02   # 2% of CAPEX annually

        # Electricity cost saving: SOFC ~15% cheaper than grid in India
        sus_row = sus_df[sus_df["factory"] == factory]
        total_kwh = float(sus_row["total_energy_kwh"].iloc[0]) / 3 if not sus_row.empty else 0  # 1-yr avg
        elec_saving_inr = total_kwh * 0.015 * 9.0   # Rs 9/kWh grid rate, 1.5 Rs saving

        annual_benefit = scrap_saving + elec_saving_inr - opex_inr

        # NPV
        cash_flows = [-capex_inr] + [annual_benefit] * project_years
        npv = sum(cf / (1 + discount_rate) ** t for t, cf in enumerate(cash_flows))

        # Payback
        cumulative = 0
        payback = project_years  # default: full term
        for yr in range(1, project_years + 1):
            cumulative += annual_benefit
            if cumulative >= capex_inr:
                payback = yr
                break

        # IRR (Newton-Raphson approximation)
        irr = _estimate_irr(cash_flows)

        rows.append({
            "factory": factory,
            "press_capacity_kw": press_kw,
            "capex_inr": round(capex_inr),
            "annual_scrap_saving_inr": round(scrap_saving),
            "annual_elec_saving_inr": round(elec_saving_inr),
            "annual_opex_inr": round(opex_inr),
            "net_annual_benefit_inr": round(annual_benefit),
            "npv_inr": round(npv),
            "irr_pct": round(irr * 100, 1) if irr else None,
            "payback_years": payback,
            "recommendation": "DEPLOY" if npv > 0 and payback <= 7 else "REVIEW",
            "priority": cfg["grid_reliability_score"],
        })

    df = pd.DataFrame(rows).sort_values("priority")
    return df


def _estimate_irr(cash_flows: list, guess: float = 0.10, tol: float = 1e-6) -> float:
    """Newton-Raphson IRR estimation."""
    rate = guess
    for _ in range(200):
        npv     = sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows))
        d_npv   = sum(-t * cf / (1 + rate) ** (t + 1) for t, cf in enumerate(cash_flows))
        if abs(d_npv) < tol:
            break
        rate -= npv / d_npv
        if rate <= -1:
            return 0.0
    return rate


# ── Module entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data.data_ingestion import unified_energy

    print("Loading energy data…")
    energy = unified_energy()

    print("\nUptime Comparison (Grid vs. SOFC):")
    print(uptime_comparison(energy).to_string(index=False))

    print("\nSustainability Metrics:")
    print(sustainability_metrics(energy).to_string(index=False))

    print("\nPredictive Maintenance Alerts (last 10):")
    alerts = predictive_maintenance_alerts(energy)
    if not alerts.empty:
        print(alerts.tail(10).to_string(index=False))

    print("\nSOFC ROI Model:")
    print(sofc_roi_model(energy).to_string(index=False))
