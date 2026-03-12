"""
Streamlit Dashboard — Tyre Manufacturing Supply Chain Optimizer
==============================================================
Unified command centre implementing the full strategy:
  Wave 1 → Digital Twin / Unified Data Layer
  Wave 2 → AI Demand Forecasting + MEIO Safety Stock
  Wave 3 → Logistics Optimisation + Bloom Energy SOFC

Run:  streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import FACTORIES, SEGMENTS, FINANCIALS, BLOOM_SOFC
from data.data_ingestion import network_snapshot
from forecasting.demand_forecasting import (
    build_features, GBMDemandForecaster, promotion_impact_report, accuracy_summary,
)
from inventory.meio import (
    calculate_safety_stock, network_rebalance,
    postponement_opportunities, meio_kpis, capital_liberation_report,
)
from logistics.transport_optimizer import (
    route_cost_matrix, optimal_mode, load_utilisation_report,
    modal_shift_savings, carbon_report, logistics_kpis,
)
from energy.power_reliability import (
    uptime_comparison, sustainability_metrics,
    predictive_maintenance_alerts, sofc_roi_model,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tyre Supply Chain Optimizer",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

FACTORY_COLORS = {"Gurgaon": "#1f77b4", "Mumbai": "#ff7f0e", "Chennai": "#2ca02c"}
SEGMENT_COLORS = {
    "Car & SUV": "#636EFA",
    "Van & Light Truck": "#EF553B",
    "Truck & Bus": "#00CC96",
}


# ── Data loading (cached) ─────────────────────────────────────────────────────

@st.cache_data(show_spinner="Building Digital Twin snapshot…")
def load_data():
    snap = network_snapshot()
    return snap["sales"], snap["inventory"], snap["logistics"], snap["energy"]


@st.cache_data(show_spinner="Training GBM demand models…")
def load_forecasts(sales_df):
    features = build_features(sales_df)
    forecaster = GBMDemandForecaster()
    forecaster.fit(features)
    forecasts = forecaster.predict(features, horizon=30)
    promo_report = promotion_impact_report(sales_df)
    acc = accuracy_summary(forecaster)
    return forecasts, promo_report, acc, forecaster


@st.cache_data(show_spinner="Running MEIO optimisation…")
def load_meio(inventory_df):
    ss = calculate_safety_stock(inventory_df)
    transfers = network_rebalance(ss)
    postponement = postponement_opportunities(ss)
    kpis = meio_kpis(inventory_df, ss)
    capital = capital_liberation_report(inventory_df, ss)
    return ss, transfers, postponement, kpis, capital


@st.cache_data(show_spinner="Optimising logistics routes…")
def load_logistics(logistics_df):
    rcm = route_cost_matrix()
    best_modes = optimal_mode(rcm)
    util = load_utilisation_report(logistics_df)
    shifts = modal_shift_savings(logistics_df)
    carbon = carbon_report(logistics_df)
    kpis = logistics_kpis(logistics_df)
    return best_modes, util, shifts, carbon, kpis


@st.cache_data(show_spinner="Modelling SOFC reliability…")
def load_energy(energy_df):
    uptime = uptime_comparison(energy_df)
    sus = sustainability_metrics(energy_df)
    alerts = predictive_maintenance_alerts(energy_df)
    roi = sofc_roi_model(energy_df)
    return uptime, sus, alerts, roi


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Tyre_icon.svg/120px-Tyre_icon.svg.png",
        width=60,
    )
    st.sidebar.title("Tyre SC Optimizer")
    st.sidebar.caption("AI-Driven Supply Chain · Bloom Energy SOFC · MEIO")

    page = st.sidebar.radio(
        "Navigate",
        [
            "📊 Executive Summary",
            "🔮 Demand Forecasting",
            "📦 MEIO Inventory",
            "🚚 Logistics Optimizer",
            "⚡ Bloom Energy (SOFC)",
        ],
    )
    st.sidebar.divider()
    st.sidebar.subheader("Filters")
    factory_filter = st.sidebar.multiselect(
        "Factories", list(FACTORIES.keys()), default=list(FACTORIES.keys())
    )
    segment_filter = st.sidebar.multiselect(
        "Segments", SEGMENTS, default=SEGMENTS
    )
    return page, factory_filter, segment_filter


# ── Executive Summary ─────────────────────────────────────────────────────────

def page_executive_summary(sales_df, inventory_df, logistics_df, energy_df,
                            meio_kpis_dict, log_kpis_dict, uptime_df):
    st.title("📊 Executive Summary — Decision Advantage Dashboard")
    st.caption("Single Version of the Truth across Gurgaon · Mumbai · Chennai")

    # Top KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Carrying Cost Saving / yr",
        f"₹{meio_kpis_dict['annual_carrying_saving_inr']:,.0f}",
        f"Target ₹{FINANCIALS['target_carrying_cost_saving_usd'] * 83:,.0f}",
    )
    c2.metric(
        "Inventory Reduction",
        f"{meio_kpis_dict['inventory_reduction_pct']:.1f}%",
        f"Target {FINANCIALS['target_inventory_reduction_pct']}%",
    )
    c3.metric(
        "Logistics Saving Potential",
        f"₹{log_kpis_dict['potential_modal_shift_saving_inr']:,.0f}",
        f"Target {FINANCIALS['target_logistics_saving_pct']}% cost cut",
    )
    c4.metric(
        "Stockout Risk Nodes",
        f"{meio_kpis_dict['at_stockout_risk']}",
        f"of {meio_kpis_dict['total_sku_factory_combinations']} SKU-nodes",
        delta_color="inverse",
    )
    c5.metric(
        "SOFC Priority Factories",
        uptime_df[uptime_df["sofc_priority"] == "HIGH"]["factory"].count(),
        "require immediate deployment",
        delta_color="inverse",
    )

    st.divider()

    # Monthly sales trend
    st.subheader("Monthly Network Sales Trend")
    monthly = (
        sales_df.groupby(["year", "month", "segment"])["units_sold"]
        .sum()
        .reset_index()
    )
    monthly["period"] = pd.to_datetime(
        monthly[["year", "month"]].assign(day=1)
    )
    fig = px.line(
        monthly, x="period", y="units_sold", color="segment",
        color_discrete_map=SEGMENT_COLORS, markers=False,
        title="Network-wide Monthly Demand by Segment",
        labels={"units_sold": "Units Sold", "period": "Month"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Factory contribution
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Factory Revenue Share")
        rev = sales_df.groupby("factory")["revenue_inr"].sum().reset_index()
        fig2 = px.pie(
            rev, names="factory", values="revenue_inr",
            color="factory", color_discrete_map=FACTORY_COLORS,
            hole=0.45,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Stockout vs. Overstock Events (last 90 days)")
        recent = inventory_df[inventory_df["year"] >= inventory_df["year"].max() - 1]
        events = recent.groupby("factory").agg(
            stockouts=("stockout_flag", "sum"),
            overstock=("overstock_flag", "sum"),
        ).reset_index()
        fig3 = px.bar(
            events.melt(id_vars="factory"),
            x="factory", y="value", color="variable", barmode="group",
            color_discrete_map={"stockouts": "#EF553B", "overstock": "#FFA15A"},
            labels={"value": "Events"},
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Implementation roadmap
    st.subheader("Implementation Roadmap")
    roadmap = pd.DataFrame([
        {"Wave": "Wave 1", "Phase": "Data Centralization & Visibility",
         "Duration": "Months 1–3", "Status": "✅ Complete"},
        {"Wave": "Wave 2", "Phase": "Predictive Analytics & Demand Sensing",
         "Duration": "Months 4–8", "Status": "🔄 Active"},
        {"Wave": "Wave 3", "Phase": "Network Optimization & MEIO",
         "Duration": "Months 9–15", "Status": "📋 Planned"},
    ])
    st.dataframe(roadmap, use_container_width=True, hide_index=True)


# ── Demand Forecasting ────────────────────────────────────────────────────────

def page_demand_forecasting(sales_df, forecasts_df, promo_df, acc_df, factory_f, segment_f):
    st.title("🔮 AI Demand Forecasting — Wave 2")
    st.caption("GBM multi-variable model with seasonal, promotional & economic drivers")

    # Accuracy banner
    mean_mape = acc_df[acc_df["factory"] != "ALL"]["cv_mape"].mean()
    a1, a2, a3 = st.columns(3)
    a1.metric("Mean CV MAPE", f"{mean_mape:.1%}")
    a2.metric("Improvement vs. Naive", f"{acc_df[acc_df['factory']=='ALL']['improvement_vs_naive_pct'].iloc[0]:.1f}%",
              f"Target: {FINANCIALS['target_forecast_accuracy_improvement_pct']}%")
    a3.metric("Models Trained", len(acc_df) - 1)

    st.divider()

    # 30-day forecast chart
    st.subheader("30-Day Network Demand Forecast")
    filt = forecasts_df[
        (forecasts_df["factory"].isin(factory_f)) &
        (forecasts_df["segment"].isin(segment_f))
    ]
    agg = filt.groupby(["date", "segment"]).agg(
        forecast_units=("forecast_units", "sum"),
        lower_95=("lower_95", "sum"),
        upper_95=("upper_95", "sum"),
    ).reset_index()

    fig = go.Figure()
    for seg in agg["segment"].unique():
        s = agg[agg["segment"] == seg]
        color = SEGMENT_COLORS.get(seg, "#7f7f7f")
        fig.add_trace(go.Scatter(
            x=s["date"], y=s["forecast_units"], mode="lines",
            name=seg, line=dict(color=color, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([s["date"], s["date"][::-1]]),
            y=pd.concat([s["upper_95"], s["lower_95"][::-1]]),
            fill="toself", fillcolor=color.replace(")", ", 0.12)").replace("rgb(", "rgba("),
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False, name=f"{seg} CI",
        ))
    fig.update_layout(
        title="30-Day Demand Forecast with 95% Confidence Intervals",
        xaxis_title="Date", yaxis_title="Forecast Units",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Promotion impact
    st.subheader("Promotion Lift Analysis (Bullwhip Quantification)")
    if not promo_df.empty:
        fig2 = px.bar(
            promo_df.head(20), x="lift_pct", y="promotion",
            color="segment", orientation="h",
            color_discrete_map=SEGMENT_COLORS,
            labels={"lift_pct": "Demand Lift (%)", "promotion": "Promotion Type"},
            title="Top Promotions by Demand Lift % vs. No-Promo Baseline",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Model accuracy table
    st.subheader("Forecast Accuracy by Factory × SKU")
    st.dataframe(
        acc_df.style.highlight_min(subset=["cv_mape"], color="#c6efce"),
        use_container_width=True, hide_index=True,
    )


# ── MEIO ──────────────────────────────────────────────────────────────────────

def page_meio(inventory_df, ss_df, transfers_df, postponement_df, kpis_dict,
              capital_df, factory_f, segment_f):
    st.title("📦 MEIO — Multi-Echelon Inventory Optimisation")
    st.caption("Network-level stock balancing: SS = Z × σ_LT × √L")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Annual Carrying Saving",
              f"₹{kpis_dict['annual_carrying_saving_inr']:,.0f}")
    k2.metric("Inventory Reduction",
              f"{kpis_dict['inventory_reduction_pct']:.1f}%",
              f"Target: {kpis_dict['target_reduction_pct']}%")
    k3.metric("Stockout Risk Nodes",
              kpis_dict["at_stockout_risk"], delta_color="inverse")
    k4.metric("Overstock Nodes",
              kpis_dict["overstock_nodes"], delta_color="inverse")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Safety Stock Heatmap (Units)")
        pivot = ss_df.pivot_table(
            values="safety_stock_units", index="factory", columns="sku", aggfunc="sum"
        ).fillna(0)
        fig = px.imshow(
            pivot, text_auto=True, aspect="auto",
            color_continuous_scale="Blues",
            title="Safety Stock Levels (units) per Factory × SKU",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Days of Stock Distribution")
        fig2 = px.box(
            ss_df[ss_df["factory"].isin(factory_f)],
            x="factory", y="days_of_stock", color="factory",
            color_discrete_map=FACTORY_COLORS,
            title="Days of Stock per Factory (Target: 15–45 days)",
        )
        fig2.add_hline(y=15, line_dash="dash", line_color="red", annotation_text="Min 15d")
        fig2.add_hline(y=45, line_dash="dash", line_color="orange", annotation_text="Max 45d")
        st.plotly_chart(fig2, use_container_width=True)

    if not transfers_df.empty:
        st.subheader("Network Rebalance Recommendations")
        st.dataframe(transfers_df, use_container_width=True, hide_index=True)

    if not postponement_df.empty:
        st.subheader("Postponement Opportunities (Semi-Finished Carcass Pooling)")
        st.dataframe(postponement_df, use_container_width=True, hide_index=True)

    st.subheader("Capital Liberation Detail (Inventory Right-Sizing)")
    if not capital_df.empty:
        fig3 = px.bar(
            capital_df[capital_df["factory"].isin(factory_f)].head(20),
            x="sku", y="annual_carrying_saving_inr", color="factory",
            color_discrete_map=FACTORY_COLORS,
            title="Annual Carrying Cost Saving by SKU × Factory",
        )
        st.plotly_chart(fig3, use_container_width=True)


# ── Logistics ─────────────────────────────────────────────────────────────────

def page_logistics(logistics_df, best_modes_df, util_df, shifts_df, carbon_df, kpis_dict):
    st.title("🚚 Logistics & Transportation Optimiser — Wave 3")
    st.caption("Road → Rail (DFC) modal shift · Load consolidation · Carbon tracking")

    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Total Freight Spend", f"₹{kpis_dict['total_freight_spend_inr']:,.0f}")
    l2.metric("Road Mode Share", f"{kpis_dict['road_mode_share_pct']:.0f}%",
              "Target: <40%", delta_color="inverse")
    l3.metric("Rail Mode Share", f"{kpis_dict['rail_mode_share_pct']:.0f}%",
              "Target: >60%")
    l4.metric("Modal Shift Saving",
              f"₹{kpis_dict['potential_modal_shift_saving_inr']:,.0f}")

    st.divider()

    # Route cost comparison
    st.subheader("Optimal Modal Mix — Route Cost Matrix")
    opt = best_modes_df[best_modes_df["recommendation"] == "✓ Optimal"]
    fig = px.scatter(
        opt, x="transit_days", y="cost_per_unit_inr",
        color="mode", symbol="segment", size="distance_km",
        hover_data=["origin", "destination"],
        color_discrete_map={
            "Road": "#EF553B", "Rail (DFC)": "#00CC96", "Sea": "#636EFA"
        },
        title="Cost vs. Transit Time by Mode (bubble = distance)",
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Load Utilisation (Current State)")
        if not util_df.empty:
            fig2 = px.bar(
                util_df, x="origin", y="avg_utilisation_pct",
                color="transport_mode", barmode="group",
                title="Avg Load Factor % by Origin & Mode (Target: >85%)",
            )
            fig2.add_hline(y=85, line_dash="dash", line_color="green")
            st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("CO₂ Index by Transport Mode")
        if not carbon_df.empty:
            fig3 = px.pie(
                carbon_df, names="transport_mode", values="total_co2_index",
                title="CO₂ Emission Share by Current Modal Mix",
                color_discrete_map={
                    "Road": "#EF553B", "Rail (DFC)": "#00CC96", "Sea": "#636EFA"
                },
            )
            st.plotly_chart(fig3, use_container_width=True)

    if not shifts_df.empty:
        st.subheader("Modal Shift Savings — Long-Haul Routes (>600 km)")
        st.dataframe(shifts_df, use_container_width=True, hide_index=True)


# ── Bloom Energy / SOFC ───────────────────────────────────────────────────────

def page_energy(energy_df, uptime_df, sus_df, alerts_df, roi_df):
    st.title("⚡ Bloom Energy SOFC — Always-On Manufacturing")
    st.caption("Solid Oxide Fuel Cell deployment · Curing press protection · C3 AI PM proxy")

    e1, e2, e3 = st.columns(3)
    e1.metric(
        "Max Grid Uptime Gap",
        f"{uptime_df['uptime_gap_pp'].max():.2f} pp",
        "vs SOFC 99.99% guarantee",
        delta_color="inverse",
    )
    e2.metric(
        "Annual CO₂ Saved (if all SOFC)",
        f"{sus_df['co2_saved_kg'].sum() / 1000:,.0f} tonnes",
    )
    e3.metric(
        "Annual Scrap Saving Potential",
        f"₹{uptime_df['annual_scrap_saving_inr'].sum():,.0f}",
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Grid Reliability vs. SOFC Target")
        fig = px.bar(
            uptime_df, x="factory", y=["grid_uptime_pct", "target_uptime_pct"],
            barmode="group",
            labels={"value": "Uptime %", "variable": ""},
            color_discrete_map={
                "grid_uptime_pct": "#EF553B",
                "target_uptime_pct": "#00CC96",
            },
            title="Actual Grid Uptime vs. SOFC Target (99.99%)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("SOFC Sustainability Impact")
        fig2 = px.bar(
            sus_df, x="factory",
            y=["co2_saved_kg", "water_saved_kl"],
            barmode="group",
            title="CO₂ Saved (kg) & Water Saved (kL) per Factory",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("SOFC ROI Analysis — Full DCF Model")
    st.dataframe(
        roi_df[["factory", "capex_inr", "net_annual_benefit_inr",
                "npv_inr", "irr_pct", "payback_years", "recommendation"]],
        use_container_width=True, hide_index=True,
    )

    # Predictive maintenance alerts
    if not alerts_df.empty:
        st.subheader("⚠️ Predictive Maintenance Alerts (C3 AI Reliability Suite Proxy)")
        crit = alerts_df[alerts_df["risk_level"] == "CRITICAL"]
        warn = alerts_df[alerts_df["risk_level"] == "WARNING"]
        m1, m2 = st.columns(2)
        m1.metric("Critical Alerts", len(crit), delta_color="inverse")
        m2.metric("Warning Alerts", len(warn), delta_color="inverse")
        st.dataframe(
            alerts_df.tail(20).sort_values("risk_level"),
            use_container_width=True, hide_index=True,
        )


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    page, factory_f, segment_f = render_sidebar()

    # Load all data
    sales_df, inventory_df, logistics_df, energy_df = load_data()

    # Apply filters
    sales_f = sales_df[
        sales_df["factory"].isin(factory_f) & sales_df["segment"].isin(segment_f)
    ]
    inv_f = inventory_df[inventory_df["factory"].isin(factory_f)]

    # Pre-compute
    forecasts, promo_report, acc, forecaster = load_forecasts(sales_df)
    ss, transfers, postponement, kpis_meio, capital = load_meio(inventory_df)
    best_modes, util, shifts, carbon, log_kpis = load_logistics(logistics_df)
    uptime, sus, alerts, roi = load_energy(energy_df)

    if page == "📊 Executive Summary":
        page_executive_summary(
            sales_f, inv_f, logistics_df, energy_df,
            kpis_meio, log_kpis, uptime,
        )
    elif page == "🔮 Demand Forecasting":
        page_demand_forecasting(
            sales_f, forecasts, promo_report, acc, factory_f, segment_f
        )
    elif page == "📦 MEIO Inventory":
        page_meio(inv_f, ss, transfers, postponement, kpis_meio, capital, factory_f, segment_f)
    elif page == "🚚 Logistics Optimizer":
        page_logistics(logistics_df, best_modes, util, shifts, carbon, log_kpis)
    elif page == "⚡ Bloom Energy (SOFC)":
        page_energy(energy_df, uptime, sus, alerts, roi)


if __name__ == "__main__":
    main()
