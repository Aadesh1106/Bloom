"""
Central configuration for the Tyre Manufacturing Supply Chain Optimizer.
Covers all three factory nodes: Gurgaon, Chennai, Mumbai.
"""

# ── Factory nodes ─────────────────────────────────────────────────────────────
FACTORIES = {
    "Gurgaon": {
        "region": "North",
        "corridor": "Delhi-Mumbai Industrial Corridor",
        "segments": ["Car & SUV"],
        "port_access": False,
        "grid_reliability_score": 0.82,   # 1.0 = perfect; <0.85 = SOFC candidate
        "lat": 28.4595,
        "lon": 77.0266,
    },
    "Mumbai": {
        "region": "West",
        "corridor": "Golden Quadrilateral / JNPT Port",
        "segments": ["Van & Light Truck", "Commercial"],
        "port_access": True,
        "grid_reliability_score": 0.88,
        "lat": 19.0760,
        "lon": 72.8777,
    },
    "Chennai": {
        "region": "South",
        "corridor": "Chennai-Bangalore Corridor / Sea",
        "segments": ["Truck & Bus", "Specialty"],
        "port_access": True,
        "grid_reliability_score": 0.79,   # Most grid-vulnerable → first SOFC site
        "lat": 13.0827,
        "lon": 80.2707,
    },
}

# ── Product segments ──────────────────────────────────────────────────────────
SEGMENTS = ["Car & SUV", "Van & Light Truck", "Truck & Bus"]

# SKU catalogue (factory × segment)
SKUS = [
    "CAR-185-65R15",
    "CAR-205-55R16",
    "SUV-235-65R17",
    "VLT-195-80R15",
    "VLT-215-75R17",
    "TRB-295-80R22",
    "TRB-315-80R22",
]

# ── Inventory & service-level parameters ─────────────────────────────────────
SERVICE_LEVEL_Z = {
    "Car & SUV": 1.96,        # 97.5 % — high retail sensitivity
    "Van & Light Truck": 1.65, # 95 %
    "Truck & Bus": 1.88,       # 97 %  — contractual B2B
}

# Days of inventory targets (min / max)
INVENTORY_TARGETS = {
    "min_days": 15,
    "max_days": 45,
    "safety_stock_buffer_days": 7,
}

# ── Logistics ────────────────────────────────────────────────────────────────
TRANSPORT_MODES = {
    "Road": {
        "cost_per_ton_km": 2.58,       # Rs
        "avg_speed_kmph": 25,
        "carbon_factor": 1.0,          # relative baseline
        "reliability": 0.78,
    },
    "Rail (DFC)": {
        "cost_per_ton_km": 1.41,       # Rs
        "avg_speed_kmph": 62,
        "carbon_factor": 0.44,         # 2.25× lower CO₂
        "reliability": 0.95,
    },
    "Sea": {
        "cost_per_ton_km": 0.80,
        "avg_speed_kmph": 18,
        "carbon_factor": 0.30,
        "reliability": 0.90,
    },
}

# Inter-factory distances (km) — approximate road distances
DISTANCES_KM = {
    ("Gurgaon", "Mumbai"): 1420,
    ("Gurgaon", "Chennai"): 2180,
    ("Mumbai", "Chennai"): 1340,
    ("Mumbai", "Gurgaon"): 1420,
    ("Chennai", "Gurgaon"): 2180,
    ("Chennai", "Mumbai"): 1340,
}

# Average tyre weight (tonnes / unit) for cost calculations
TYRE_WEIGHT_TONNES = {
    "Car & SUV": 0.010,
    "Van & Light Truck": 0.018,
    "Truck & Bus": 0.065,
}

# ── Energy / Bloom SOFC parameters ───────────────────────────────────────────
BLOOM_SOFC = {
    "uptime_guarantee": 0.9999,       # "always-on"
    "grid_uptime": 0.97,              # typical Indian industrial grid
    "emission_reduction_pct": 0.275,  # ~19-36 % midpoint
    "water_saved_liters_per_kwh": 4.2,
    "curing_press_power_kw": 250,     # per press
    "presses_per_factory": 40,
    "scrap_cost_per_batch_inr": 85_000,
}

# ── Demand & festive seasonality ─────────────────────────────────────────────
# Multiplicative demand lift by month (1.0 = baseline)
SEASONAL_INDICES = {
    1: 0.92,   # Jan
    2: 0.88,   # Feb
    3: 0.95,   # Mar
    4: 0.98,   # Apr
    5: 0.90,   # May (pre-monsoon dip)
    6: 0.80,   # Jun (monsoon start)
    7: 0.78,   # Jul (peak monsoon)
    8: 0.82,   # Aug
    9: 0.96,   # Sep (post-monsoon recovery)
    10: 1.28,  # Oct (Diwali / festive peak)
    11: 1.22,  # Nov (festive tail)
    12: 1.05,  # Dec
}

# Promotion lift (additive on top of seasonal index)
PROMOTION_LIFT = {
    "Cash Discount": 0.18,
    "Extended Warranty": 0.12,
    "Combo Offer": 0.22,
    "None": 0.0,
}

# ── Financial benchmarks ─────────────────────────────────────────────────────
FINANCIALS = {
    "target_sales_lift_pct": 9.7,
    "target_inventory_reduction_pct": 25.0,
    "target_carrying_cost_saving_usd": 4_150_000,
    "target_logistics_saving_pct": 15.0,
    "target_forecast_accuracy_improvement_pct": 4.5,
    "target_production_yield_pct": 98.0,
    "avg_tyre_price_inr": {
        "Car & SUV": 4_500,
        "Van & Light Truck": 8_200,
        "Truck & Bus": 22_000,
    },
    "holding_cost_rate_annual": 0.22,   # 22 % of inventory value
}
