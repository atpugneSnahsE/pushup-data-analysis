import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =====================================================
# CONFIG
# =====================================================

CSV_PATH = (
    "feature_interest_outputs/"
    "feature_interest_analysis.csv"
)

OUTPUT_DIR = Path(
    "pricing_strategy_outputs"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)

CURRENT_PRICE = 2


# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading dataset...")

df = pd.read_csv(CSV_PATH)

print("Dataset loaded.")
print(f"Shape: {df.shape}")


# =====================================================
# BASELINE
# =====================================================

print("\n" + "=" * 60)
print("CURRENT €2 MODEL")
print("=" * 60)

baseline_revenue = (
    df["revenue_from_push_ups"]
    .sum()
)

baseline_pushups = (
    baseline_revenue
    / CURRENT_PRICE
)

print(
    f"\nCurrent Revenue: "
    f"€{baseline_revenue:,.2f}"
)

print(
    f"Estimated Pushups Sold: "
    f"{baseline_pushups:,.0f}"
)


# =====================================================
# PRICE SCENARIOS
# =====================================================

print("\n" + "=" * 60)
print("PRICE SCENARIOS")
print("=" * 60)

price_scenarios = {
    1.0: 1.40,
    1.5: 1.20,
    2.0: 1.00,
    2.5: 0.85,
    3.0: 0.70,
    4.0: 0.50
}

results = []

for price, adoption_factor in (
    price_scenarios.items()
):

    estimated_pushups = (
        baseline_pushups
        * adoption_factor
    )

    estimated_revenue = (
        estimated_pushups
        * price
    )

    revenue_change_pct = (
        (
            estimated_revenue
            - baseline_revenue
        )
        / baseline_revenue
    ) * 100

    results.append({
        "price": price,
        "adoption_factor":
        adoption_factor,

        "estimated_pushups":
        estimated_pushups,

        "estimated_revenue":
        estimated_revenue,

        "revenue_change_pct":
        revenue_change_pct
    })


scenario_df = pd.DataFrame(results)

print(
    "\nScenario Comparison:"
)

print(scenario_df)


# =====================================================
# CATEGORY LEVEL IMPACT
# =====================================================

print("\n" + "=" * 60)
print("CATEGORY IMPACT")
print("=" * 60)

premium_categories = df[
    df["avg_listing_price_eur"]
    > 50
].copy()

low_price_categories = df[
    df["avg_listing_price_eur"]
    < 15
].copy()

print(
    f"\nPremium categories: "
    f"{len(premium_categories)}"
)

print(
    f"Low price categories: "
    f"{len(low_price_categories)}"
)

premium_revenue = (
    premium_categories[
        "revenue_from_push_ups"
    ].sum()
)

low_price_revenue = (
    low_price_categories[
        "revenue_from_push_ups"
    ].sum()
)

print(
    f"\nPremium category "
    f"revenue: "
    f"€{premium_revenue:,.2f}"
)

print(
    f"Low price category "
    f"revenue: "
    f"€{low_price_revenue:,.2f}"
)


# =====================================================
# BEST STRATEGY
# =====================================================

best_strategy = scenario_df.loc[
    scenario_df[
        "estimated_revenue"
    ].idxmax()
]

print("\n" + "=" * 60)
print("BEST SIMULATED STRATEGY")
print("=" * 60)

print(
    f"\nBest price: "
    f"€{best_strategy['price']}"
)

print(
    f"Expected revenue: "
    f"€{best_strategy['estimated_revenue']:,.2f}"
)

print(
    f"Revenue change: "
    f"{best_strategy['revenue_change_pct']:.2f}%"
)


# =====================================================
# GRAPH
# =====================================================

plt.figure(figsize=(10, 6))

plt.plot(
    scenario_df["price"],
    scenario_df[
        "estimated_revenue"
    ],
    marker="o"
)

plt.xlabel(
    "Push Up Price (€)"
)

plt.ylabel(
    "Estimated Revenue (€)"
)

plt.title(
    "Revenue Impact of "
    "Push Up Price Changes"
)

plt.tight_layout()

graph_path = (
    OUTPUT_DIR /
    "price_vs_revenue.png"
)

plt.savefig(graph_path)

plt.close()


# =====================================================
# BAR COMPARISON
# =====================================================

plt.figure(figsize=(10, 6))

labels = [
    f"€{x}"
    for x in
    scenario_df["price"]
]

plt.bar(
    labels,
    scenario_df[
        "estimated_revenue"
    ]
)

plt.xlabel(
    "Price Scenario"
)

plt.ylabel(
    "Revenue (€)"
)

plt.title(
    "Comparison of "
    "Pricing Strategies"
)

plt.tight_layout()

bar_path = (
    OUTPUT_DIR /
    "pricing_comparison.png"
)

plt.savefig(bar_path)

plt.close()


# =====================================================
# SAVE RESULTS
# =====================================================

save_path = (
    OUTPUT_DIR /
    "pricing_scenarios.csv"
)

scenario_df.to_csv(
    save_path,
    index=False
)

print("\nResults saved:")
print(save_path.resolve())

print("\nGraphs saved:")
print(OUTPUT_DIR.resolve())

print(
    "\nPRICING ANALYSIS "
    "COMPLETED."
)