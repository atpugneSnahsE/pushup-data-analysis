import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import mannwhitneyu

import statsmodels.api as sm
from statsmodels.stats.power import TTestIndPower


# =====================================================
# CONFIG
# =====================================================

CSV_PATH = (
    "preprocessing_outputs/"
    "cleaned_dataset.csv"
)

PUSHUP_PRICE = 2

OUTPUT_DIR = Path(
    "advanced_analysis_outputs"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading dataset...")

df = pd.read_csv(CSV_PATH)

print("Dataset loaded.")
print(f"Shape: {df.shape}")


# =====================================================
# CREATE FEATURES
# =====================================================

print("\n" + "=" * 70)
print("1. FEATURE ENGINEERING")
print("=" * 70)

df["estimated_pushups_sold"] = (
    df["revenue_from_push_ups"]
    / PUSHUP_PRICE
)

df["pushup_rate"] = (
    df["estimated_pushups_sold"]
    / df["number_of_listings"]
)

df["revenue_per_listing"] = (
    df["revenue_from_push_ups"]
    / df["number_of_listings"]
)

# Relative promotion cost
df["relative_promo_cost"] = (
    PUSHUP_PRICE
    / df["avg_listing_price_eur"]
)

print("\nFeatures created:")
print([
    "estimated_pushups_sold",
    "pushup_rate",
    "revenue_per_listing",
    "relative_promo_cost"
])


# =====================================================
# SECTION 1
# PEARSON + SPEARMAN
# =====================================================

print("\n" + "=" * 70)
print("2. CORRELATION ANALYSIS")
print("=" * 70)

metrics = [
    "avg_listing_price_eur",
    "number_of_listings",
    "relative_promo_cost"
]

corr_results = []

for metric in metrics:

    pearson_corr, pearson_p = (
        pearsonr(
            df[metric],
            df["pushup_rate"]
        )
    )

    spearman_corr, spearman_p = (
        spearmanr(
            df[metric],
            df["pushup_rate"]
        )
    )

    corr_results.append({
        "metric": metric,

        "pearson_corr":
        pearson_corr,

        "pearson_p":
        pearson_p,

        "spearman_corr":
        spearman_corr,

        "spearman_p":
        spearman_p
    })

corr_df = pd.DataFrame(
    corr_results
)

print("\nCorrelation Results:")
print(corr_df)

corr_df.to_csv(
    OUTPUT_DIR /
    "correlation_results.csv",
    index=False
)


# =====================================================
# SECTION 2
# MULTIVARIATE REGRESSION
# =====================================================

print("\n" + "=" * 70)
print("3. MULTIVARIATE REGRESSION")
print("=" * 70)

# Log transform listings
df["log_listings"] = np.log1p(
    df["number_of_listings"]
)

X = df[
    [
        "avg_listing_price_eur",
        "log_listings",
        "relative_promo_cost"
    ]
]

X = sm.add_constant(X)

y = df["pushup_rate"]

model = sm.OLS(y, X).fit()

print(model.summary())

with open(
    OUTPUT_DIR /
    "regression_summary.txt",
    "w"
) as f:

    f.write(
        str(model.summary())
    )


# =====================================================
# SECTION 3
# PRICE BAND SEGMENTATION
# =====================================================

print("\n" + "=" * 70)
print("4. PRICE BAND SEGMENTATION")
print("=" * 70)

df["price_band"] = pd.qcut(
    df["avg_listing_price_eur"],
    q=4,
    labels=[
        "Low",
        "Medium",
        "High",
        "Premium"
    ]
)

price_band_summary = (
    df.groupby("price_band")
    .agg({
        "avg_listing_price_eur":
        "mean",

        "pushup_rate":
        "mean",

        "revenue_from_push_ups":
        "sum",

        "number_of_listings":
        "sum"
    })
    .reset_index()
)

print("\nPrice Band Summary:")
print(price_band_summary)

price_band_summary.to_csv(
    OUTPUT_DIR /
    "price_band_summary.csv",
    index=False
)


# =====================================================
# SECTION 4
# DATA DRIVEN PRICING
# =====================================================

print("\n" + "=" * 70)
print("5. DATA DRIVEN PRICING")
print("=" * 70)

def recommend_price(pushup_rate):

    if pushup_rate < 0.02:
        return 1.5

    elif pushup_rate < 0.05:
        return 2.0

    elif pushup_rate < 0.08:
        return 2.5

    return 3.0


price_band_summary[
    "recommended_price"
] = (
    price_band_summary[
        "pushup_rate"
    ]
    .apply(recommend_price)
)

print("\nRecommended Prices:")
print(
    price_band_summary[
        [
            "price_band",
            "recommended_price"
        ]
    ]
)


# =====================================================
# SECTION 5
# EMPIRICAL REVENUE SIMULATION
# =====================================================

print("\n" + "=" * 70)
print("6. REVENUE SIMULATION")
print("=" * 70)

simulation_results = []

for band in (
    price_band_summary[
        "price_band"
    ]
):

    subset = df[
        df["price_band"]
        == band
    ]

    current_revenue = (
        subset[
            "revenue_from_push_ups"
        ]
        .sum()
    )

    current_pushups = (
        current_revenue
        / PUSHUP_PRICE
    )

    pushup_rate = (
        subset[
            "pushup_rate"
        ]
        .mean()
    )

    recommended_price = (
        price_band_summary[
            price_band_summary[
                "price_band"
            ]
            == band
        ][
            "recommended_price"
        ]
        .iloc[0]
    )

    price_ratio = (
        recommended_price
        / PUSHUP_PRICE
    )

    adoption_adjustment = (
        1 / price_ratio
    ) ** 0.5

    estimated_pushups = (
        current_pushups
        * adoption_adjustment
    )

    estimated_revenue = (
        estimated_pushups
        * recommended_price
    )

    revenue_change_pct = (
        (
            estimated_revenue
            - current_revenue
        )
        / current_revenue
    ) * 100

    simulation_results.append({

        "price_band":
        band,

        "current_revenue":
        current_revenue,

        "recommended_price":
        recommended_price,

        "estimated_revenue":
        estimated_revenue,

        "revenue_change_pct":
        revenue_change_pct
    })

simulation_df = pd.DataFrame(
    simulation_results
)

print("\nRevenue Simulation:")
print(simulation_df)

simulation_df.to_csv(
    OUTPUT_DIR /
    "pricing_simulation.csv",
    index=False
)


# =====================================================
# SECTION 6
# POWER ANALYSIS
# =====================================================

print("\n" + "=" * 70)
print("7. A/B TEST POWER ANALYSIS")
print("=" * 70)

baseline_rate = (
    df["pushup_rate"]
    .mean()
)

mde = baseline_rate * 0.05

effect_size = (
    mde
    / df["pushup_rate"].std()
)

analysis = TTestIndPower()

sample_size = (
    analysis.solve_power(
        effect_size=effect_size,
        alpha=0.05,
        power=0.80,
        ratio=1
    )
)

print(
    f"\nBaseline "
    f"PushUpRate: "
    f"{baseline_rate:.4f}"
)

print(
    f"Minimum Detectable "
    f"Effect (5%): "
    f"{mde:.4f}"
)

print(
    f"Required sample "
    f"size per group: "
    f"{int(np.ceil(sample_size))}"
)


# =====================================================
# SECTION 7
# NETWORK EFFECT COMMENT
# =====================================================

print("\n" + "=" * 70)
print("8. EXPERIMENT DESIGN NOTE")
print("=" * 70)

print(
    "\nMarketplace "
    "experiments may violate "
    "independence assumptions."
)

print(
    "Recommended approach:"
)

print(
    "- Cluster randomized test"
)

print(
    "- Randomize by category "
    "or seller cohort"
)

print(
    "- Avoid individual "
    "seller randomization"
)


# =====================================================
# VISUALIZATION
# =====================================================

print("\nGenerating plots...")

# Price vs PushUpRate
plt.figure(figsize=(10, 6))

plt.scatter(
    df["avg_listing_price_eur"],
    df["pushup_rate"],
    alpha=0.7
)

z = np.polyfit(
    df["avg_listing_price_eur"],
    df["pushup_rate"],
    1
)

p = np.poly1d(z)

plt.plot(
    df["avg_listing_price_eur"],
    p(
        df["avg_listing_price_eur"]
    )
)

plt.xlabel(
    "Average Listing Price (€)"
)

plt.ylabel(
    "Push Up Adoption Rate"
)

plt.title(
    "Price vs Push Up Adoption"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "price_vs_pushup_rate.png"
)

plt.close()


# Revenue Simulation Plot
plt.figure(figsize=(10, 6))

plt.bar(
    simulation_df[
        "price_band"
    ],
    simulation_df[
        "estimated_revenue"
    ]
)

plt.xlabel(
    "Price Band"
)

plt.ylabel(
    "Estimated Revenue (€)"
)

plt.title(
    "Estimated Revenue "
    "by Price Band"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "revenue_simulation.png"
)

plt.close()


print("\nOutputs saved:")
print(OUTPUT_DIR.resolve())

print(
    "\nADVANCED ANALYSIS "
    "COMPLETED."
)
