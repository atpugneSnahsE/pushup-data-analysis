import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# CONFIG

CSV_PATH = (
    "preprocessing_outputs/"
    "cleaned_dataset.csv"
)

PUSHUP_PRICE = 2

OUTPUT_DIR = Path(
    "feature_interest_outputs"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)

# LOAD DATA
print("\nLoading cleaned dataset...")

df = pd.read_csv(CSV_PATH)

print("Dataset loaded.")
print(f"Shape: {df.shape}")

# Q1
# DEFINE FEATURE INTEREST METRIC

print("\n" + "=" * 60)
print("Q1 FEATURE INTEREST METRIC")
print("=" * 60)

print(
    "\nMetric Selected:"
)

print(
    "Push Up Adoption Rate"
)

print(
    "\nFormula:"
)

print(
    "estimated_pushups_sold "
    "/ number_of_listings"
)

# Estimated pushups sold
df["estimated_pushups_sold"] = (
    df["revenue_from_push_ups"]
    / PUSHUP_PRICE
)

# Feature interest metric
df["pushup_rate"] = (
    df["estimated_pushups_sold"]
    / df["number_of_listings"]
)

print(
    "\nMetric calculated "
    "successfully."
)

print(
    "\nSummary:"
)

print(
    df["pushup_rate"]
    .describe()
)

# Q2
# BEST CATEGORIES

print("\n" + "=" * 60)
print("Q2 BEST PERFORMING CATEGORIES")
print("=" * 60)

top_categories = (
    df.sort_values(
        "pushup_rate",
        ascending=False
    )[
        [
            "category_2",
            "category_3",
            "number_of_listings",
            "avg_listing_price_eur",
            "pushup_rate"
        ]
    ]
    .head(15)
)

print(
    "\nTop categories "
    "by Push Up Rate:"
)

print(top_categories)

top_categories.to_csv(
    OUTPUT_DIR /
    "top_categories.csv",
    index=False
)

print(
    "\nSaved:"
)

print(
    OUTPUT_DIR /
    "top_categories.csv"
)

# Q3
# DEFINE CORRELATED METRIC

print("\n" + "=" * 60)
print("Q3 CORRELATED METRIC")
print("=" * 60)

print(
    "\nMetric Selected:"
)

print(
    "Competition Intensity"
)

print(
    "\nProxy Variable:"
)

print(
    "number_of_listings"
)

# Correlation
correlation = (
    df["pushup_rate"]
    .corr(
        df["number_of_listings"]
    )
)

print(
    f"\nCorrelation "
    f"(Push Up Rate vs Listings): "
    f"{correlation:.4f}"
)

# GRAPH

print(
    "\nGenerating graph..."
)

plt.figure(figsize=(10, 6))

plt.scatter(
    df["number_of_listings"],
    df["pushup_rate"],
    alpha=0.7
)

# trend line
z = np.polyfit(
    df["number_of_listings"],
    df["pushup_rate"],
    1
)

p = np.poly1d(z)

plt.plot(
    df["number_of_listings"],
    p(
        df["number_of_listings"]
    )
)

plt.xlabel(
    "Competition Intensity "
    "(Number of Listings)"
)

plt.ylabel(
    "Push Up Adoption Rate"
)

plt.title(
    "Relationship Between "
    "Competition and "
    "Push Up Adoption"
)

plt.tight_layout()

graph_path = (
    OUTPUT_DIR /
    "competition_vs_pushup_rate.png"
)

plt.savefig(graph_path)

plt.close()

print(
    f"\nGraph saved:"
)

print(graph_path.resolve())

# Q4
# GIRLS_CLOTHING/FOR_BABIES

print("\n" + "=" * 60)
print("Q4 FOR_BABIES ANALYSIS")
print("=" * 60)

target = df[
    (
        df["category_2"]
        == "GIRLS_CLOTHING"
    )
    &
    (
        df["category_3"]
        == "FOR_BABIES"
    )
]

if len(target) > 0:

    avg_price = (
        target[
            "avg_listing_price_eur"
        ].iloc[0]
    )

    pushup_share = (
        PUSHUP_PRICE
        / avg_price
    )

    print(
        "\nFOR_BABIES "
        "Category Found"
    )

    print(
        f"\nAverage Listing "
        f"Price: "
        f"€{avg_price:.2f}"
    )

    print(
        f"Push Up Cost: "
        f"€{PUSHUP_PRICE}"
    )

    print(
        f"Push Up Cost as "
        f"% of Listing Price: "
        f"{pushup_share*100:.2f}%"
    )

    print(
        "\nInterpretation:"
    )

    if pushup_share > 0.5:
        print(
            "Push up price is "
            "more than half "
            "of item value."
        )

        print(
            "Possible reasons "
            "for adoption:"
        )

        reasons = [
            "Faster selling "
            "priority",

            "Baby items "
            "become obsolete "
            "quickly",

            "Competition in "
            "the category",

            "Bundle sales "
            "may increase "
            "perceived value",

            "Convenience "
            "outweighs cost"
        ]

        for r in reasons:
            print(f"- {r}")

else:
    print(
        "\nFOR_BABIES "
        "category not found."
    )

# SAVE FINAL DATA

save_path = (
    OUTPUT_DIR /
    "feature_interest_analysis.csv"
)

df.to_csv(
    save_path,
    index=False
)

print("\nSaved final dataset:")
print(save_path.resolve())

print(
    "\nFEATURE ANALYSIS "
    "COMPLETED."
)