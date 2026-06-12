import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# CONFIG

CSV_PATH = "/home/vgtu/Assignment/Pricing Push Ups - data.csv"

PUSH_UP_PRICE = 2  # €2 per push up

# LOAD DATA

print("\nLoading dataset...")

df = pd.read_csv(CSV_PATH)

print("\nDataset loaded successfully.")

# BASIC DATASET OVERVIEW

print("\n" + "=" * 60)
print("1. DATASET OVERVIEW")
print("=" * 60)

print("\nShape of dataset:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nFirst 5 Rows:")
print(df.head())

# MISSING VALUE ANALYSIS

print("\n" + "=" * 60)
print("2. MISSING VALUE ANALYSIS")
print("=" * 60)

missing_values = df.isnull().sum()

print("\nMissing Values:")
print(missing_values)

missing_percentage = (
    df.isnull().sum() / len(df)
) * 100

print("\nMissing Percentage:")
print(missing_percentage.round(2))

# DUPLICATE CHECK

print("\n" + "=" * 60)
print("3. DUPLICATE CHECK")
print("=" * 60)

duplicates = df.duplicated().sum()

print(f"\nTotal duplicate rows: {duplicates}")

category_duplicates = df.duplicated(
    subset=["category_2", "category_3"]
).sum()

print(
    f"Duplicate category pairs: "
    f"{category_duplicates}"
)

# DESCRIPTIVE STATISTICS

print("\n" + "=" * 60)
print("4. DESCRIPTIVE STATISTICS")
print("=" * 60)

numeric_cols = [
    "number_of_listings",
    "avg_listing_price_eur",
    "revenue_from_push_ups"
]

print("\nSummary Statistics:")
print(df[numeric_cols].describe())

# CATEGORY ANALYSIS

print("\n" + "=" * 60)
print("5. CATEGORY ANALYSIS")
print("=" * 60)

print("\nUnique category_2 count:")
print(df["category_2"].nunique())

print("\nUnique category_3 count:")
print(df["category_3"].nunique())

print("\nTop 10 category_2:")
print(
    df["category_2"]
    .value_counts()
    .head(10)
)

print("\nTop 10 category_3:")
print(
    df["category_3"]
    .value_counts()
    .head(10)
)

# DATA QUALITY TESTING

print("\n" + "=" * 60)
print("6. DATA QUALITY TESTS")
print("=" * 60)

negative_listings = (
    df["number_of_listings"] < 0
).sum()

negative_prices = (
    df["avg_listing_price_eur"] < 0
).sum()

negative_revenue = (
    df["revenue_from_push_ups"] < 0
).sum()

print(f"\nNegative listings: "
      f"{negative_listings}")

print(f"Negative prices: "
      f"{negative_prices}")

print(f"Negative revenue: "
      f"{negative_revenue}")


# Check invalid data types
non_integer_listings = (
    df["number_of_listings"] % 1 != 0
).sum()

print(
    f"Non integer listing counts: "
    f"{non_integer_listings}"
)

# CATEGORY CONSISTENCY CHECK

print("\n" + "=" * 60)
print("7. CATEGORY CONSISTENCY CHECK")
print("=" * 60)

category_mapping = (
    df.groupby("category_3")
    ["category_2"]
    .nunique()
)

inconsistent_categories = (
    category_mapping[
        category_mapping > 1
    ]
)

print(
    "\nCategory_3 appearing in "
    "multiple category_2:"
)

print(inconsistent_categories)

# BUSINESS LOGIC TESTING

print("\n" + "=" * 60)
print("8. BUSINESS LOGIC TESTING")
print("=" * 60)

df["estimated_pushups_sold"] = (
    df["revenue_from_push_ups"]
    / PUSH_UP_PRICE
)

df["pushup_rate"] = (
    df["estimated_pushups_sold"]
    / df["number_of_listings"]
)

print("\nPushup Metrics:")
print(
    df[
        [
            "category_2",
            "category_3",
            "estimated_pushups_sold",
            "pushup_rate"
        ]
    ].head()
)

print("\nTop categories by pushup revenue:")
print(
    df.sort_values(
        "revenue_from_push_ups",
        ascending=False
    )[
        [
            "category_2",
            "category_3",
            "revenue_from_push_ups"
        ]
    ].head(10)
)

print("\nTop categories by listing volume:")
print(
    df.sort_values(
        "number_of_listings",
        ascending=False
    )[
        [
            "category_2",
            "category_3",
            "number_of_listings"
        ]
    ].head(10)
)

print("\nTop categories by avg price:")
print(
    df.sort_values(
        "avg_listing_price_eur",
        ascending=False
    )[
        [
            "category_2",
            "category_3",
            "avg_listing_price_eur"
        ]
    ].head(10)
)

# CORRELATION ANALYSIS=

print("\n" + "=" * 60)
print("9. CORRELATION ANALYSIS")
print("=" * 60)

correlation = (
    df[numeric_cols]
    .corr()
)

print("\nCorrelation Matrix:")
print(correlation)

# VISUALIZATION

print("\nGenerating plots...")

output_dir = Path("eda_outputs")
output_dir.mkdir(exist_ok=True)


# Histogram: Listings
plt.figure(figsize=(8, 5))
plt.hist(df["number_of_listings"], bins=30)
plt.xlabel("Number of Listings")
plt.ylabel("Frequency")
plt.title("Distribution of Listings")
plt.tight_layout()
plt.savefig(
    output_dir / "listing_distribution.png"
)
plt.close()


# Histogram: Avg Price
plt.figure(figsize=(8, 5))
plt.hist(
    df["avg_listing_price_eur"],
    bins=30
)
plt.xlabel("Average Listing Price (€)")
plt.ylabel("Frequency")
plt.title(
    "Distribution of Listing Prices"
)
plt.tight_layout()
plt.savefig(
    output_dir / "price_distribution.png"
)
plt.close()


# Histogram: Push Up Revenue
plt.figure(figsize=(8, 5))
plt.hist(
    df["revenue_from_push_ups"],
    bins=30
)
plt.xlabel("Push Up Revenue (€)")
plt.ylabel("Frequency")
plt.title(
    "Distribution of Push Up Revenue"
)
plt.tight_layout()
plt.savefig(
    output_dir / "pushup_revenue_distribution.png"
)
plt.close()


# Scatter Plot
plt.figure(figsize=(8, 5))
plt.scatter(
    df["avg_listing_price_eur"],
    df["revenue_from_push_ups"]
)
plt.xlabel(
    "Average Listing Price (€)"
)
plt.ylabel(
    "Push Up Revenue (€)"
)
plt.title(
    "Price vs Push Up Revenue"
)
plt.tight_layout()
plt.savefig(
    output_dir / "price_vs_revenue.png"
)
plt.close()


print("\nPlots saved in:")
print(output_dir.resolve())

print("\nEDA COMPLETED.")