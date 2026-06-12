import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# CONFIG

CSV_PATH = "/home/vgtu/Assignment/Pricing Push Ups - data.csv"   # Change this
PUSHUP_PRICE = 2

OUTPUT_DIR = Path("preprocessing_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# LOAD DATA

print("\nLoading dataset...")

df = pd.read_csv(CSV_PATH)

print("Dataset loaded successfully.")
print(f"Initial shape: {df.shape}")

# STEP 1: DUPLICATE INSPECTION

print("\n" + "=" * 60)
print("1. DUPLICATE INSPECTION")
print("=" * 60)

duplicates = df[
    df.duplicated(
        subset=["category_2", "category_3"],
        keep=False
    )
]

if len(duplicates) > 0:
    print("\nDuplicate category pairs found:")
    print(
        duplicates.sort_values(
            ["category_2", "category_3"]
        )
    )

    duplicates.to_csv(
        OUTPUT_DIR / "duplicate_rows.csv",
        index=False
    )

    print(
        "\nSaved duplicate rows to:"
    )
    print(
        OUTPUT_DIR /
        "duplicate_rows.csv"
    )
else:
    print("\nNo duplicate category pairs found.")

# STEP 2: REMOVE EXACT DUPLICATES

print("\n" + "=" * 60)
print("2. REMOVE EXACT DUPLICATES")
print("=" * 60)

before_rows = len(df)

df = df.drop_duplicates()

after_rows = len(df)

print(
    f"\nRemoved "
    f"{before_rows - after_rows} "
    f"duplicate rows"
)

print(f"Current shape: {df.shape}")

# STEP 3: MISSING CATEGORY VALUES

print("\n" + "=" * 60)
print("3. HANDLE MISSING CATEGORY VALUES")
print("=" * 60)

missing_category_rows = df[
    df["category_2"].isna()
    | df["category_3"].isna()
]

print(
    f"\nRows with missing "
    f"category labels: "
    f"{len(missing_category_rows)}"
)

if len(missing_category_rows) > 0:
    missing_category_rows.to_csv(
        OUTPUT_DIR /
        "missing_categories.csv",
        index=False
    )

    print(
        "Saved missing category "
        "rows for inspection."
    )

# Drop rows with missing categories
df = df.dropna(
    subset=["category_2", "category_3"]
)

print(
    f"Shape after removing "
    f"missing categories: "
    f"{df.shape}"
)

# STEP 4: HANDLE MISSING REVENUE

print("\n" + "=" * 60)
print("4. HANDLE MISSING REVENUE")
print("=" * 60)

missing_revenue = (
    df["revenue_from_push_ups"]
    .isna()
    .sum()
)

print(
    f"\nMissing revenue rows: "
    f"{missing_revenue}"
)

print(
    "\nAssumption: Missing "
    "revenue means zero "
    "push up purchases."
)

df["revenue_from_push_ups"] = (
    df["revenue_from_push_ups"]
    .fillna(0)
)

print(
    "Missing revenue values "
    "filled with 0."
)

# STEP 5: DATA QUALITY TESTS

print("\n" + "=" * 60)
print("5. DATA QUALITY TESTS")
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

print(
    f"\nNegative listings: "
    f"{negative_listings}"
)

print(
    f"Negative prices: "
    f"{negative_prices}"
)

print(
    f"Negative revenue: "
    f"{negative_revenue}"
)

# STEP 6: CREATE BUSINESS FEATURES

print("\n" + "=" * 60)
print("6. CREATE BUSINESS FEATURES")
print("=" * 60)

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

print("\nBusiness features added:")
print(
    [
        "estimated_pushups_sold",
        "pushup_rate",
        "revenue_per_listing"
    ]
)

# STEP 7: CATEGORY SEGMENTATION

print("\n" + "=" * 60)
print("7. CATEGORY SEGMENTATION")
print("=" * 60)

listing_threshold = (
    df["number_of_listings"]
    .median()
)

revenue_threshold = (
    df["revenue_from_push_ups"]
    .median()
)

df["segment"] = np.select(
    [
        (
            df["number_of_listings"]
            >= listing_threshold
        )
        &
        (
            df["revenue_from_push_ups"]
            >= revenue_threshold
        ),

        (
            df["number_of_listings"]
            >= listing_threshold
        )
        &
        (
            df["revenue_from_push_ups"]
            < revenue_threshold
        ),

        (
            df["number_of_listings"]
            < listing_threshold
        )
        &
        (
            df["revenue_from_push_ups"]
            >= revenue_threshold
        )
    ],
    [
        "High Supply High Revenue",
        "High Supply Low Revenue",
        "Low Supply High Revenue"
    ],
    default="Low Supply Low Revenue"
)

print("\nSegment counts:")
print(
    df["segment"]
    .value_counts()
)

# STEP 8: SUMMARY TABLES


print("\n" + "=" * 60)
print("8. TOP CATEGORIES")
print("=" * 60)

print("\nTop 10 by pushup revenue:")
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

print("\nTop 10 by pushup rate:")
print(
    df.sort_values(
        "pushup_rate",
        ascending=False
    )[
        [
            "category_2",
            "category_3",
            "pushup_rate"
        ]
    ].head(10)
)

print("\nTop 10 revenue per listing:")
print(
    df.sort_values(
        "revenue_per_listing",
        ascending=False
    )[
        [
            "category_2",
            "category_3",
            "revenue_per_listing"
        ]
    ].head(10)
)

# STEP 9: VISUALIZATION

print("\nGenerating plots...")

# Listings vs Revenue
plt.figure(figsize=(10, 6))

sizes = (
    df["avg_listing_price_eur"]
    * 5
)

plt.scatter(
    df["number_of_listings"],
    df["revenue_from_push_ups"],
    s=sizes,
    alpha=0.6
)

plt.xlabel("Number of Listings")
plt.ylabel("Push Up Revenue (€)")
plt.title(
    "Listings vs Push Up Revenue"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "listings_vs_revenue.png"
)

plt.close()


# Revenue per listing distribution
plt.figure(figsize=(8, 5))

plt.hist(
    df["revenue_per_listing"],
    bins=30
)

plt.xlabel(
    "Revenue per Listing"
)

plt.ylabel("Frequency")

plt.title(
    "Revenue per Listing Distribution"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "revenue_per_listing_distribution.png"
)

plt.close()


# Pushup rate distribution
plt.figure(figsize=(8, 5))

plt.hist(
    df["pushup_rate"],
    bins=30
)

plt.xlabel(
    "Push Up Rate"
)

plt.ylabel("Frequency")

plt.title(
    "Push Up Rate Distribution"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "pushup_rate_distribution.png"
)

plt.close()

# STEP 10: SAVE CLEAN DATASET

cleaned_path = (
    OUTPUT_DIR /
    "cleaned_dataset.csv"
)

df.to_csv(
    cleaned_path,
    index=False
)

print("\nCleaned dataset saved:")
print(cleaned_path.resolve())

print("\nPlots saved in:")
print(OUTPUT_DIR.resolve())

print("\nPREPROCESSING COMPLETED.")