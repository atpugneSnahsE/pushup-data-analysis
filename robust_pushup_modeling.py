import pandas as pd
import numpy as np
from pathlib import Path

import statsmodels.api as sm
import statsmodels.formula.api as smf


# =====================================================
# CONFIG
# =====================================================

CSV_PATH = (
    "preprocessing_outputs/"
    "cleaned_dataset.csv"
)

PUSHUP_PRICE = 2

OUTPUT_DIR = Path(
    "robust_model_outputs"
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
# FEATURE ENGINEERING
# =====================================================

print("\nCreating features...")

df["estimated_pushups_sold"] = (
    df["revenue_from_push_ups"]
    / PUSHUP_PRICE
)

df["pushup_rate"] = (
    df["estimated_pushups_sold"]
    / df["number_of_listings"]
)

df["relative_promo_cost"] = (
    PUSHUP_PRICE
    / df["avg_listing_price_eur"]
)

df["log_listings"] = np.log1p(
    df["number_of_listings"]
)

# log transformed target
df["log_pushup_rate"] = np.log1p(
    df["pushup_rate"]
)

print("Features created.")


# =====================================================
# MODEL 1
# STANDARD OLS
# =====================================================

print("\n" + "=" * 70)
print("1. STANDARD OLS")
print("=" * 70)

X = df[
    [
        "avg_listing_price_eur",
        "log_listings",
        "relative_promo_cost"
    ]
]

X = sm.add_constant(X)

y = df["pushup_rate"]

ols_model = sm.OLS(y, X).fit()

print(ols_model.summary())


# =====================================================
# MODEL 2
# ROBUST STANDARD ERRORS
# =====================================================

print("\n" + "=" * 70)
print("2. OLS WITH HC3 ROBUST ERRORS")
print("=" * 70)

robust_model = sm.OLS(
    y, X
).fit(cov_type="HC3")

print(robust_model.summary())


# =====================================================
# MODEL 3
# LOG TARGET MODEL
# =====================================================

print("\n" + "=" * 70)
print("3. LOG TRANSFORMED TARGET")
print("=" * 70)

y_log = df["log_pushup_rate"]

log_model = sm.OLS(
    y_log,
    X
).fit(cov_type="HC3")

print(log_model.summary())


# =====================================================
# MODEL 4
# QUANTILE REGRESSION
# =====================================================

print("\n" + "=" * 70)
print("4. QUANTILE REGRESSION")
print("=" * 70)

quantile_model = smf.quantreg(
    """
    pushup_rate
    ~ avg_listing_price_eur
    + log_listings
    + relative_promo_cost
    """,
    df
).fit(q=0.5)

print(
    quantile_model.summary()
)


# =====================================================
# COEFFICIENT COMPARISON
# =====================================================

print("\n" + "=" * 70)
print("5. MODEL COMPARISON")
print("=" * 70)

comparison = pd.DataFrame({

    "OLS":
    ols_model.params,

    "OLS_HC3":
    robust_model.params,

    "Log_OLS":
    log_model.params,

    "Quantile":
    quantile_model.params
})

print(comparison)

comparison.to_csv(
    OUTPUT_DIR /
    "coefficient_comparison.csv"
)

print(
    "\nCoefficient comparison saved."
)


# =====================================================
# INTERPRETATION CHECK
# =====================================================

print("\n" + "=" * 70)
print("6. DIRECTIONAL STABILITY CHECK")
print("=" * 70)

variables = [
    "avg_listing_price_eur",
    "log_listings",
    "relative_promo_cost"
]

for var in variables:

    signs = [

        np.sign(
            ols_model.params[var]
        ),

        np.sign(
            robust_model.params[var]
        ),

        np.sign(
            log_model.params[var]
        ),

        np.sign(
            quantile_model.params[var]
        )
    ]

    stable = (
        len(set(signs))
        == 1
    )

    print(
        f"{var}: "
        f"{'Stable' if stable else 'Unstable'}"
    )


# =====================================================
# SAVE REPORT
# =====================================================

with open(
    OUTPUT_DIR /
    "model_interpretation.txt",
    "w"
) as f:

    f.write(
        "Regression Robustness "
        "Summary\n"
    )

    f.write(
        "=" * 50 + "\n\n"
    )

    f.write(
        "OLS assumptions "
        "were imperfect due "
        "to skewness and "
        "heavy tails.\n\n"
    )

    f.write(
        "Robust standard "
        "errors, log "
        "transformation, "
        "and quantile "
        "regression were "
        "used to evaluate "
        "directional "
        "stability.\n\n"
    )

    f.write(
        "If coefficient "
        "directions remain "
        "stable across "
        "models, findings "
        "are considered "
        "robust.\n"
    )

print("\nOutputs saved:")
print(OUTPUT_DIR.resolve())

print(
    "\nROBUST MODELING "
    "COMPLETED."
)