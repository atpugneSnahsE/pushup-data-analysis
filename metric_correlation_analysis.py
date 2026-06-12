import pandas as pd
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
    "metric_correlation_outputs"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# =====================================================
# LOAD
# =====================================================

df = pd.read_csv(CSV_PATH)

target_metric = "pushup_rate"

candidate_metrics = [
    "number_of_listings",
    "avg_listing_price_eur",
    "estimated_pushups_sold",
    "revenue_per_listing",
    "revenue_from_push_ups"
]


print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

correlations = {}

for metric in candidate_metrics:

    corr = (
        df[target_metric]
        .corr(df[metric])
    )

    correlations[metric] = corr

    print(
        f"{metric}: "
        f"{corr:.4f}"
    )


# =====================================================
# BEST METRIC
# =====================================================

best_metric = max(
    correlations,
    key=lambda x: abs(
        correlations[x]
    )
)

best_corr = correlations[
    best_metric
]

print("\n" + "=" * 60)
print("BEST CORRELATED METRIC")
print("=" * 60)

print(
    f"\nBest metric: "
    f"{best_metric}"
)

print(
    f"Correlation: "
    f"{best_corr:.4f}"
)


# =====================================================
# GRAPH
# =====================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df[best_metric],
    df[target_metric],
    alpha=0.7
)

plt.xlabel(best_metric)

plt.ylabel("Push Up Rate")

plt.title(
    f"Push Up Rate vs "
    f"{best_metric}"
)

plt.tight_layout()

graph_path = (
    OUTPUT_DIR /
    "best_metric_relationship.png"
)

plt.savefig(graph_path)

plt.close()

print(
    "\nGraph saved:"
)

print(
    graph_path.resolve()
)