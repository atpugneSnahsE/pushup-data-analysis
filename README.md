# Push Up Pricing and Performance Analysis

This repository contains a structured analytical workflow for evaluating the performance, adoption behavior, and monetization strategy of the marketplace push up feature.

The goal of the analysis was not only to understand current push up performance, but also to identify behavioral drivers, pricing inefficiencies, and statistically defensible monetization opportunities.

The analysis was intentionally structured in stages. Each script answers a specific analytical question and builds on findings from the previous stage.

---

# Project Objective

The marketplace currently charges a fixed €2 fee for the push up feature, which temporarily increases listing visibility.

The main analytical objectives were:

1. Understand the structure and quality of the dataset.
2. Measure seller interest in the push up feature.
3. Identify behavioral drivers of push up adoption.
4. Evaluate whether the €2 pricing model is equally effective across categories.
5. Explore alternative pricing strategies.
6. Validate findings using statistically robust methods.
7. Propose an experimentation framework for pricing changes.

---

# Analysis Workflow

The project follows a sequential analytical pipeline:

```text
EDA
 ↓
Preprocessing
 ↓
Feature Adoption Analysis
 ↓
Correlation Analysis
 ↓
Pricing Strategy Evaluation
 ↓
Advanced Statistical Analysis
 ↓
Robustness Validation
```

Each stage exists to answer a different business or statistical question.

---

# 1. `eda_pushup_analysis.py`

## Purpose

Exploratory Data Analysis (EDA).

This script was created to understand the structure, quality, and characteristics of the dataset before any modeling decisions were made.

Performing EDA first is essential because assumptions made during later analysis depend on understanding:

* Data completeness
* Distributional properties
* Missingness
* Category structure
* Potential anomalies

Skipping this step could result in incorrect assumptions and misleading conclusions.

---

## Why this step was necessary

Before measuring seller behavior or pricing performance, it was important to understand:

### What does one row represent?

The dataset was aggregated at:

```text
category_2 × category_3
```

granularity.

Meaning:

Each row represented a marketplace subcategory rather than individual seller or listing behavior.

This immediately defined what types of conclusions were possible.

For example:

Possible:

* Category level monetization
* Adoption behavior

Not possible:

* Seller level causal inference
* Individual conversion analysis

---

## What this script analyzed

### Dataset overview

Reason:

To understand:

* number of observations
* variable types
* category structure

This helps determine whether transformations are needed.

---

### Missing value analysis

Reason:

Missing values can bias analysis.

Example:

Missing revenue could represent:

* failed data collection
* true zero purchases

This distinction matters for monetization analysis.

---

### Duplicate analysis

Reason:

Duplicate category rows artificially inflate results.

Removing duplicates prevents:

* biased revenue totals
* misleading adoption rates

---

### Descriptive statistics

Reason:

Summary statistics reveal:

* skewness
* outliers
* marketplace concentration

This step later justified the use of:

* Spearman correlation
* robust regression

instead of relying purely on normality assumptions.

---

### Correlation exploration

Reason:

Initial relationships were explored to identify candidate explanatory variables.

Example:

Does listing volume relate to push up performance?

This helped guide later feature engineering.

---

# 2. `preprocessing_analysis.py`

## Purpose

Data cleaning and preparation.

The objective of preprocessing was to transform raw data into an analysis ready dataset.

This stage ensures downstream statistical results are trustworthy.

---

## Why preprocessing was necessary

Raw marketplace data frequently contains:

* missing values
* duplicates
* inconsistencies
* invalid observations

Statistical models are highly sensitive to poor quality data.

Therefore:

Cleaning decisions were made before any behavioral conclusions.

---

## Key preprocessing steps

### Removing duplicates

Reason:

Duplicate category entries distort:

* revenue
* adoption rates
* averages

Exact duplicates were removed to prevent double counting.

---

### Handling missing categories

Reason:

Missing category labels make segmentation impossible.

These rows were removed because they could not contribute meaningfully to category based analysis.

---

### Handling missing revenue

Reason:

Push up revenue was missing in several observations.

Assumption made:

Missing revenue = zero push up purchases.

Why?

Because:

No revenue is economically more plausible than undefined monetization.

This assumption was explicitly documented.

---

### Feature engineering

New variables were introduced to better represent seller behavior.

#### Estimated Push Ups Sold

Reason:

The dataset did not contain actual purchase counts.

Since push ups cost €2:

Estimated purchases were inferred from revenue.

Purpose:

Approximate feature usage.

---

#### Push Up Adoption Rate

Reason:

Revenue alone favors large categories.

Large categories naturally generate more revenue.

To normalize adoption:

Push up purchases were scaled relative to listing volume.

Purpose:

Measure seller interest independent of category size.

---

#### Revenue per Listing

Reason:

To measure monetization efficiency.

Purpose:

Understand which categories monetize visibility most effectively.

---

# 3. `feature_interest_analysis.py`

## Purpose

Measure seller interest in the push up feature.

This script directly addresses the business question:

```text
Are sellers genuinely interested in push ups?
```

---

## Why this step was necessary

Revenue alone cannot measure interest.

Example:

A large category may generate high revenue simply due to marketplace size.

This would overestimate feature popularity.

Instead:

Adoption had to be normalized.

---

## Metric selection

### Push Up Adoption Rate

Selected because it measures:

```text
push ups purchased
relative to
marketplace supply
```

This provides a better approximation of seller willingness to pay.

---

## Category ranking

Reason:

To identify:

* strongest performing categories
* weakest performing categories

This helps reveal:

Behavioral heterogeneity across marketplace segments.

Finding:

Higher value categories often showed stronger adoption.

---

## Special category analysis

`GIRLS_CLOTHING / FOR_BABIES`

Reason:

The push up fee exceeded half the listing price.

This appeared economically irrational.

The analysis explored:

* urgency effects
* seller convenience
* bundle selling
* marketplace competition

Purpose:

Move beyond pure numerical interpretation.

---

# 4. `metric_correlation_analysis.py`

## Purpose

Identify variables associated with push up adoption.

The objective was to understand:

```text
What drives seller interest?
```

---

## Why this step was necessary

Simple observation cannot explain seller behavior.

Correlation analysis provides evidence regarding:

Potential behavioral mechanisms.

---

## Initial hypothesis

Hypothesis:

Higher competition drives push up purchases.

Proxy:

`number_of_listings`

Reason:

More listings imply more seller competition.

---

## Why the hypothesis changed

Result:

Near zero relationship.

Interpretation:

Competition alone does not explain seller decisions.

This motivated a stronger analytical approach.

---

# 5. `pricing_strategy_analysis.py`

## Purpose

Evaluate implications of alternative pricing strategies.

This script addressed:

```text
What happens if the €2 price changes?
```

---

## Why this step was necessary

The assignment required evaluating:

* price increases
* price decreases
* monetization tradeoffs

---

## Why simulations were used

The dataset contains:

No historical price changes.

Therefore:

True elasticity estimation was impossible.

Instead:

Scenario based revenue modeling was used.

Purpose:

Explore plausible marketplace outcomes.

Important:

These results should be interpreted as:

Directional evidence.

Not causal proof.

---

## Business logic evaluated

### Lower prices

Potential benefits:

* increased adoption
* better accessibility

Risks:

* lower revenue
* visibility inflation

---

### Higher prices

Potential benefits:

* better monetization

Risks:

* reduced adoption
* fairness concerns

---

# 6. `advanced_pushup_analysis.py`

## Purpose

Upgrade the statistical rigor of the analysis.

This script addressed methodological weaknesses in earlier stages.

---

## Why Pearson correlation alone was insufficient

Marketplace data showed:

* skewness
* outliers
* heavy tails

Pearson assumes:

Linear relationships.

Therefore:

Spearman correlation was added.

Reason:

Spearman is more robust for non normal marketplace data.

---

## Relative Promotion Cost

Reason for introduction:

A simple relationship between price and adoption only describes a pattern.

It does not explain the mechanism.

Relative promotion cost explains:

```text
How expensive push ups feel
relative to item value
```

This became the strongest behavioral predictor.

---

## Multivariate regression

Reason:

Seller behavior is multi factor.

Variables interact.

Regression allows simultaneous evaluation of:

* listing value
* competition
* affordability

Purpose:

Estimate which factors jointly influence adoption.

---

## Price band segmentation

Reason:

Arbitrary thresholds are difficult to justify.

Instead:

Quartile based segmentation was used.

Purpose:

Create statistically defensible category tiers.

---

## Power analysis

Reason:

A/B testing without sufficient sample size risks:

False negatives.

Purpose:

Estimate required observations for meaningful experimentation.

---

## Cluster randomization discussion

Reason:

Marketplace experiments violate independence assumptions.

Visibility is competitive.

Treating one seller changes outcomes for others.

Purpose:

Design more realistic experimentation.

---

# 7. `robust_pushup_modeling.py`

## Purpose

Validate whether conclusions remain stable under stronger statistical assumptions.

This script addresses the largest remaining methodological concern:

```text
Can we trust the regression findings?
```

---

## Why robustness checks were necessary

Initial OLS diagnostics showed:

* severe skewness
* extreme kurtosis
* non normal residuals

This weakens confidence in:

* p values
* standard errors

---

## HC3 robust standard errors

Reason:

Correct for heteroscedasticity.

Purpose:

More reliable statistical inference.

---

## Log transformed target

Reason:

Marketplace adoption showed heavy skewness.

Purpose:

Reduce influence of extreme observations.

---

## Quantile regression

Reason:

Marketplace behavior is highly heterogeneous.

Quantile regression estimates:

Median seller behavior.

Purpose:

Avoid reliance on normal residual assumptions.

---

## Coefficient stability analysis

Reason:

If findings change dramatically across models:

Results may not be trustworthy.

Purpose:

Check whether:

* signs remain stable
* behavioral interpretation remains consistent

Finding:

Relationships remained directionally stable across models.

This substantially increased confidence in the analysis.

---

# Dataset

The analysis uses:

```text
Pricing Push Ups - data.csv
```

---

# Technologies

The project was implemented using:

* Python
* Pandas
* NumPy
* Matplotlib
* SciPy
* Statsmodels

Purpose of tools:

### Pandas

Data loading and preprocessing.

### NumPy

Numerical operations and transformations.

### Matplotlib

Visualization.

### SciPy

Statistical testing.

### Statsmodels

Regression modeling and inference.

---

# Final Note

The project evolved iteratively.

Earlier findings motivated stronger analytical approaches.

Each stage was designed to either:

1. Improve understanding of marketplace behavior.

or

2. Strengthen confidence in the statistical conclusions.

The final workflow balances:

* business interpretability
* statistical rigor
* practical marketplace relevance
