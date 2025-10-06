# Energy Efficiency Dataset – EDA and Initial Cleaning

## 1. Introduction
This notebook documents the **Exploratory Data Analysis (EDA)** and the **initial cleaning phase** for the Energy Efficiency dataset (UCI).  
The dataset describes the geometrical and physical characteristics of buildings (`X1–X8`) and their energy loads (`Y1`, `Y2`).  

The purpose of this first notebook is to:
- Explore the raw noisy dataset.
- Detect and understand missing, invalid, and inconsistent values.
- Decide how to handle missing data, generating **two candidate cleaning approaches**:
  1. **Drop missing values** (strict version).
  2. **Fill missing values** (preserve data version).  

Subsequent notebooks will handle deeper cleaning, outlier detection, and final preprocessing steps for each branch.

---

## 2. Data Loading and Overview

The raw dataset (`data/raw/energy_efficiency_modified.csv`) was loaded using pandas.
The original dataset is (`data/raw/energy_efficiency_original.csv`) and the documentation is available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/242/energy+efficiency).

- **Initial shape:** (783, 11)  
- **Columns:** `X1`–`X8`, `Y1`, `Y2`, and `mixed_type_col` (extra column with mixed content).  
- **Initial data types:** Mostly `object` due to text-based noise and formatting issues.  

Basic exploratory statistics and value inspection revealed:
- Several non-numeric values such as `"error"` or blank entries.
- Unexpectedly wide numeric ranges suggesting noise (e.g., extreme values in `X2`, `Y1`, and `Y2`).

---

## 3. Data Cleaning (Initial Phase)
### **Type correction**
- Converted all numeric features using `pd.to_numeric(errors='coerce')`.
- Invalid textual entries were replaced with `NaN`.

### **Null inspection**
The following variables contained missing or invalid entries:
| Variable | Missing count | % of total |
|-----------|----------------|-------------|
| X5 | 19 | 2.43% |
| X7 | 14 | 1.79% |
| X3 | 13 | 1.66% |
| X6 | 13 | 1.66% |
| X8 | 11 | 1.40% |
| X2 | 10 | 1.28% |
| X4 | 10 | 1.28% |
| X1 | 9 | 1.15% |
| Y2 | 9 | 1.15% |
| Y1 | 7 | 0.89% |

No variables were fully empty, so both strategies (drop vs. fill) were feasible.

---

## 4. Missing-Value Handling Strategies
To preserve reproducibility and facilitate comparison, two alternative cleaning strategies were implemented:

1. **Drop approach (`df_drop`)**  
   Rows with any `NaN` values were removed using `df.dropna()`.  
   - Shape after cleaning: **(680, 10)**  
   - Pros: Ensures complete data consistency.  
   - Cons: Reduces dataset size by ~13%.

2. **Fill approach (`df_fill`)**  
   Missing numeric values were filled using the **median** of each column (`df.fillna(df.median())`).  
   - Shape after cleaning: **(783, 10)**  
   - Pros: Retains full dataset size.  
   - Cons: Slightly alters original value distributions.

The column `mixed_type_col` was dropped from both versions due to irreparable inconsistencies.

---

## 5. Comparison Overview
A comparison of both datasets was documented for future analysis:
| Dataset | Rows | NaNs Remaining | Description |
|----------|------|----------------|--------------|
| `df_drop` | 680 | 0 | Conservative cleaning, fewer records. |
| `df_fill` | 783 | 0 | Retains all rows with imputed values. |

Both datasets were stored for versioning under:
- `data/interim/energy_drop.csv`
- `data/interim/energy_fill.csv`

Each will be analyzed in separate notebooks focusing on:
- Outlier detection and treatment.  
- Feature distribution analysis.  
- Correlation and statistical relationships.  

---