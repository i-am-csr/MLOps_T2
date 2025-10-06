# Energy Efficiency Dataset – EDA and Cleaning (Fill-NaN Version)

## 1. Introduction
This notebook continues the data preparation process for the **Energy Efficiency Dataset (UCI)**,  
focusing on the version where missing values were **filled** instead of dropped.  

In the previous notebook, two strategies were defined to handle missing values:
1. **Drop-NaN version:** rows with missing data were removed.
2. **Fill-NaN version:** missing values were filled with column medians.

This notebook explores and cleans the **Fill-NaN version**, aiming to:
- Detect and remove outliers.
- Clean inconsistent categorical variables (`X6`, `X8`).
- Perform exploratory visualizations to confirm data consistency.
- Prepare the dataset for preprocessing and modeling.

---

## 2. Dataset Overview
The starting dataset (`data/interim/energy_fill.csv`) was loaded and inspected.

- **Shape:** (783, 10)  
- **Columns:** `X1`–`X8`, `Y1`, `Y2`.  
- **Missing values:** none (all previously filled).  
- **Types:** all continuous columns correctly converted to `float64`, while `X6` and `X8` are discrete numeric.

A preliminary statistical summary (`df.describe()`) revealed that some columns still contained extreme numeric values, suggesting the presence of outliers.

---

## 3. Outlier Detection and Removal (IQR Method)
Outliers were detected using the **Interquartile Range (IQR)** method applied to continuous numeric columns:  
`X1`, `X2`, `X3`, `X4`, `X5`, `X7`, `Y1`, and `Y2`.

For each variable:
\[
IQR = Q3 - Q1,\quad \text{Lower Bound} = Q1 - 1.5 \times IQR,\quad \text{Upper Bound} = Q3 + 1.5 \times IQR
\]
Rows containing values outside these bounds were removed.

**Results:**
- Initial rows: 783  
- After filtering: 614  
- Columns most affected: `X2`, `X3`, `X4`, `Y1`, `Y2`.  

### Validation
Boxplots before and after filtering confirmed that:
- Extreme geometric and load values (e.g., 46,000+ for `X2`, >1,600 for `Y1`) were removed.
- Remaining distributions show balanced ranges with no excessive skewness.

---

## 4. Categorical Cleaning (`X6`, `X8`)
Both categorical features (`X6` = Orientation, `X8` = Glazing Area Distribution) contained **invalid codes** introduced by noise.

### **Detected invalid values:**
- `X6`: contained {44, 72, 244, 420, 524, ...}  
- `X8`: contained {155, 316, 971, ...}

### **Fix applied:**
Only valid category codes were retained:
- `X6 ∈ {2, 3, 4, 5}`  
- `X8 ∈ {0, 1, 2, 3, 4, 5}`  

After cleaning:
- **Shape:** (614, 10)  
- `X6` unique values: [2, 3, 4, 5]  
- `X8` unique values: [0, 1, 2, 3, 4, 5]

### **Verification**
Countplots confirmed consistent category distributions with all invalid classes removed.

---

## 5. Exploratory Data Analysis

### **Histograms**
Histograms were plotted for all continuous features (`X1–X5`, `X7`, `Y1`, `Y2`):
- All distributions now appear compact and consistent.
- Skewness is low across all features.
- No evidence of new missing or extreme values after filtering.

### **Boxplots**
Boxplots reaffirmed that all outliers were successfully removed — whiskers are well-defined,  
and all features exhibit coherent ranges.

### **Categorical Countplots**
- `X6`: evenly distributed across four orientations (2–5).  
- `X8`: balanced distribution across glazing configurations (0–5).

All categorical distributions now align with expected dataset definitions.

---

## 6. Correlation Analysis
A **Pearson correlation heatmap** was computed for continuous variables (`X1–X5`, `X7`, `Y1`, `Y2`).

**Key insights:**
- **X1 vs X2:** Strong negative correlation (−0.99), consistent with geometric properties of compactness and surface area.  
- **X5 vs Y1/Y2:** Strong positive correlation (~0.9), indicating taller buildings have higher heating/cooling demands.  
- **Y1 vs Y2:** Very high correlation (0.98), confirming both targets are closely related.  
- **X7 (Glazing Area):** Weak correlation with all features (impact likely nonlinear).

These relationships confirm the physical and statistical consistency of the cleaned dataset.

---

## 7. Data Saving and Versioning
After cleaning, the final filled dataset was saved under: