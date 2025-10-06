# Energy Efficiency Dataset – EDA and Cleaning (Drop-NaN Version)

## 1. Introduction
This notebook continues the cleaning and analysis workflow for the **Energy Efficiency Dataset (UCI)**,  
focusing on the version where all rows containing missing values were **dropped**.  

In the previous notebook, two strategies were defined:
1. **Drop-NaN version:** remove rows containing missing data.
2. **Fill-NaN version:** fill missing values using median imputation.

This notebook explores and processes the **Drop-NaN version**, emphasizing data integrity and precision.  
The goal is to:
- Detect and remove outliers using statistical methods.
- Validate distributions through visual analysis.
- Ensure categorical variables (`X6`, `X8`) contain only valid categories.
- Prepare a highly reliable dataset for later modeling.

---

## 2. Dataset Overview
The dataset used here (`data/interim/energy_drop.csv`) was loaded and reviewed.

- **Shape:** (680, 10)  
- **Columns:** `X1`–`X8`, `Y1`, `Y2`.  
- **Missing values:** none (all removed).  
- **Data types:** continuous features are `float64`; `X6` and `X8` are discrete integers.  

Compared to the original noisy dataset (783 rows), 103 records were eliminated due to missing or invalid entries.  
This version is smaller but guarantees full data consistency across features.

---

## 3. Outlier Detection and Removal (IQR Method)
Outliers were identified and removed using the **Interquartile Range (IQR)** method applied to continuous features:  
`X1`, `X2`, `X3`, `X4`, `X5`, `X7`, `Y1`, and `Y2`.

For each numeric feature:
\[
IQR = Q3 - Q1,\quad \text{Lower Bound} = Q1 - 1.5 \times IQR,\quad \text{Upper Bound} = Q3 + 1.5 \times IQR
\]

Rows containing values outside these bounds were removed to ensure realistic numeric ranges.

**Results:**
- **Rows before filtering:** 680  
- **Rows after filtering:** 542  
- **Most affected features:** `X2`, `X3`, `X4`, and `Y1`, `Y2`.  

### **Visual validation**
Boxplots and histograms show:
- Extreme values previously stretching the x-axis were eliminated.
- All variables now exhibit compact, smooth distributions.
- Skewness and kurtosis values have normalized across features.

---

## 4. Categorical Cleaning (`X6`, `X8`)
Both categorical variables contained noise-induced invalid category codes.

### **Invalid categories detected:**
- `X6` contained additional numeric values (e.g., 44, 72, 155, 420, 971).  
- `X8` contained out-of-range codes (e.g., 155, 316, 971).

### **Filtering strategy:**
Only valid category codes were retained:
- `X6 ∈ {2, 3, 4, 5}`  
- `X8 ∈ {0, 1, 2, 3, 4, 5}`  

After filtering:
- **Shape:** (542, 10)  
- `X6` unique values: [2, 3, 4, 5]  
- `X8` unique values: [0, 1, 2, 3, 4, 5]

### **Verification**
Countplots of both variables confirm the removal of invalid categories, leaving only consistent, well-defined classes.

---

## 5. Exploratory Data Analysis

### **Histograms**
Histograms for all continuous variables (`X1–X5`, `X7`, `Y1`, `Y2`) show:
- Clear, normally distributed shapes without heavy tails.
- Homogeneous scales across variables.
- Expected physical patterns (e.g., compactness concentrated around 0.7–0.9).

### **Boxplots**
Boxplots show well-defined whiskers and no isolated points, confirming that all outliers were removed successfully.  

### **Categorical Countplots**
- `X6` displays balanced frequency across the four valid orientations (2–5).  
- `X8` exhibits a near-uniform distribution across glazing configurations (0–5).  

---

## 6. Correlation Analysis
A **Pearson correlation heatmap** was computed using all continuous variables (`X1–X5`, `X7`, `Y1`, `Y2`).

**Main findings:**
- **X1 vs X2:** Strong negative correlation (−0.99), consistent with physical geometry (compactness vs surface area).  
- **X5 vs Y1/Y2:** Strong positive correlation (~0.9), showing that taller buildings have higher energy demand.  
- **Y1 vs Y2:** Very high correlation (0.98), meaning heating and cooling loads are strongly related.  
- **X7 (Glazing Area):** Weak correlation with others, confirming its limited linear effect.

This matrix validates the dataset’s internal consistency and the success of the cleaning process.

---

## 7. Data Saving and Versioning
The final cleaned dataset was saved under: