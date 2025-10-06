Getting started
===============


### 🧩 Repository Structure

```bash
project/
│
├── data/
│   ├── raw/              ← Original noisy dataset
│   ├── interim/          ← Intermediate datasets (drop / fill NaN versions)
│   └── processed/        ← Final cleaned datasets ready for modeling
│
├── notebooks/
│   ├── 01_EDA_and_Cleaning.ipynb
│   ├── 02_EDA_Clean_Fill.ipynb
│   └── 03_EDA_Clean_Drop.ipynb
│
└── dvc.yaml              ← DVC tracking file
```

---

### ⚙️ Environment Setup

To ensure reproducibility, install all dependencies from the provided `requirements.txt` file.  
This will automatically install the correct versions of all required libraries.

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)

# Install dependencies
pip install -r requirements.txt
```
⸻

### 📦 Data Access

The dataset is versioned using DVC (Data Version Control).
To pull the latest versions of all datasets (raw, interim, processed):

```bash
dvc pull
```

You can also retrieve specific files if needed:
```bash
dvc pull data/raw/energy_noisy.csv
dvc pull data/interim/energy_drop.csv
dvc pull data/interim/energy_fill.csv
```

⸻

### 🚀 How to Run
* Notebooks version
1. Start with the main notebook:
   - EDA_and_Cleaning.ipynb → Performs raw dataset exploration, null analysis, and creates two branches (drop vs fill).
2. Continue with:
   - EDA_Clean_Fill.ipynb → Applies outlier detection and full cleaning to the Fill-NaN version.
   - 03_EDA_Clean_Drop.ipynb → Applies outlier detection and full cleaning to the Drop-NaN version.
3. All results are automatically saved and versioned via DVC under data/processed/.

* Python script version
⸻