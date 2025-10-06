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

To ensure reproducibility, use [uv](https://github.com/astral-sh/uv), a fast Python package manager and resolver. uv can automatically create and activate your virtual environment, install dependencies, and synchronize your environment with the lock file.

```bash
# Create and activate a virtual environment using uv
uv venv

# Install dependencies and synchronize with lock file
uv pip sync
```

If you need to add a new package, use:
```bash
uv pip install <package-name>
```
This will automatically update requirements.txt and uv.lock.

If you want to upgrade all dependencies to their latest compatible versions:
```bash
uv pip compile --upgrade
uv pip sync
```

For more details, see the [uv documentation](https://github.com/astral-sh/uv).

---

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
