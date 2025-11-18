# Energy Efficiency ML Pipeline

## 📋 Descripción

Este proyecto implementa un pipeline completo de Machine Learning para predecir la **Carga de Calefacción (Heating Load)** y **Carga de Refrigeración (Cooling Load)** de edificios basándose en sus características geométricas y físicas.

El proyecto sigue las mejores prácticas de **MLOps**, incluyendo:
- ✅ Pipelines de preprocesamiento con **scikit-learn**
- ✅ Tracking de experimentos con **MLflow**
- ✅ Versionado de datos y modelos con **DVC**
- ✅ Código modular siguiendo principios **SOLID**
- ✅ Documentación completa con **docstrings**
- ✅ Cumplimiento de **PEP 8** y mejores prácticas

---

## 🏗️ Arquitectura del Proyecto

```
proyecto_final/
├── config.py                    # Configuración central del proyecto
├── train.py                     # Script principal de entrenamiento
├── predict.py                   # Script principal de predicción
├── dvc.yaml                     # Pipeline DVC
│
├── configs/                     # Configuraciones de modelos
│   ├── xgb.yaml                # Config XGBoost
│   └── rf.yaml                 # Config Random Forest
│
├── data/                        # Módulos de datos
│   ├── data_loader.py          # Carga/guardado de datos
│   ├── clean_data.py           # Limpieza de datos (legacy)
│   ├── preprocessing.py        # Orchestrador de preprocesamiento
│   └── schemas.py              # Esquemas de validación
│
├── preprocessing/               # Pipeline de preprocesamiento
│   ├── transformers.py         # Transformadores personalizados
│   └── pipeline.py             # Factory de pipelines
│
├── modeling/                    # Módulos de modelado
│   ├── trainer.py              # Entrenamiento con MLflow
│   └── predictor.py            # Predicción con pipelines
│
└── scripts/                     # Scripts de utilidad
    └── run_preprocessing.py    # Preprocesamiento standalone
```

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone <repository-url>
```

### 2. Instalar dependencias con `uv` (Recomendado)

Este proyecto usa **`uv`** para gestión de dependencias (hasta 10-100x más rápido que pip).

```bash
# Instalar uv si no lo tienes
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sincronizar dependencias (crea .venv automáticamente)
uv sync
```

**¿Por qué `uv`?**
- ⚡ **Súper rápido**: Instalaciones 10-100x más rápidas que pip
- 🔒 **Reproducible**: `uv.lock` asegura versiones exactas
- 🎯 **Todo en uno**: Reemplaza pip, pip-tools, virtualenv
- 📦 **pyproject.toml**: Estándar moderno de Python (PEP 621)

**Alternativa con pip tradicional** (no recomendado):

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -e .  # Instala desde pyproject.toml
```

### 3. Configurar MLflow (opcional)

Si deseas usar un servidor MLflow remoto, edita `config.py`:

```python
MLFLOW_SERVER_IP = "your.server.ip"
MLFLOW_SERVER_PORT = 5050
```

---

## 📊 Pipeline de Datos

### Etapas del Preprocesamiento

El pipeline implementa las siguientes transformaciones en orden:

1. **Type Conversion**: Conversión de tipos y manejo de valores inválidos
2. **Missing Value Imputation**: Imputación con mediana (numéricos) y moda (categóricos)
3. **Outlier Handling**: Remoción (entrenamiento) o clipping/none (predicción) usando método IQR
4. **Categorical Cleaning**: Limpieza de categorías raras (< 1% frecuencia)
5. **Duplicate Removal**: Eliminación de filas duplicadas
6. **Feature Selection**: Remoción de features correlacionadas (X2, X4)
7. **Train-Test Split**: División estratificada (80/20)
8. **Encoding**: One-Hot Encoding para variables categóricas
9. **Scaling**: MinMax Scaling para variables numéricas

> **⚠️ IMPORTANTE:** Para predicción con modelos robustos (XGBoost/Random Forest), recomendamos **NO procesar outliers** (`handle_outliers='none'`). Ver [guía detallada](docs/OUTLIER_HANDLING.md)

### Ejecutar Preprocesamiento

```bash
# Desde la raíz del proyecto
cd proyecto_final

# Con uv (recomendado)
uv run python scripts/run_preprocessing.py

# O activar el entorno virtual primero
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
python scripts/run_preprocessing.py
```

**Outputs:**
- `data/interim/energy_efficiency_interim_clean.csv`
- `data/processed/energy_efficiency_train_prepared.csv`
- `data/processed/energy_efficiency_test_prepared.csv`
- `models/initial_cleaning_pipeline.joblib`
- `models/encoding_scaling_transformer.joblib`

---

## 🤖 Entrenamiento de Modelos

### Modelos Disponibles

#### 1. XGBoost Regressor

```bash
# Con uv (recomendado)
uv run python train.py --config configs/xgb.yaml

# O con venv activado
python train.py --config configs/xgb.yaml
```

**Hiperparámetros optimizados:**
- `n_estimators`: [100, 200, 400, 800]
- `learning_rate`: [0.05, 0.1, 0.2]
- `max_depth`: [3, 6, 8]
- `subsample`: [0.7, 0.9, 1.0]

#### 2. Random Forest Regressor

```bash
# Con uv (recomendado)
uv run python train.py --config configs/rf.yaml

# O con venv activado
python train.py --config configs/rf.yaml
```

**Hiperparámetros optimizados:**
- `n_estimators`: [100, 200, 400, 800]
- `max_depth`: [None, 5, 10, 20]
- `min_samples_split`: [2, 4, 8]
- `max_features`: [1.0, "sqrt", "log2"]

### Opciones de Entrenamiento

```bash
# Entrenamiento completo (preprocesamiento + entrenamiento)
uv run python train.py --config configs/xgb.yaml

# Solo entrenamiento (usar datos preprocesados existentes)
uv run python train.py --config configs/xgb.yaml --skip-preprocessing

# Con nombre personalizado
uv run python train.py --config configs/xgb.yaml --run-name my_experiment
```

### Tracking con MLflow

Los experimentos se registran automáticamente en MLflow con:
- ✅ Parámetros del modelo
- ✅ Métricas (MAE, RMSE, R²)
- ✅ Artefactos (modelos, pipelines)
- ✅ Resultados de HPO

**Ver experimentos:**

```bash
mlflow ui
# Navega a http://localhost:5000
```

---

## 🔮 Predicción

### Predicción Simple

```bash
# Predecir ambos targets (heating y cooling)
uv run python predict.py --input data/test_sample.csv --output predictions.json

# Predecir solo heating
uv run python predict.py --input data/test_sample.csv --target heating --output heating_preds.json

# Predecir solo cooling
uv run python predict.py --input data/test_sample.csv --target cooling --output cooling_preds.json
```

### Formato de Entrada

El archivo CSV debe contener las siguientes columnas:

```csv
X1,X2,X3,X4,X5,X6,X7,X8
0.98,514.5,294.0,110.25,7.0,2.0,0.0,0.0
0.90,563.5,318.5,122.50,7.0,3.0,0.0,0.0
```

**Descripción de variables:**
- `X1`: Relative Compactness
- `X2`: Surface Area
- `X3`: Wall Area
- `X4`: Roof Area
- `X5`: Overall Height
- `X6`: Orientation (categórica: 2, 3, 4, 5)
- `X7`: Glazing Area
- `X8`: Glazing Area Distribution (categórica: 0-5)

### Formato de Salida

```json
{
  "target": "both",
  "num_predictions": 2,
  "predictions": [
    {"heating": 15.5, "cooling": 21.3},
    {"heating": 20.8, "cooling": 28.2}
  ]
}
```

---

## 🔄 Versionado con DVC

### Inicializar DVC

```bash
cd proyecto_final
dvc init
```

### Ejecutar Pipeline Completo

```bash
# Ejecutar todas las etapas
dvc repro

# Ejecutar solo preprocesamiento
dvc repro preprocessing

# Ejecutar solo entrenamiento
dvc repro train_xgboost
```

### Trackear Datos y Modelos

```bash
# Agregar datos raw
dvc add ../data/raw/energy_efficiency_modified.csv

# Push a remote (configurar primero)
dvc remote add -d myremote s3://my-bucket/dvc-storage
dvc push
```

---

## 📈 Métricas de Evaluación

Los modelos se evalúan usando:

- **MAE (Mean Absolute Error)**: Error promedio absoluto
- **RMSE (Root Mean Squared Error)**: Raíz del error cuadrático medio
- **R² (Coefficient of Determination)**: Proporción de varianza explicada

### Resultados de Referencia (XGBoost)

| Target | MAE | RMSE | R² |
|--------|-----|------|----|
| Heating Load | 0.68 | 4.30 | 0.958 |
| Cooling Load | 1.15 | 7.20 | 0.917 |

---

## 🧪 Testing

```bash
# Ejecutar tests unitarios
uv run pytest tests/

# Con coverage
uv run pytest --cov=proyecto_final tests/
```

---

## 📚 Documentación del Código

Todo el código está completamente documentado siguiendo el estilo **Google Docstrings**:

```python
def train_single_target(
    self,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_name: str,
    run_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train model for a single target variable.

    This method orchestrates:
    1. Model creation
    2. Hyperparameter optimization (if enabled)
    3. Training
    4. Evaluation
    5. MLflow logging
    6. Model saving

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        target_name: Name of target variable (for logging)
        run_name: Optional custom run name

    Returns:
        Dictionary with metrics and model info
    """
```

---

## 🛠️ Mejores Prácticas Implementadas

### SOLID Principles

- **S**ingle Responsibility: Cada clase tiene una responsabilidad única
- **O**pen/Closed: Extensible sin modificar código existente
- **L**iskov Substitution: Transformadores intercambiables
- **I**nterface Segregation: Interfaces mínimas y específicas
- **D**ependency Inversion: Dependencias inyectadas

### DRY (Don't Repeat Yourself)

- Código reutilizable en módulos
- Transformadores genéricos
- Configuración centralizada

### KISS (Keep It Simple, Stupid)

- Funciones pequeñas y enfocadas
- Lógica clara y directa
- Nombres descriptivos

### PEP 8 Compliance

- Formato consistente
- Naming conventions
- Imports organizados
- Line length < 100 caracteres

---

## 🐛 Troubleshooting

### Error: MLflow tracking URI not accessible

**Solución**: Configura un servidor local:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

### Error: Pipeline artifacts not found

**Solución**: Ejecuta primero el preprocesamiento:

```bash
uv run python scripts/run_preprocessing.py
```

### Error: Dependencias faltantes

**Solución**: Sincroniza las dependencias con `uv`:

```bash
# Reinstalar todas las dependencias
uv sync --reinstall

# O instalar una dependencia específica
uv add xgboost
```

---

## 📦 Gestión de Dependencias

Este proyecto usa `uv` con los siguientes archivos:

- **`pyproject.toml`**: Especifica las dependencias del proyecto (estándar PEP 621)
- **`uv.lock`**: Lock file con versiones exactas de todas las dependencias (equivalente a `requirements.txt` + `pip freeze`)

**Comandos útiles:**

```bash
# Sincronizar dependencias (lee pyproject.toml y uv.lock)
uv sync

# Agregar una nueva dependencia
uv add <package>

# Agregar dependencia de desarrollo
uv add --dev <package>

# Actualizar todas las dependencias
uv lock --upgrade

# Exportar a requirements.txt (si es necesario)
uv pip freeze > requirements.txt
```

**Migración desde pip:**

Si estás migrando desde `pip` + `requirements.txt`, `uv` puede leer `pyproject.toml` directamente. No necesitas `requirements.txt` ni `requirements-dev.txt`.

---

## Orquestación

**Schema de entrada/salida de la API**
...

**Modelo y artefactos**

- Artefacto principal del modelo:
  - `models:/energy-efficiency/xgboost/0.1.0`

- Archivos empaquetados en la imagen Docker bajo `/app/models`:
  - `initial_cleaning_pipeline.joblib`
  - `encoding_scaling_transformer.joblib`
  - `xgboost_heating_model.joblib`
  - `xgboost_cooling_model.joblib`

### Construcción y ejecución del contenedor

 - Construir la imagen
docker build -t ml-service:latest .

 - Ejecutar el contenedor
docker run --rm -p 8000:8000 ml-service:latest

### Imágenes Docker (Docker Hub)

Las imágenes se publican en Docker Hub bajo:

- `<user>/ml-service:0.1.0` – primera versión estable del servicio
- `<user>/ml-service:latest` – alias a la versión estable más reciente.

## 📖 Referencias

- [Dataset UCI](https://archive.ics.uci.edu/dataset/242/energy+efficiency)
- [Scikit-learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [uv Documentation](https://docs.astral.sh/uv/)

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT.

---