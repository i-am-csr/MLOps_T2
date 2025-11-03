from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal
from time import strftime

from mlflow import MlflowClient
from pandas import DataFrame, to_numeric
import os

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MLFLOW_SERVER_IP = "138.68.15.126"
MLFLOW_SERVER_PORT = 5050
# MLFLOW_TRACKING_URI = f"http://{MLFLOW_SERVER_IP}:{MLFLOW_SERVER_PORT}"
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"http://{MLFLOW_SERVER_IP}:{MLFLOW_SERVER_PORT}"
)

DEFAULT_MODEL_URI: Optional[str] = None

@dataclass(frozen=True)
class DataConfig:
    csv_path: str
    features: List[str]
    targets: Dict[str, str]

DATA = DataConfig(
    csv_path = "../data/processed/energy_efficiency_prepared.csv",
    features = ["X1","X3","X5","X7","X6_3","X6_4","X6_5","X8_1","X8_2","X8_3","X8_4","X8_5"],
    targets = {"heating": "Y1", "cooling": "Y2"},
)

@dataclass(frozen=True)
class MlflowConfig:
    tracking_uri: str
    experiment: str
    register_name: str

MLFLOW = MlflowConfig(
    tracking_uri = MLFLOW_TRACKING_URI,
    experiment = "EnergyEfficiency",
    register_name = "energy-efficiency-regressor",
)

@dataclass
class ModelConfig:
    name: str
    library: Literal["xgboost_sklearn", "sklearn"]
    params: Dict

@dataclass
class SplitConfig:
    test_size: float = 0.2
    random_state: int = 42

@dataclass
class HpoConfig:
    enabled: bool = False
    search: Literal["random", "grid"] = "random"
    cv: int = 5
    scoring: str = "neg_root_mean_squared_error"
    n_jobs: int = -1
    n_iter: int = 30
    param_grid: Optional[Dict] = None

@dataclass
class TrainConfig:
    model: ModelConfig
    split: SplitConfig
    hpo: HpoConfig

def nowstamp() -> str:
    return strftime("%Y-%m-%d_%H-%M-%S")

def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train_config_from_yaml(path: str) -> TrainConfig:
    raw = load_yaml(path)
    m = raw["model"]
    s = raw["split"]
    h = raw.get("hpo", {})
    return TrainConfig(
        model = ModelConfig(
            name = m["name"],
            library = m["library"],
            params = dict(m["params"])
        ),
        split = SplitConfig(
            test_size = float(s.get("test_size", 0.2)),
            random_state = int(s.get("random_state", 42))
        ),
        hpo = HpoConfig(
            enabled = bool(h.get("enabled", False)),
            search = h.get("search", "random"),
            cv = int(h.get("cv", 5)),
            scoring = h.get("scoring", "neg_root_mean_squared_error"),
            n_jobs = int(h.get("n_jobs", -1)),
            n_iter = int(h.get("n_iter", 30)),
            param_grid = h.get("param_grid", None),
        ),
    )

def setup_mlflow():
    """
    Configura MLflow, usando un servidor remoto si está disponible.
    """
    import mlflow
    from loguru import logger
    from pathlib import Path

    tracking_uri = MLFLOW_TRACKING_URI.strip()

    # Si es un servidor remoto (http/https)
    if tracking_uri.startswith("http"):
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"[MLflow] Usando servidor remoto: {tracking_uri}")
    else:
        # Ruta local
        tracking_path = Path(tracking_uri.replace("file://", ""))
        tracking_path.mkdir(parents=True, exist_ok=True)
        resolved_uri = f"file:{tracking_path.resolve()}"
        mlflow.set_tracking_uri(resolved_uri)
        logger.info(f"[MLflow] Usando tracking local: {resolved_uri}")

    # Configurar experimento
    if MLFLOW.experiment:
        client = MlflowClient()
        exp = client.get_experiment_by_name(MLFLOW.experiment)
        if exp is None:
            client.create_experiment(
                name=MLFLOW.experiment,
                artifact_location="mlflow-artifacts:/"
            )
        mlflow.set_experiment(MLFLOW.experiment)
        logger.info(f"[MLflow] Experimento activo: {MLFLOW.experiment}")


def to_numeric_df(df: DataFrame) -> DataFrame:
    return df.apply(to_numeric, errors="coerce")

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
