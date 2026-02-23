from pathlib import Path
from rich.console import Console
from src.utils.logging import Logger
from torch.cuda import is_available as torch_cuda_available

ROOT_DIR = Path(__file__).parent.parent
RANDOM_SEED = 33
DATA_DIR = ROOT_DIR / Path("data/ag_news")
MODEL_DIR = ROOT_DIR / Path("models")
RESULTS_DIR = ROOT_DIR / Path("results")
DEVICE = "cuda" if torch_cuda_available() else "cpu"
DEBUG = True
CONSOLE = Console()
LOGGER = Logger("nlp_pipeline")
RETRAIN_MODEL = False
