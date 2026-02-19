from pathlib import Path
from rich.console import Console

ROOT_DIR = Path(__file__).parent.parent
RANDOM_SEED = 33
DATA_DIR = ROOT_DIR / Path("data/ag_news")
MODEL_DIR = ROOT_DIR / Path("models")
RESULTS_DIR = ROOT_DIR / Path("results")
DEBUG = True
CONSOLE = Console()
RETRAIN_MODEL = False
