from sklearn.base import BaseEstimator
from src.data.data import AGNews

from src.const import MODEL_DIR
import pickle

# This file contains functions to train and load simple baseline models on the AG News dataset.

def train_model(model: BaseEstimator, ds: AGNews, save: bool = True):
    """Train a model on the AG News dataset."""
    model.fit(ds.X_train, ds.y_train)
    if save:
        MODEL_DIR.mkdir(exist_ok=True)
        model_path = MODEL_DIR / f"{model.__class__.__name__}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    return model

def get_model(model: BaseEstimator, ds: AGNews) -> BaseEstimator:
    """Load a trained model from disk."""
    model_path = MODEL_DIR / f"{model.__class__.__name__}.pkl"
    if not model_path.exists():
        model = train_model(model, ds)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model
            