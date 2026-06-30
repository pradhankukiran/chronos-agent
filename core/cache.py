import os
import pickle
import time

# Resolve models directory path relative to this file (chronos-agent/models)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# 24 hours in seconds
DEFAULT_EXPIRY = 24 * 60 * 60

def get_model_path(ticker: str) -> str:
    """Returns the absolute file path for a ticker's cached model."""
    return os.path.join(MODELS_DIR, f"{ticker.upper()}_hybrid.pkl")

def is_model_cached(ticker: str, expiry_seconds: int = DEFAULT_EXPIRY) -> bool:
    """
    Checks if a valid, unexpired model is cached for the given ticker.
    """
    model_path = get_model_path(ticker)
    if not os.path.exists(model_path):
        return False
        
    # Check if the file is older than the expiry limit
    file_age = time.time() - os.path.getmtime(model_path)
    return file_age < expiry_seconds

def save_model(ticker: str, model_data: dict) -> str:
    """
    Saves the trained models dict (containing both Prophet & XGBoost models) to disk.
    """
    model_path = get_model_path(ticker)
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    return model_path

def load_model(ticker: str) -> dict:
    """
    Loads the cached models dict for a ticker.
    """
    model_path = get_model_path(ticker)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No cached model found for '{ticker}' at {model_path}")
        
    with open(model_path, "rb") as f:
        return pickle.load(f)
