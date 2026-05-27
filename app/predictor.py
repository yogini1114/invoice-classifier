import os
import re
import pickle
from typing import Tuple

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/classifier.pkl")

_pipeline = None


def _load_model():
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Model file not found. Please run `python model/train.py` first."
            )
        with open(MODEL_PATH, "rb") as f:
            _pipeline = pickle.load(f)
    return _pipeline


def _preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict(text: str) -> Tuple[str, float]:

    pipeline = _load_model()
    cleaned = _preprocess(text)
    category = pipeline.predict([cleaned])[0]
    proba = pipeline.predict_proba([cleaned])[0]
    confidence = round(float(max(proba)), 4)
    return category, confidence
