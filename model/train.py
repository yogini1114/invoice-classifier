import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re


DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/training_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "classifier.pkl")


def basic_clean(text: str) -> str:
    
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(path: str):
    df = pd.read_csv(path)
    df.dropna(subset=["text", "category"], inplace=True)
    df["text"] = df["text"].apply(basic_clean)
    return df["text"].tolist(), df["category"].tolist()


def build_pipeline() -> Pipeline:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
    )
    clf = LogisticRegression(
        max_iter=300,
        C=1.5,
        solver="lbfgs",
    )
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def train():
    print("Loading data...")
    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
