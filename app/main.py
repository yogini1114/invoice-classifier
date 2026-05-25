"""
main.py - FastAPI app for Invoice Expense Classification
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from app.predictor import predict

app = FastAPI(
    title="Invoice Expense Classifier",
    description="Classifies invoice text into expense categories using TF-IDF + Logistic Regression.",
    version="1.0.0",
)


class InvoiceRequest(BaseModel):
    text: str

    @validator("text")
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("text field cannot be empty")
        return v.strip()


class PredictionResponse(BaseModel):
    category: str
    confidence: float


@app.get("/")
def root():
    return {"message": "Invoice Classifier API is running. POST to /predict"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_category(payload: InvoiceRequest):
    try:
        category, confidence = predict(payload.text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictionResponse(category=category, confidence=confidence)
