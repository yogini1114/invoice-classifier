"""
test_api.py - Unit tests for the Invoice Classifier API
Run with: pytest tests/
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ── Basic API tests ──────────────────────────────────────────────────────────

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_logistics():
    response = client.post("/predict", json={"text": "Blue Dart courier charges for warehouse delivery"})
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Logistics"
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_cloud_software():
    response = client.post("/predict", json={"text": "AWS monthly cloud hosting bill"})
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Cloud/Software"


def test_predict_office_supplies():
    response = client.post("/predict", json={"text": "Printer paper and stapler refills"})
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Office Supplies"


def test_predict_utilities():
    response = client.post("/predict", json={"text": "Monthly electricity bill for office"})
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Utilities"


def test_predict_travel():
    response = client.post("/predict", json={"text": "Hotel stay and cab expenses for client visit"})
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Travel"


def test_predict_inventory():
    response = client.post("/predict", json={"text": "Raw material steel rods purchase"})
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Inventory"


# ── Validation tests ─────────────────────────────────────────────────────────

def test_empty_text_returns_422():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_missing_text_field_returns_422():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_whitespace_only_text_returns_422():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_confidence_is_float():
    response = client.post("/predict", json={"text": "FedEx freight invoice for delivery"})
    assert response.status_code == 200
    assert isinstance(response.json()["confidence"], float)


def test_category_is_string():
    response = client.post("/predict", json={"text": "Google Workspace subscription"})
    assert response.status_code == 200
    assert isinstance(response.json()["category"], str)
