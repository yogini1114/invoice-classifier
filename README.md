# Invoice Expense Classifier

A small FastAPI service that classifies invoice text into expense categories. Built with TF-IDF vectorisation and Logistic Regression — nothing fancy, but it gets the job done reliably for this kind of text.

## Categories supported

- Logistics
- Office Supplies
- Cloud/Software
- Utilities
- Travel
- Inventory

---

## Project layout

```
invoice-classifier/
├── app/
│   ├── main.py          # FastAPI app and routes
│   └── predictor.py     # Model loading + inference logic
├── data/
│   └── training_data.csv
├── model/
│   ├── train.py         # Training script
│   └── classifier.pkl   # Saved model (generated after training)
├── tests/
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Quick start (local)

**1. Clone the repo and set up a virtual environment**

```bash
git clone https://github.com/your-username/invoice-classifier.git
cd invoice-classifier

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**2. Train the model**

This generates `model/classifier.pkl`. You need to do this once before running the API.

```bash
python model/train.py
```

You should see something like:

```
Training on 56 samples, testing on 14 samples.

Test Accuracy: 0.9286

Classification Report:
              precision    recall  f1-score   support
...
Model saved to model/classifier.pkl
```

**3. Start the server**

```bash
uvicorn app.main:app --reload
```

API will be live at `http://localhost:8000`.

---

## API usage

### POST `/predict`

**Request**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Blue Dart courier charges for warehouse delivery"}'
```

**Response**

```json
{
  "category": "Logistics",
  "confidence": 0.9413
}
```

More examples:

```bash
# Cloud/Software
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "AWS monthly cloud hosting bill"}'

# Office Supplies
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Printer paper and stapler refills"}'

# Travel
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hotel stay and cab charges for client meeting"}'
```

### GET `/health`

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

Interactive docs are available at `http://localhost:8000/docs` (Swagger UI).

---

## Running tests

```bash
pytest tests/ -v
```

Make sure you've trained the model first or the tests will fail with a 503.

---

## Docker

**Build and run with Docker Compose (recommended)**

```bash
docker-compose up --build
```

This builds the image, trains the model inside the container, and starts the server on port 8000. No extra steps needed.

**Or with plain Docker**

```bash
docker build -t invoice-classifier .
docker run -p 8000:8000 invoice-classifier
```

---

## How it works

The pipeline is straightforward:

1. **Preprocessing** — lowercase, strip punctuation, normalise whitespace.
2. **TF-IDF** — unigrams + bigrams, top 5000 features, sublinear TF scaling.
3. **Logistic Regression** — multinomial, LBFGS solver. Works well on short text like invoice descriptions.

The confidence score comes from the softmax probability of the predicted class. Anything above ~0.75 is generally reliable; lower scores may mean the text is ambiguous or doesn't match any category well.

---

## Adding more training data

Just append rows to `data/training_data.csv` (two columns: `text`, `category`) and re-run `python model/train.py`. The model will retrain from scratch.

---

## Notes

- The model is intentionally simple — TF-IDF + LR is interpretable, fast, and works well for this kind of structured short-text classification.
- If you want to experiment with a Naive Bayes variant, swap out `LogisticRegression` for `MultinomialNB` in `model/train.py`.
- Training data is small (70 samples) so accuracy on completely unseen domains will vary. More data always helps.
