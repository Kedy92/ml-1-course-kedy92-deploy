# Assignment 2: Credit Card Default

This project trains and compares two models on the UCI Default of Credit Card Clients dataset:

- a PyTorch MLP with embeddings for categorical features
- a CatBoost classifier, with a random forest baseline for context

The notebook required by the assignment is [`tabular.ipynb`](../tabular.ipynb).

## Dataset

- Name: Default of Credit Card Clients
- Original source: UCI Machine Learning Repository
- Local CSV used here: `data/credit_card_default/default_of_credit_card_clients.csv`

## Project Layout

- `train_models.py`: runs the full training pipeline and saves artifacts
- `src/`: shared data prep, model, metrics, and tree utilities
- `api/main.py`: FastAPI app with both prediction endpoints
- `models/`: saved MLP and CatBoost artifacts
- `artifacts/`: metrics tables and JSON summaries
- `notebook_assets/`: saved plots used by the notebook and Discord post

## Local Training

Use the existing project virtualenv:

```bash
.venv-assignment1/bin/python assignment2_credit_default/train_models.py
```

## API

Run locally:

```bash
.venv-assignment1/bin/uvicorn api.main:app --app-dir assignment2_credit_default --reload
```

Endpoints:

- `GET /health`
- `POST /predict/mlp`
- `POST /predict/tree`

Example payload:

```json
{
  "limit_bal": 120000,
  "sex": "female",
  "education": "university",
  "marriage": "single",
  "age": 26,
  "pay_0": -1,
  "pay_2": 2,
  "pay_3": 0,
  "pay_4": 0,
  "pay_5": 0,
  "pay_6": 2,
  "bill_amt1": 2682,
  "bill_amt2": 1725,
  "bill_amt3": 2682,
  "bill_amt4": 3272,
  "bill_amt5": 3455,
  "bill_amt6": 3261,
  "pay_amt1": 0,
  "pay_amt2": 1000,
  "pay_amt3": 1000,
  "pay_amt4": 1000,
  "pay_amt5": 0,
  "pay_amt6": 2000
}
```

The same payload is saved in `sample_request.json`.

## Streamlit Frontend

There is also a small Streamlit UI in `frontend/app.py` that can call the deployed API.

Run locally:

```bash
API_URL=https://credit-card-default-api.onrender.com \
.venv-assignment1/bin/streamlit run assignment2_credit_default/frontend/app.py
```

## Docker Compose

From `assignment2_credit_default/`:

```bash
docker-compose up --build
```

## Railway Deployment

Deploy the API and UI as two separate services from the same repo.

API service:

- root directory: `assignment2_credit_default`
- dockerfile: `docker/Dockerfile.api`

UI service:

- root directory: `assignment2_credit_default`
- dockerfile: `frontend/Dockerfile`
- set env var `API_URL` to your Railway API URL

Railway injects `PORT` automatically, and both Dockerfiles now respect it.
