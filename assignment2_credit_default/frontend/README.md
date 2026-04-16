# Frontend

Small Streamlit UI for trying the deployed API with a form instead of raw Swagger docs.

## Run locally

```bash
API_URL=https://credit-card-default-api.onrender.com \
.venv-assignment1/bin/streamlit run assignment2_credit_default/frontend/app.py
```

## Railway

Create a separate Railway service for this frontend and point it at:

- root directory: `assignment2_credit_default`
- dockerfile: `frontend/Dockerfile`

Set `API_URL` to the Railway URL of your deployed API service.
