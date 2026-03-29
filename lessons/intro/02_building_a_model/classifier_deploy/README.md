# CIFAR10 Classifier Deployment

This deployment package exposes your image classifier as a FastAPI API and includes an optional Streamlit frontend.

For the assignment, the simplest production path is:

1. export `cifar10_classifier.pkl`
2. place it in `models/`
3. deploy the API to Render

## What Changed

- PostgreSQL was removed because it was only used for prediction logging
- the API now runs as a single container
- Dockerfiles now honor Render's `PORT` environment variable
- the API image copies `models/` into the container at build time

## Required File

The API expects this file:

- `models/cifar10_classifier.pkl`

If it is missing, startup will fail.

## Local Docker Run

From this directory:

```bash
docker-compose up --build
```

Services:

- API: `http://localhost:8000`
- health check: `http://localhost:8000/health`
- Streamlit frontend: `http://localhost:8501`

Test the API:

```bash
curl -X POST -F "file=@cat.jpg" http://localhost:8000/predict
```

## Render Deployment

The repo root contains a `render.yaml` Blueprint for the API service.

### Before deploying

1. Make sure `lessons/intro/02_building_a_model/classifier_deploy/models/cifar10_classifier.pkl` exists
2. Commit that file, or if it is too large for GitHub, replace it with a smaller exported model and keep the filename the same

### Deploy steps

1. Push the repo to GitHub
2. In Render, create a new Blueprint service from the repo
3. Render will detect `render.yaml`
4. Deploy the `cifar10-classifier-api` service
5. After deploy, open:

```text
https://<your-service>.onrender.com/health
```

You should get:

```json
{"status":"ok","model_loaded":true}
```

Test prediction with any HTTP client:

```bash
curl -X POST -F "file=@cat.jpg" https://<your-service>.onrender.com/predict
```

## Optional Streamlit Frontend

You can also deploy the frontend as a second Render web service, but it is not required for the assignment.

If you want it:

1. create another web service from the same repo
2. set the root directory to `lessons/intro/02_building_a_model/classifier_deploy`
3. use `docker/Dockerfile.frontend`
4. set env var:

```text
API_URL=https://<your-api-service>.onrender.com/predict
```

## Files

- `app/main.py`: FastAPI app and model loading
- `app/predict.py`: image upload and inference endpoint
- `docker/Dockerfile.api`: API container
- `docker/Dockerfile.frontend`: Streamlit container
- `docker-compose.yml`: local two-service setup
