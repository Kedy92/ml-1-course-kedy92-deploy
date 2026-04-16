# Railway Deployment Notes

Use two Railway services from the same repo.

## API

- Root Directory: `assignment2_credit_default`
- Builder: Dockerfile
- Dockerfile Path: `docker/Dockerfile.api`

Expected routes:

- `/health`
- `/docs`
- `/predict/mlp`
- `/predict/tree`

## UI

- Root Directory: `assignment2_credit_default`
- Builder: Dockerfile
- Dockerfile Path: `frontend/Dockerfile`

Environment variables:

- `API_URL=https://<your-api-service>.up.railway.app`

## After deploy

1. Open the API URL and test `/health`
2. Open the UI URL and run one prediction
3. Update your Discord thread links if you switch away from Render
