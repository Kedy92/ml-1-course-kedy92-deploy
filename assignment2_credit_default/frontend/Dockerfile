FROM python:3.13-slim

WORKDIR /app

COPY frontend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend /app/frontend

ENV API_URL=https://credit-card-default-api.onrender.com

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
