FROM python:3.13-slim

WORKDIR /app

COPY frontend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend /app/frontend

ENV API_URL=https://credit-card-default-api.onrender.com
ENV PORT=8501

EXPOSE 8501

CMD ["sh", "-c", "streamlit run frontend/app.py --server.port=${PORT} --server.address=0.0.0.0"]
