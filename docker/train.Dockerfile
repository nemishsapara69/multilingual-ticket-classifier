FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "src.train", "--data_dir", "data/processed", "--model_dir", "models/best", "--params", "params.yaml", "--metrics_path", "models/metrics.json"]
