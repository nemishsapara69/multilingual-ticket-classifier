# End-to-End Multilingual Customer Support Ticket Classifier

This project is your Smart Router for support tickets.

When a customer writes a message like "Mi luz inteligente no funciona", the system predicts the right ticket category (for example: `Technical Support`) with a confidence score.

## Release Snapshot (v1.0.0)

- Problem: Route multilingual customer tickets to the right support queue automatically.
- Languages: English, Spanish, German.
- Classes: `Technical Support`, `Billing Inquiry`, `Order Tracking`.
- Serving stack: FastAPI API + React demo UI.
- Reproducibility: DVC pipeline, Dockerfiles, and CI workflows.

## Measured Results

- Curated evaluation set (`data/eval/eval_queries.csv`):
  - Accuracy: `100%` (`reports/eval_report.json`)
- Unseen real-world style set (`data/eval/eval_queries_realworld.csv`):
  - Accuracy: `100%` (`reports/eval_report_realworld.json`)

## Architecture (At a Glance)

1. Data ingestion and cleaning with `src/data_preprocessing.py`
2. Model training with `src/train.py` (transformer path + robust baseline fallback)
4. Inference routing with `src/inference.py`
5. API serving with `src/api.py`
6. Interactive React demo with `frontend/src/App.jsx`
6. Automated regression checks with `src/evaluate.py`

## What You Built (Simple View)

1. Data pipeline with versioning (DVC)
2. Multilingual NLP model training (XLM-RoBERTa)
3. Experiment tracking (MLflow)
4. Inference API (FastAPI)
5. Demo website (React)
6. Containerized services (Docker)
7. CI/CD automation (GitHub Actions)

## Project Structure

- `src/data_preprocessing.py`: cleans raw tickets and creates train/val/test splits.
- `src/train.py`: fine-tunes multilingual transformer model and logs metrics to MLflow.
- `src/inference.py`: loads model and predicts category + confidence.
- `src/api.py`: FastAPI service with `/predict` endpoint.
- `frontend/src/App.jsx`: React demo that calls API and displays results.
- `dvc.yaml`: pipeline stages for preprocess + train.
- `params.yaml`: training hyperparameters.
- `docker/*.Dockerfile`: reproducible containers for train/api/webapp.
- `.github/workflows/ci.yaml`: tests + docker build checks.
- `.github/workflows/retrain.yaml`: scheduled/manual retraining workflow.

## Step-by-Step: How Maria's Message Is Processed

### Phase 1: Data Acquisition and Versioning

1. Put your raw spreadsheet export in `data/raw/tickets.csv`.
2. Track dataset versions with DVC.

```bash
dvc init
dvc add data/raw/tickets.csv
git add data/raw/tickets.csv.dvc .gitignore .dvc/
git commit -m "Track raw ticket dataset with DVC"
```

What this means: your dataset has version history (v1.0, v1.1, etc.) without bloating Git.

### Phase 2: Data Cleaning and Preprocessing

Run preprocessing:

```bash
python -m src.data_preprocessing --input data/raw/tickets.csv --output_dir data/processed
```

What happens:

1. Finds text and category columns
2. Cleans text (removes emojis, extra spaces, control chars)
3. Keeps multilingual text intact
4. Splits into train/validation/test
5. Saves metadata for reproducibility

### Phase 3: Model Training and Experiment Tracking

Train model:

```bash
python -m src.train --data_dir data/processed --model_dir models/best --params params.yaml --metrics_path models/metrics.json
```

What happens:

1. Loads `xlm-roberta-base`
2. Learns from your labeled tickets
3. Evaluates on validation and test sets
4. Logs hyperparameters + metrics to MLflow
5. Saves best model in `models/best`

Optional push to Hugging Face Hub:

```bash
set PUSH_TO_HUB=1
set HF_REPO_ID=your-username/your-model-repo
set HF_TOKEN=your_hf_token
python -m src.train --data_dir data/processed --model_dir models/best --params params.yaml --metrics_path models/metrics.json
```

### Phase 4: Serve Model with FastAPI

Start API:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Test API:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"text\":\"Mi luz inteligente no funciona\"}"
```

Expected output style:

```json
{
  "category": "Technical Support",
  "confidence": 0.98,
  "cleaned_text": "Mi luz inteligente no funciona"
}
```

### Phase 5: Interactive Web Demo with React

Run web app:

```bash
cd frontend
npm install
npm run dev
```

What happens:

1. User types message into text box
2. Web app calls FastAPI `/predict`
3. API calls model
4. App shows predicted category + confidence

### Phase 6: Reproducibility with Docker

Build images:

```bash
docker build -f docker/api.Dockerfile -t mtc-api .
docker build -f docker/webapp.Dockerfile -t mtc-webapp .
docker build -f docker/train.Dockerfile -t mtc-train .
```

Run API + Web app together:

```bash
docker compose up --build
```

### Phase 7: CI/CD Automation

- `ci.yaml`: runs tests and builds Docker images on push/PR.
- `retrain.yaml`: manual/scheduled retraining job.

## Quick Start (One Path)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.data_preprocessing --input data/raw/tickets.csv --output_dir data/processed
python -m src.train --data_dir data/processed --model_dir models/best --params params.yaml --metrics_path models/metrics.json
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

## One-Click Runner (Windows)

From project root:

```bash
scripts\run_all.bat
```

Useful options:

```bash
scripts\run_all.bat -SkipInstall
scripts\run_all.bat -SkipInstall -SkipTrain
scripts\run_all.bat -RunEval
scripts\run_all.bat -ApiOnly
```

What it does:

1. Installs dependencies (Python + frontend) unless `-SkipInstall`
2. Runs preprocessing + training unless `-SkipTrain`
3. Starts FastAPI server
4. Starts React UI (`frontend`) unless `-ApiOnly`
5. Optionally runs both eval suites with `-RunEval`

## Deployment (Step-by-Step)

This section is the easiest path to put your project online.

### Step 1: Push Latest Code to GitHub

Make sure your latest code is on the `main` branch.

### Step 2: Deploy Backend API on Render

1. Go to Render and create a **Blueprint** service.
2. Connect your GitHub repo.
3. Render will auto-detect `render.yaml` and create the API service.
4. Wait for deployment to finish.
5. Open your backend URL and test:

```bash
https://<your-render-service>.onrender.com/health
```

Expected output should include:

```json
{ "status": "ok", "model_loaded": true }
```

Notes:

- Ticket persistence is enabled with SQLite at `/var/data/tickets.db` using Render persistent disk.
- Main env variables are documented in `.env.example`.

### Step 3: Deploy Frontend on Vercel

1. Go to Vercel and import the same GitHub repo.
2. Set **Root Directory** to `frontend`.
3. Add environment variable:

```bash
VITE_API_URL=https://<your-render-service>.onrender.com/predict
```

4. Deploy.

`frontend/vercel.json` already includes SPA rewrite config.

### Step 4: End-to-End Live Test

1. Open your Vercel frontend URL.
2. Submit a test message, for example:
  - `Mi luz inteligente no funciona`
  - `I was charged twice this month`
  - `Wo ist mein Paket?`
3. Confirm:
  - prediction appears,
  - queue count updates,
  - recent ticket row appears,
  - clicking row shows that ticket timeline.

### Step 4.1: Login and Roles (v2)

The app now supports role-based login.

Default demo users:

- `admin` (full access)
- `agent` (can classify + view dashboard)
- `viewer` (read-only dashboard)

Default passwords are set via environment variables in `render.yaml`:

- `AUTH_ADMIN_PASSWORD`
- `AUTH_AGENT_PASSWORD`
- `AUTH_VIEWER_PASSWORD`

For safety, change these values in Render environment after first deploy.

### Step 5: Production Checks

1. Backend health endpoint responds consistently.
2. Frontend can classify after page refresh.
3. After backend restart/redeploy, dashboard still retains previous tickets.


## Notes

- First training can be slow because model weights download from Hugging Face.
- For production, store DVC data remote and secure your HF token in GitHub Secrets.
