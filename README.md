# Project to learn MLOPs (The URL is currently suspended so the page cannot be found at the moment)

# 0_0 Note-to-self: Own Guide

## Initial Setup

1. initiate python -m venv venv >> add activate.bat and template.py
2. Update .gitignore
3. pip install pip-tools >> update requirements.in && requirements-dev.in
4. pip-compile requirements-dev.in >> pip-sync requirements-dev.txt
5. update pyproject.toml to package code, and Dockerfile.
6. pip install -e . >> run pyproject.toml >> package code
7. if build Docker -> pip-compile requirements.in before building image

## Workflows:

1. update config.yaml
2. update secrets.yaml [Optional]
3. update params.yaml
4. update the entity
5. update the configuration manager in src config
6. update the components
7. update the pipeline
8. update main.py
9. update the dvc.yaml


# 🐔 Chicken Disease Classification

A learning project following the MLOps path — classifying chicken fecal images as **Healthy** or **Coccidiosis** using PyTorch + a proper MLOps pipeline.

> Built while following Krish Naik's MLOps course, but swapped TensorFlow → PyTorch, Flask → FastAPI, and AWS → GCP. Because why not make it harder.

**Live API:** https://chicken-disease-app-601694768853.asia-southeast1.run.app/docs

---

## Stack

| Layer | Tool |
|---|---|
| Model | VGG16 (PyTorch) |
| Pipeline | DVC |
| Experiment Tracking | MLflow + DagsHub |
| API | FastAPI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud | GCP Cloud Run + Artifact Registry |

---

## Architecture

```
params.yaml
    ↓
dvc repro
    ├── data_ingestion        # download & unzip dataset
    ├── prepare_base_model    # load VGG16, modify classifier head
    ├── training              # train + log to MLflow/DagsHub
    └── evaluation            # val metrics → scores.json + MLflow
    
GitHub push
    ↓
GitHub Actions
    ├── CI: lint + test (placeholder)
    ├── CD: docker build → push to Artifact Registry
    └── Deploy: Cloud Run
```

---

## DVC Pipeline

Reproduce the full training pipeline:

```bash
dvc repro
```

Run a specific stage:

```bash
dvc repro training
```

Tweak hyperparams in `params.yaml`, re-run, DVC will skip unchanged stages automatically.

```yaml
EPOCHS: 10
BATCH_SIZE: 32
LEARNING_RATE: 0.001
IMAGE_SIZE: [224, 224, 3]
```

---

## MLflow + DagsHub

Experiments are tracked on DagsHub. Each training + evaluation run logs:
- Params: epochs, batch size, learning rate, image size
- Metrics: train loss, val loss, accuracy (per epoch)
- Model artifact: registered as `ChickenDiseaseClassifier`

To run with your own DagsHub:

```bash
# .env
MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<username>
MLFLOW_TRACKING_PASSWORD=<token>
```

---

## Docker + GCP Deploy

Build and run locally:

```bash
# compile prod deps first
pip-compile requirements.in

docker build -t chicken-disease .
docker run -p 8000:8000 chicken-disease
```

Deployment is automated via GitHub Actions on every push to `main`:

```
push to main
  → build Docker image
  → push to GCP Artifact Registry
  → deploy to Cloud Run
```

Required GitHub Secrets:
- `GCP_CREDENTIALS` — service account JSON
- `GCP_PROJECT_ID` — GCP project ID

---

## What I actually learned

- DVC stages + caching beat running scripts manually every time
- MLflow tracking belongs in the pipeline file, not the component
- VGG16 needs 2Gi RAM on Cloud Run. Found out the hard way.
- `.dockerignore` matters a lot when you have 1GB of training data sitting around
- FastAPI's `/docs` is genuinely great for testing ML APIs
