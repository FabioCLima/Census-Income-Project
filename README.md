# Census Income Classification — MLOps Project

GitHub repository: [FabioCLima/Census-Income-Project](https://github.com/FabioCLima/Census-Income-Project)

## Overview

End-to-end MLOps project that trains a Random Forest classifier on the UCI Adult
Census Income dataset and deploys it as a REST API on Heroku with full CI/CD.

## Project Structure

- `src/census/` — data loading, preprocessing, training, evaluation, inference, slicing
- `main.py` — FastAPI application (GET welcome + POST inference)
- `train_model.py` — training entrypoint
- `tests/` — 185 unit and API tests
- `model/` — trained pipeline, encoder, CV results, slice output
- `model_card.md` — model documentation
- `live_api_request.py` — script to POST against the live Heroku API
- `.github/workflows/ci.yml` — CI (lint + test) and CD (deploy to Heroku on main)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install uv
uv sync --all-groups
```

## Train the model

```bash
python train_model.py
```

## Run the API locally

```bash
uvicorn main:app --reload
```

## Run tests

```bash
pytest
```

## Live API

Base URL: [census-income-api-7cfe90f1b0a4.herokuapp.com](https://census-income-api-7cfe90f1b0a4.herokuapp.com)

- `GET /` — welcome message
- `POST /predict` — income prediction (`>50K` or `<=50K`)

```bash
python live_api_request.py
```
