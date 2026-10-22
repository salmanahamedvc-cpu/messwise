# MessWise: AI-Based Hostel Food Demand and Waste Predictor

Forecasts how many students will eat each meal, with an 80% uncertainty range, and recommends how much to cook so the mess wastes less food without running short.

> **Honest note:** the dataset is **synthetic** (600 students, 183 days, 3 meals/day; patterns are assumptions listed in `data/data_dictionary.md`). All savings figures are **estimates on held-out synthetic data**, not measurements from a real mess. The pipeline is built to accept real mess logs in the same format.

## Results (last 35 days held out, time-based split)
| | Baseline (4-week same-day average) | MessWise (gradient boosting) |
|---|---|---|
| MAE (servings) | 33.2 | 15.6 (53% better) |
| MAPE | 9.5% | 4.2% |

- 80% interval covers about 84% of real outcomes (conformal-calibrated quantile models).
- Waste under current practice (cook headcount + 12%): 6,406 servings. With MessWise recommended prep: 2,878 (about 55% less).
- Shortage meals: 14 today, 12 with MessWise (no extra shortages).
- At Rs 60 per serving that is roughly Rs 2.1 lakh over 35 days (about Rs 1.8 lakh per month) for a 600-student hostel, in this simulation.

## How it works
1. `data/generate_data.py` builds the dataset. `data/data_dictionary.md` documents every field.
2. `model/train.py` fits the baseline, a gradient boosting model, and 10%/90% quantile models, widened by a conformal offset so the range is honest. It evaluates on a time-based split and saves artifacts to `model/artifacts/`.
3. `api/main.py` serves the model. If saved artifacts are missing or were built with a different scikit-learn version, it retrains automatically at startup.
4. Recommended prep = predicted + 60% of the way to the upper bound of the 80% range.

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python data/generate_data.py
python model/train.py
uvicorn api.main:app --reload   # http://localhost:8000/docs
```
Tests: `pip install pytest httpx` then `pytest -q`.

## API
| Endpoint | Purpose |
|---|---|
| `POST /predict` | `{date, meal, menu_item?, is_holiday?, expected_attendance?, weather?, menu_popularity_score?, is_exam_week?, planned_prep?}` returns `predicted, low, high, recommended_prep, planned_prep, risk, top_factors` |
| `GET /history?days=30` | Held-out actual vs predicted series |
| `GET /summary` | Waste avoided, cost saved, shortage counts, worst meals |
| `GET /meta` | Menus, defaults, holidays, metrics, feature importance |
| `GET /health` | `{"status":"ok"}` |

Try: `curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d "{\"date\":\"2026-10-25\",\"meal\":\"lunch\"}"`

## Deploy the API on Render (free)
1. Push this repo to GitHub.
2. render.com: **New** > **Web Service** > connect the repo.
3. Runtime **Python 3**. Build command: `pip install -r requirements.txt`. Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`.
4. Add env var `PYTHON_VERSION` = `3.11.9`. Instance type: **Free**. Click **Create**.
5. Open `https://YOUR-SERVICE.onrender.com/health`. It should return `{"status":"ok"}`.

**Cold start:** the free tier sleeps after about 15 minutes idle, and the first request can take 30-60 seconds (plus a few seconds to retrain if artifacts were stale). Open `/health` a few minutes before any demo.

## Tools and AI used
Claude (Anthropic) for code and planning help; Python, pandas, NumPy, scikit-learn, FastAPI, Uvicorn, joblib. Add every other tool your team used.
