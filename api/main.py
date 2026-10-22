"""MessWise API.  Run:  uvicorn api.main:app --reload"""
import json
import math
import sys
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import train as T  # noqa: E402

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
REF_DATE = "2026-10-21"  # a Wednesday, used as the "neutral weekday" reference


def load_state():
    """Load saved artifacts; retrain automatically if missing or built with another scikit-learn version."""
    def _load():
        meta = json.loads((T.ART / "meta.json").read_text())
        if meta.get("sklearn") != sklearn.__version__:
            raise RuntimeError("scikit-learn version changed")
        return dict(
            models=joblib.load(T.ART / "models.joblib"), meta=meta,
            summary=json.loads((T.ART / "summary.json").read_text()),
            history=json.loads((T.ART / "history.json").read_text()))
    try:
        return _load()
    except Exception:
        print("Artifacts missing or stale, retraining...")
        T.run(verbose=False)
        return _load()


STATE = load_state()

app = FastAPI(title="MessWise API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictIn(BaseModel):
    date: str
    meal: str
    menu_item: Optional[str] = None
    is_holiday: Optional[bool] = None
    expected_attendance: Optional[int] = None
    weather: str = "normal"
    menu_popularity_score: Optional[float] = None
    is_exam_week: Optional[bool] = None
    planned_prep: Optional[int] = None


def _frame(date, meal, pop, holiday, exam, weather, expected):
    return pd.DataFrame([{"date": date, "meal": meal, "menu_popularity_score": pop, "is_holiday": int(holiday),
                          "is_exam_week": int(exam), "weather": weather, "expected_attendance": expected}])


def _mean(**kw):
    return float(STATE["models"]["mean"].predict(T.featurize(_frame(**kw)))[0])


def _factors(d, meal, item, pop, hol, exam, weather, exp, typical):
    """Counterfactual explanation: set one input to neutral and see how the forecast moves."""
    base = dict(date=d, meal=meal, pop=pop, holiday=hol, exam=exam, weather=weather, expected=exp)
    p0 = _mean(**base)

    def delta(**change):
        return p0 - _mean(**{**base, **change})

    def more_less(v, up="more", down="fewer"):
        return f"about {abs(round(v))} {up if v > 0 else down} servings"

    items = []
    if hol:
        v = delta(holiday=0)
        items.append(("Holiday", v, f"Holiday: many students go home, {more_less(v)}"))
    if exam:
        v = delta(exam=0)
        items.append(("Exam week", v, f"Exam week: students skip or shift meals, {more_less(v)}"))
    if weather != "normal":
        v = delta(weather="normal")
        items.append(("Weather", v, f"{weather.capitalize()} weather: {more_less(v)}"))
    v = delta(pop=3)
    kind = "a crowd favourite" if pop >= 4 else ("less popular" if pop <= 2 else "an average dish")
    items.append(("Menu popularity", v, f"{item or 'This dish'} is {kind}: {more_less(v)}"))
    v = delta(expected=typical)
    if (exp - typical) * v > 0:  # only explain when direction is consistent (avoids confusing tree artefacts)
        side = "above" if exp > typical else "below"
        items.append(("Expected headcount", v, f"Expected headcount {exp} is {side} the usual {typical} for {meal}: {more_less(v)}"))
    items = sorted((i for i in items if abs(i[1]) >= 3), key=lambda i: -abs(i[1]))
    return [{"factor": f, "impact": int(round(v)), "text": t} for f, v, t in items[:3]]


@app.post("/predict")
def predict(body: PredictIn):
    meta = STATE["meta"]
    if body.meal not in meta["meals"]:
        raise HTTPException(422, f"meal must be one of {meta['meals']}")
    if body.weather not in T.WEATHER_CODE:
        raise HTTPException(422, f"weather must be one of {list(T.WEATHER_CODE)}")
    try:
        dt = pd.Timestamp(body.date)
    except Exception:
        raise HTTPException(422, "date must look like 2026-10-25")
    dow = dt.weekday()
    item = body.menu_item or meta["default_menu"][body.meal][dow]
    pop = body.menu_popularity_score if body.menu_popularity_score is not None else meta["popularity"].get(item, 3)
    hol = body.is_holiday if body.is_holiday is not None else body.date in meta["holidays"]
    exam = body.is_exam_week if body.is_exam_week is not None else any(a <= body.date <= b for a, b in meta["exam_windows"])
    typical = meta["expected_defaults"][body.meal][str(dow)]
    exp = body.expected_attendance if body.expected_attendance is not None else typical

    X = T.featurize(_frame(body.date, body.meal, pop, hol, exam, body.weather, exp))
    p, lo, hi = (float(v[0]) for v in T.predict_interval(STATE["models"], X))
    rec = math.ceil(p + meta["rec_buffer"] * (hi - p))
    planned = body.planned_prep if body.planned_prep is not None else int(round(exp * meta["prep_buffer"]))
    risk = "over" if planned > hi else ("under" if planned < lo else "balanced")
    mean_typical = int(round(sum(meta["expected_defaults"][body.meal].values()) / 7))
    return {
        "predicted": round(p), "low": round(lo), "high": round(hi), "recommended_prep": rec,
        "planned_prep": planned, "risk": risk,
        "top_factors": _factors(body.date, body.meal, item, pop, hol, exam, body.weather, exp, mean_typical),
        "inputs_used": {"menu_item": item, "menu_popularity_score": pop, "is_holiday": bool(hol),
                        "is_exam_week": bool(exam), "expected_attendance": exp, "weather": body.weather},
    }


@app.get("/history")
def history(days: int = Query(30, ge=1, le=90)):
    series = STATE["history"]
    cutoff = (pd.Timestamp(series[-1]["date"]) - pd.Timedelta(days=days - 1)).strftime("%Y-%m-%d")
    rows = [r for r in series if r["date"] >= cutoff]
    return {"days": len({r["date"] for r in rows}), "series": rows}


@app.get("/summary")
def summary():
    return STATE["summary"]


@app.get("/meta")
def meta():
    return STATE["meta"]


@app.get("/health")
def health():
    return {"status": "ok"}
