"""Run:  pytest -q   (from the repo root; needs: pip install pytest httpx)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from api.main import app

c = TestClient(app)


def test_health():
    assert c.get("/health").json() == {"status": "ok"}


def test_meta():
    m = c.get("/meta").json()
    assert set(m["meals"]) == {"breakfast", "lunch", "dinner"}
    assert len(m["default_menu"]["lunch"]) == 7
    assert m["metrics"]["model_mae"] < m["metrics"]["baseline_mae"]
    assert 70 <= m["metrics"]["interval_coverage_pct"] <= 95


def test_predict_defaults():
    r = c.post("/predict", json={"date": "2026-10-25", "meal": "lunch"}).json()
    assert r["low"] <= r["predicted"] <= r["recommended_prep"] <= r["high"] + 1
    assert r["risk"] in ("over", "under", "balanced")
    assert r["inputs_used"]["menu_item"] == "Chicken Biryani"  # Sunday lunch


def test_predict_holiday_lowers_demand():
    base = c.post("/predict", json={"date": "2026-10-20", "meal": "lunch", "expected_attendance": 465, "is_holiday": False}).json()
    hol = c.post("/predict", json={"date": "2026-10-20", "meal": "lunch", "expected_attendance": 465, "is_holiday": True}).json()
    assert hol["predicted"] < base["predicted"] - 100
    assert hol["top_factors"][0]["factor"] == "Holiday"


def test_predict_validation():
    assert c.post("/predict", json={"date": "2026-10-25", "meal": "brunch"}).status_code == 422
    assert c.post("/predict", json={"date": "not-a-date", "meal": "lunch"}).status_code == 422


def test_history():
    h = c.get("/history?days=30").json()
    assert h["days"] == 30 and len(h["series"]) == 90
    assert {"actual", "predicted", "low", "high", "baseline", "prepared", "recommended"} <= set(h["series"][0])


def test_summary():
    s = c.get("/summary").json()
    assert s["waste_messwise"] < s["waste_current"]
    assert s["shortage_meals_messwise"] <= s["shortage_meals_current"]
    assert s["worst_meals"] and s["worst_meals"][0]["flag"] in ("over", "short")
