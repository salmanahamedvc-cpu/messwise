"""Train baseline + gradient-boosting forecaster, evaluate on a time-based split, export artifacts.

Run:  python model/train.py
"""
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import generate_data as gd  # noqa: E402

DATA = ROOT / "data" / "messwise_data.csv"
ART = Path(__file__).parent / "artifacts"
FRONTEND_FALLBACK = ROOT / "frontend" / "src" / "data" / "fallback.json"

MEAL_CODE = {"breakfast": 0, "lunch": 1, "dinner": 2}
WEATHER_CODE = {"normal": 0, "hot": 1, "rain": 2}
FEATURES = ["day_of_week", "is_weekend", "meal_code", "menu_popularity_score",
            "is_holiday", "is_exam_week", "weather_code", "expected_attendance"]
FEATURE_LABELS = {
    "expected_attendance": "Expected headcount", "menu_popularity_score": "Menu popularity",
    "is_holiday": "Holiday", "meal_code": "Meal type", "day_of_week": "Day of week",
    "is_weekend": "Weekend", "is_exam_week": "Exam week", "weather_code": "Weather"}
TEST_DAYS = 35
COST_PER_SERVING = 60
REC_BUFFER = 0.6      # recommended prep = predicted + 60% of the way to the upper bound of the 80% interval
BASELINE_BUFFER = 1.10


def featurize(df):
    d = pd.to_datetime(df["date"])
    return pd.DataFrame({
        "day_of_week": d.dt.weekday.values,
        "is_weekend": (d.dt.weekday >= 5).astype(int).values,
        "meal_code": df["meal"].map(MEAL_CODE).values,
        "menu_popularity_score": df["menu_popularity_score"].astype(float).values,
        "is_holiday": df["is_holiday"].astype(int).values,
        "is_exam_week": df["is_exam_week"].astype(int).values,
        "weather_code": df["weather"].map(WEATHER_CODE).values,
        "expected_attendance": df["expected_attendance"].astype(float).values,
    })[FEATURES]


def _gbr(loss, alpha=None):
    kw = dict(n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.8, min_samples_leaf=5, random_state=7)
    return GradientBoostingRegressor(loss=loss, **({"alpha": alpha} if alpha else {}), **kw)


def fit_models(X, y):
    return {"mean": _gbr("squared_error").fit(X, y), "q10": _gbr("quantile", 0.10).fit(X, y),
            "q90": _gbr("quantile", 0.90).fit(X, y)}


def predict_interval(models, X):
    """Quantile GBM interval, widened by a conformal offset q (CQR) so the 80% range is really ~80%."""
    q = models.get("q", 0.0)
    p = models["mean"].predict(X)
    lo = np.minimum(models["q10"].predict(X), p) - q
    hi = np.maximum(models["q90"].predict(X), p) + q
    return p, np.maximum(lo, 0), hi


def calibrate(train):
    """Fit on the older part of train, measure interval misses on the latest 28 days, return conformal offset."""
    cal_start = train["date"].max() - pd.Timedelta(days=27)
    a, c = train[train["date"] < cal_start], train[train["date"] >= cal_start]
    mdl = fit_models(featurize(a), a["actual_servings"])
    _, lo, hi = predict_interval(mdl, featurize(c))
    y = c["actual_servings"].values
    scores = np.maximum(lo - y, y - hi)
    n = len(scores)
    return max(float(np.quantile(scores, min(1.0, 0.8 * (1 + 1 / n)), method="higher")), 0.0)


def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
def mape(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b)) / np.asarray(a)) * 100)


def run(verbose=True):
    ART.mkdir(exist_ok=True)
    if not DATA.exists():
        gd.generate()
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    df["mcode"] = df["meal"].map(MEAL_CODE)
    df = df.sort_values(["date", "mcode"]).reset_index(drop=True)

    # Baseline: average of the same weekday + meal over the previous 4 weeks
    df["baseline"] = df.groupby(["day_of_week", "meal"])["actual_servings"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean())

    cutoff = df["date"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train, test = df[df["date"] < cutoff].copy(), df[df["date"] >= cutoff].copy()
    models = fit_models(featurize(train), train["actual_servings"])
    models["q"] = calibrate(train)
    p, lo, hi = predict_interval(models, featurize(test))
    test["pred"], test["low"], test["high"] = p, lo, hi
    y = test["actual_servings"].values

    m = {
        "baseline_mae": mae(y, test["baseline"]), "baseline_mape": mape(y, test["baseline"]),
        "model_mae": mae(y, p), "model_mape": mape(y, p),
        "conformal_offset": models["q"], "interval_coverage_pct": float(np.mean((y >= lo) & (y <= hi)) * 100),
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "train_period": [str(train["date"].min().date()), str(train["date"].max().date())],
        "test_period": [str(test["date"].min().date()), str(test["date"].max().date())],
    }
    m["mae_improvement_pct"] = (1 - m["model_mae"] / m["baseline_mae"]) * 100
    m["mape_improvement_pct"] = (1 - m["model_mape"] / m["baseline_mape"]) * 100

    # Decision policy on the held-out period
    test["rec"] = np.ceil(test["pred"] + REC_BUFFER * (test["high"] - test["pred"])).astype(int)
    test["base_prep"] = np.ceil(test["baseline"] * BASELINE_BUFFER).astype(int)
    waste = lambda prep: int(np.maximum(prep - test["actual_servings"], 0).sum())
    short = lambda prep: int((test["actual_servings"] > prep).sum())
    w_cur, w_msw, w_base = int(test["food_wasted"].sum()), waste(test["rec"]), waste(test["base_prep"])
    n_days = test["date"].nunique()
    s = {
        "period": m["test_period"], "days": int(n_days), "meals": int(len(test)),
        "servings_served": int(test["actual_servings"].sum()), "cost_per_serving_rs": COST_PER_SERVING,
        "waste_current": w_cur, "waste_messwise": w_msw, "waste_baseline_method": w_base,
        "waste_avoided_servings": w_cur - w_msw, "cost_saved_rs": (w_cur - w_msw) * COST_PER_SERVING,
        "waste_reduction_pct": (w_cur - w_msw) / w_cur * 100,
        "waste_reduction_vs_baseline_pct": (w_base - w_msw) / w_base * 100,
        "monthly_cost_saved_rs": int(round((w_cur - w_msw) * COST_PER_SERVING * 30 / n_days)),
        "shortage_meals_current": short(test["food_prepared"]), "shortage_meals_messwise": short(test["rec"]),
        "shortage_meals_baseline_method": short(test["base_prep"]),
    }

    hist = [dict(date=r.date.strftime("%Y-%m-%d"), meal=r.meal, menu_item=r.menu_item, actual=int(r.actual_servings),
                 predicted=round(float(r.pred), 1), low=round(float(r.low), 1), high=round(float(r.high), 1),
                 baseline=round(float(r.baseline), 1), prepared=int(r.food_prepared), recommended=int(r.rec))
            for r in test.itertuples()]
    test["gap"] = test["food_prepared"] - test["actual_servings"]
    pick = pd.concat([test.nlargest(5, "gap"), test[test["gap"] < 0].nsmallest(3, "gap")])
    s["worst_meals"] = [dict(date=r.date.strftime("%Y-%m-%d"), meal=r.meal, menu_item=r.menu_item, actual=int(r.actual_servings),
                             prepared=int(r.food_prepared), gap=int(r.gap), recommended=int(r.rec),
                             flag="over" if r.gap > 0 else "short") for r in pick.itertuples()]

    # Serve with models refit on all data
    final = fit_models(featurize(df), df["actual_servings"])
    final["q"] = models["q"]
    imp = dict(zip(FEATURES, final["mean"].feature_importances_))
    importance = sorted([{"feature": k, "label": FEATURE_LABELS[k], "importance": round(float(v) * 100, 1)}
                         for k, v in imp.items()], key=lambda r: -r["importance"])

    menus = {ml: sorted({*gd.TIMETABLE[ml], *gd.SPECIALS[ml]}) for ml in gd.MEALS}
    exp_defaults = {ml: {str(d): int(round(df[(df.meal == ml) & (df.date.dt.weekday == d)]["expected_attendance"].mean()))
                         for d in range(7)} for ml in gd.MEALS}
    meta = {
        "sklearn": sklearn.__version__, "students": gd.N_STUDENTS, "cost_per_serving_rs": COST_PER_SERVING,
        "meals": gd.MEALS, "menus": menus, "popularity": gd.POP, "default_menu": gd.TIMETABLE,
        "expected_defaults": exp_defaults, "prep_buffer": gd.PREP_BUFFER, "rec_buffer": REC_BUFFER,
        "holidays": gd.HOLIDAYS, "exam_windows": gd.EXAM_WINDOWS,
        "data_range": [str(df["date"].min().date()), str(df["date"].max().date())], "rows": int(len(df)),
        "metrics": m, "feature_importance": importance,
    }
    joblib.dump(final, ART / "models.joblib")
    for name, obj in [("meta", meta), ("summary", s), ("history", hist)]:
        (ART / f"{name}.json").write_text(json.dumps(obj, indent=1))
    if FRONTEND_FALLBACK.parent.exists():
        FRONTEND_FALLBACK.write_text(json.dumps({"meta": meta, "summary": s, "history": {"days": n_days, "series": hist}}))
    if verbose:
        print("\n=== Holdout (last %d days, time-based split) ===" % TEST_DAYS)
        print(f"{'':16}{'MAE':>8}{'MAPE %':>9}")
        print(f"{'Baseline (4wk)':16}{m['baseline_mae']:8.1f}{m['baseline_mape']:9.2f}")
        print(f"{'MessWise GBM':16}{m['model_mae']:8.1f}{m['model_mape']:9.2f}")
        print(f"Improvement: MAE {m['mae_improvement_pct']:.0f}%, MAPE {m['mape_improvement_pct']:.0f}% | 80% interval coverage {m['interval_coverage_pct']:.0f}%")
        print("\nWaste servings  current:", w_cur, " baseline-method:", w_base, " MessWise:", w_msw)
        print("Shortage meals  current:", s["shortage_meals_current"], " baseline-method:", s["shortage_meals_baseline_method"], " MessWise:", s["shortage_meals_messwise"])
        print(f"Waste reduction {s['waste_reduction_pct']:.0f}% | cost saved Rs {s['cost_saved_rs']:,} in {n_days} days (~Rs {s['monthly_cost_saved_rs']:,}/month)")
        print("\nFeature importance:", [(r['label'], r['importance']) for r in importance[:5]])
    return meta, s, hist


if __name__ == "__main__":
    run()
