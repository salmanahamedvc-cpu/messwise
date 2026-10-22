"""Generate 6 months of synthetic hostel-mess data (600 students, 3 meals/day).

Run:  python data/generate_data.py
All patterns below are ASSUMPTIONS baked into synthetic data; see data_dictionary.md.
"""
from pathlib import Path
import numpy as np
import pandas as pd

N_STUDENTS = 600
START, END = "2026-04-22", "2026-10-21"
SEED = 42
OUT = Path(__file__).parent / "messwise_data.csv"

POP = {  # menu item -> popularity score (1-5)
    "Idli Sambar": 4, "Pongal": 3, "Poori Masala": 4, "Upma": 2, "Chapati Kurma": 3, "Masala Dosa": 5,
    "Veg Meals": 3, "Lemon Rice": 3, "Sambar Rice": 2, "Veg Fried Rice": 4, "Curd Rice Meals": 3, "Chicken Biryani": 5,
    "Chapati Dal": 3, "Idiyappam Kurma": 3, "Parotta Salna": 5, "Egg Curry Rice": 4, "Veg Pulao": 3,
}
TIMETABLE = {  # Monday..Sunday
    "breakfast": ["Idli Sambar", "Pongal", "Poori Masala", "Upma", "Idli Sambar", "Chapati Kurma", "Masala Dosa"],
    "lunch": ["Veg Meals", "Lemon Rice", "Sambar Rice", "Veg Fried Rice", "Curd Rice Meals", "Veg Meals", "Chicken Biryani"],
    "dinner": ["Chapati Dal", "Idiyappam Kurma", "Parotta Salna", "Egg Curry Rice", "Veg Fried Rice", "Chapati Dal", "Veg Pulao"],
}
SPECIALS = {"breakfast": ["Masala Dosa"], "lunch": ["Chicken Biryani", "Veg Fried Rice"], "dinner": ["Parotta Salna", "Egg Curry Rice"]}
MEALS = ["breakfast", "lunch", "dinner"]
BASE_RATE = {"breakfast": 0.60, "lunch": 0.80, "dinner": 0.76}          # share of students who eat
DOW_FACTOR = {                                                            # Mon..Sun
    "breakfast": [1, 1, 1, 1, 0.97, 0.88, 0.80],
    "lunch": [1, 1, 1, 1, 0.97, 0.88, 0.93],
    "dinner": [1, 1, 1, 0.98, 0.90, 0.90, 0.68],                          # Sunday dinner drop
}
HOLIDAY = {"breakfast": 0.55, "lunch": 0.60, "dinner": 0.65}
EXAM = {"breakfast": 0.85, "lunch": 0.93, "dinner": 0.95}
RAIN = {"breakfast": 0.88, "lunch": 0.97, "dinner": 0.98}
HOT = {"breakfast": 1.0, "lunch": 0.95, "dinner": 1.0}
HOLIDAYS = ["2026-05-27", "2026-06-26", "2026-08-15", "2026-08-26", "2026-09-04", "2026-09-14", "2026-10-02"]
EXAM_WINDOWS = [("2026-06-08", "2026-06-19"), ("2026-09-28", "2026-10-09")]
PREP_BUFFER = 1.12  # current practice: cook expected headcount + 12%


def is_exam(d):
    return any(pd.Timestamp(a) <= d <= pd.Timestamp(b) for a, b in EXAM_WINDOWS)


def generate(seed=SEED, save=True):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range(START, END):
        dow = d.weekday()
        holiday = d.strftime("%Y-%m-%d") in HOLIDAYS
        exam = is_exam(d)
        p_rain = 0.30 if d.month >= 6 else 0.12
        p_hot = 0.30 if d.month <= 5 else 0.10
        u = rng.random()
        weather = "rain" if u < p_rain else ("hot" if u < p_rain + p_hot else "normal")
        for meal in MEALS:
            item = TIMETABLE[meal][dow]
            if rng.random() < 0.12:                                       # occasional special-menu swap
                item = str(rng.choice(SPECIALS[meal]))
            pop = POP[item]
            pop_mult = 1 + 0.07 * (pop - 3) + (0.04 if pop == 5 else 0)  # biryani / special spike
            demand = N_STUDENTS * BASE_RATE[meal] * DOW_FACTOR[meal][dow] * pop_mult
            if holiday: demand *= HOLIDAY[meal]
            if exam: demand *= EXAM[meal]
            if weather == "rain": demand *= RAIN[meal]
            if weather == "hot": demand *= HOT[meal]
            actual = int(min(N_STUDENTS, round(demand * (1 + rng.normal(0, 0.035)))))
            # staff headcount estimate: knows the weekday pattern, not holidays/menu/weather/exams
            expected = int(round(N_STUDENTS * BASE_RATE[meal] * DOW_FACTOR[meal][dow] * (1 + rng.normal(0, 0.05))))
            prepared = int(round(expected * PREP_BUFFER + rng.normal(0, 6)))
            rows.append(dict(
                date=d.strftime("%Y-%m-%d"), day_of_week=d.day_name(), meal=meal, menu_item=item,
                menu_popularity_score=pop, is_holiday=int(holiday), is_exam_week=int(exam), is_weekend=int(dow >= 5),
                weather=weather, expected_attendance=expected, actual_servings=actual, food_prepared=prepared,
                food_wasted=max(prepared - actual, 0), shortage_flag=int(actual > prepared)))
    df = pd.DataFrame(rows)
    if save:
        df.to_csv(OUT, index=False)
        print(f"Wrote {len(df)} rows -> {OUT}")
    return df


if __name__ == "__main__":
    generate()
