import pandas as pd
from flowradar.calendar_uk import holiday_set, is_working_day

# Adds calendar-derived columns: day of week, month, is_weekend, is_holiday, is_working_day
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out["date"])
    years = range(dates.dt.year.min(), dates.dt.year.max() + 1)
    hols = holiday_set(min(years), max(years))

    out["day_of_week"] = dates.dt.weekday
    out["day_of_month"] = dates.dt.day
    out["month"] = dates.dt.month
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_holiday"] = dates.dt.date.isin(hols.keys()).astype(int)
    out["is_working_day"] = [
        int(is_working_day(d, hols)) for d in dates.dt.date
    ]
    return out