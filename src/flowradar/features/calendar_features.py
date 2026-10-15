import pandas as pd
from flowradar.calendar_uk import (
    holiday_set,
    is_working_day,
    month_end_payday,
    tax_dates,
    vat_quarter_dates,
)

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    # adds calendar-derived columns: day of week, month, is_weekend, is_holiday, is_working_day
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

def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    # known-in-advance event days that drive large, predictable cash flows: month-end payday
    # (rolled back to the prior working day), self assessment tax dates and quarterly VAT dates.
    # These depend only on the date, so they are valid inputs at any forecast horizon
    out = df.copy()
    dates = pd.to_datetime(out["date"])
    years = range(int(dates.dt.year.min()), int(dates.dt.year.max()) + 1)
    hols = holiday_set(min(years), max(years))

    months = {(int(y), int(m)) for y, m in zip(dates.dt.year, dates.dt.month)}
    paydays = {month_end_payday(y, m, hols) for y, m in months}
    tax_days, vat_days = set(), set()
    for year in years:
        tax_days |= set(tax_dates(year))
        vat_days |= set(vat_quarter_dates(year))

    day = dates.dt.date
    out["is_payday"] = day.isin(paydays).astype(int)
    out["is_tax_date"] = day.isin(tax_days).astype(int)
    out["is_vat_date"] = day.isin(vat_days).astype(int)
    return out