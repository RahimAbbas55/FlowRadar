import numpy as np
import pandas as pd
from flowradar.calendar_uk import holiday_set, is_working_day, vat_quarter_dates

def generate_small_business(
    start_date,
    n_days: int,
    seed: int,
    noise_scale: float = 0.08,
    base_daily_sales: float = 1400.0,
    weekly_supplier_payment: float = 900.0,
    monthly_payroll: float = 4200.0,
    quarterly_vat: float = 3600.0,
) -> pd.DataFrame:
    # seeded rng for reproducibility independent of other segments
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    years = range(dates[0].year, dates[-1].year + 1)
    hols = holiday_set(min(years), max(years))
    vat_days = set()
    for y in years:
        vat_days.update(vat_quarter_dates(y))

    inflow = np.zeros(n_days)
    outflow = np.zeros(n_days)

    for i, d in enumerate(dates):
        d_date = d.date()

        # inflow: weekday-driven sales, near zero on bank holidays and weekends
        if is_working_day(d_date, hols):
            inflow[i] = base_daily_sales * (1 + rng.normal(0, noise_scale))
        else:
            inflow[i] = base_daily_sales * 0.05 * (1 + rng.normal(0, noise_scale))

        # outflow: weekly supplier payment on Mondays
        if d.weekday() == 0:
            outflow[i] += weekly_supplier_payment * (1 + rng.normal(0, noise_scale))

        # outflow: month-end payroll on the 28th
        if d.day == 28:
            outflow[i] += monthly_payroll * (1 + rng.normal(0, noise_scale))

        # outflow: quarterly VAT settlement
        if d_date in vat_days:
            outflow[i] += quarterly_vat * (1 + rng.normal(0, noise_scale))

    return pd.DataFrame(
        {
            "date": dates,
            "segment": "small_business",
            "inflow": np.round(np.maximum(inflow, 0), 2),
            "outflow": np.round(np.maximum(outflow, 0), 2),
        }
    )