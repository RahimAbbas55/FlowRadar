import numpy as np
import pandas as pd
from flowradar.calendar_uk import holiday_set, month_end_payday

# seeded rng for reproducibility independent of other segments
def generate_salaried(
    start_date,
    n_days: int,
    seed: int,
    noise_scale: float = 0.08,
    base_salary: float = 2800.0,
    base_rent: float = 950.0,
    base_bills: float = 220.0,
    base_weekend_spend: float = 60.0,
    december_uplift: float = 1.4,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    years = range(dates[0].year, dates[-1].year + 1)
    hols = holiday_set(min(years), max(years))

    # precompute one payday per month across the date range
    paydays = set()
    for y in years:
        for m in range(1, 13):
            paydays.add(month_end_payday(y, m, hols))

    inflow = np.zeros(n_days)
    outflow = np.zeros(n_days)

    for i, d in enumerate(dates):
        d_date = d.date()

        # inflow: salary lands on payday with small noise, otherwise zero
        if d_date in paydays:
            inflow[i] = base_salary * (1 + rng.normal(0, noise_scale))

        # outflow: rent and bills clustered around the 1st of the month
        if d.day in (1, 2, 3):
            outflow[i] += (base_rent + base_bills) * (1 + rng.normal(0, noise_scale)) / 3

        # outflow: weekend-weighted discretionary spend, every day but higher on Sat/Sun
        weekend_mult = 1.8 if d.weekday() >= 5 else 1.0
        month_mult = december_uplift if d.month == 12 else 1.0
        outflow[i] += base_weekend_spend * weekend_mult * month_mult * (
            1 + rng.normal(0, noise_scale)
        )

    return pd.DataFrame(
        {
            "date": dates,
            "segment": "salaried",
            "inflow": np.round(inflow, 2),
            "outflow": np.round(outflow, 2),
        }
    )