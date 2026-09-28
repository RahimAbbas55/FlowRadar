import numpy as np
import pandas as pd
from flowradar.calendar_uk import tax_dates

def generate_freelancer(
    start_date,
    n_days: int,
    seed: int,
    noise_scale: float = 0.08,
    invoices_per_month: float = 2.5,
    invoice_size_mean_log: float = 7.2,  # ~ exp(7.2) ≈ 1340 median invoice
    invoice_size_sigma_log: float = 0.6,
    base_daily_spend: float = 45.0,
    tax_payment_fraction: float = 0.22,  # rough share of income set aside
    annual_income_estimate: float = 32000.0,
) -> pd.DataFrame:
    # seeded rng for reproducibility independent of other segments
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    years = range(dates[0].year, dates[-1].year + 1)

    # merge tax dates across the year range covered
    tax_days = {}
    for y in years:
        tax_days.update(tax_dates(y))

    # poisson arrival process: expected invoices per day from a monthly rate
    daily_rate = invoices_per_month / 30.4
    arrivals = rng.poisson(daily_rate, size=n_days)

    inflow = np.zeros(n_days)
    outflow = np.zeros(n_days)

    for i, d in enumerate(dates):
        d_date = d.date()

        # inflow: sum of lognormal invoice payments landing this day
        if arrivals[i] > 0:
            invoices = rng.lognormal(invoice_size_mean_log, invoice_size_sigma_log, arrivals[i])
            inflow[i] = invoices.sum()

        # outflow: low steady daily spend
        outflow[i] += base_daily_spend * (1 + rng.normal(0, noise_scale))

        # outflow: self assessment payments on account, twice a year
        if d_date in tax_days:
            outflow[i] += (annual_income_estimate * tax_payment_fraction / 2) * (
                1 + rng.normal(0, noise_scale)
            )

    return pd.DataFrame(
        {
            "date": dates,
            "segment": "freelancer",
            "inflow": np.round(inflow, 2),
            "outflow": np.round(np.maximum(outflow, 0), 2),
        }
    )