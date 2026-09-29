# FlowRadar

![Status](https://img.shields.io/badge/status-in%20progress-yellow)
![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?logo=lightgbm&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![Prefect](https://img.shields.io/badge/Prefect-070E10?logo=prefect&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-232F3E?logo=amazonaws&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Cash flow forecasting and anomaly detection with a full MLOps lifecycle. Predicts daily net cash flow per account segment 30 days ahead with calibrated prediction intervals, flags anomalous days, and manages the complete model lifecycle: tracking, registry, drift monitoring, and automated retraining.

## Status

**Phase 1 (synthetic data generator) complete.** Seeded, deterministic generation of base signal, drift injection, and anomaly injection with ground-truth labels for all three account segments. Forecasting models, anomaly detection evaluation, and the MLOps layer are not yet built — see the roadmap below.

## Why synthetic data

Real transaction data is private, and injecting controlled drift and anomalies is the only reliable way to prove the monitoring and detection actually work. No real-world forecasting accuracy is claimed anywhere in this repo — known simplifications are listed in [Known limitations](#known-limitations).

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1. Data | Synthetic generator, drift and anomaly injection, ground-truth labels | ✅ complete |
| 2. Forecasting | Baselines, LightGBM, statsforecast, walk-forward backtest, conformal intervals | ⬜ not started |
| 3. Anomaly detection | Residual detector, evaluation vs injected labels | ⬜ not started |
| 4. MLOps core | MLflow, champion/challenger, Prefect retraining, Evidently drift monitors | ⬜ not started |
| 5. Serving and UI | FastAPI, Streamlit dashboard, Docker Compose | ⬜ not started |
| 6. Deploy and ship | AWS deployment, CI/CD, architecture docs | ⬜ not started |
| 7. Explainer (gated) | LoRA SFT + DPO anomaly-explanation model | ⬜ not started, may split into a standalone project |

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Data | pandas, numpy, pydantic (config schema) |
| Forecasting | LightGBM, statsforecast (ETS/ARIMA), conformal prediction |
| Anomaly detection | Residual-based, evaluated against injected ground-truth labels |
| MLOps | MLflow (tracking + registry), Evidently (drift), Prefect (retraining orchestration) |
| Serving | FastAPI |
| Dashboard | Streamlit |
| Infra | Docker, Docker Compose, AWS (EC2, ECR, S3), GitHub Actions |
| Testing | pytest, dedicated leakage tests for time-series splits |
| Phase 7 (gated) | Hugging Face transformers, PEFT (LoRA), TRL (SFT + DPO) |

## What the generator produces

`generate_dataset(config)` returns a dict with six keys:

- **`series`** — the final daily inflow/outflow/net series per segment, with drift and anomalies applied
- **`counterfactual`** — the same series after drift but before anomalies, so forecast error can be separated from detector error
- **`anomaly_labels`** — one row per injected anomaly: segment, type, cause label, affected component, expected vs actual value
- **`drift_events`** — one row per injected drift event: segment, type, component, start date, magnitude
- **`calendar`** — the UK bank holiday table used to drive paydays and sales patterns
- **`manifest`** — seed, config hash, generator version, and event counts, for full reproducibility

## Progress Log

### Phase 1, Stage 1 — Generator design
Designed the synthetic data contract before writing any code: 3 segments (salaried, freelancer, small business), daily aggregate grain, 3 years of history, 6 anomaly types at ~1.5% of days with ground-truth cause labels, 3 drift types, England and Wales bank holiday calendar.

### Phase 1, Stage 2 — Scaffold, calendar, config
- Repo scaffolded with a `src/` layout, minimal unpinned `requirements.txt`, and `.gitignore` covering `.env` and Terraform state files from the start
- Built a UK bank holiday calendar computed from rules (Easter offsets, weekday rules, Christmas/Boxing Day substitute-day logic), not a hardcoded list, with known-answer tests against real historical dates
- Built a `pydantic`-based `GeneratorConfig` with cross-field validation and a stable config hash for dataset reproducibility

### Phase 1, Stage 3 — Base signal generator
Built seeded, deterministic per-segment generators for all three account types, then a combined `generate_base_series()` entry point driven directly by `GeneratorConfig`.
- **Salaried**: month-end payday inflow (rolled back to the prior working day), rent/bills near the 1st, weekend-weighted spend, December uplift
- **Freelancer**: Poisson-arrival, lognormal-sized invoice payments, low steady daily spend, twice-yearly self-assessment tax outflows
- **Small business**: weekday-driven sales near zero on bank holidays, weekly supplier payments, month-end payroll, quarterly VAT

**Debugging notes**
- A `pd.Timestamp` compared directly against a `datetime.date`-keyed holiday lookup never matches, even for the same calendar day — this silently made the small business generator treat Christmas as a normal working day. Fixed by converting to `.date()` before every calendar lookup.
- `net = inflow - outflow` computed from two already-rounded columns can differ from a freshly rounded comparison by a trailing floating-point fraction. Fixed by rounding `net` itself at computation time.

### Phase 1, Stage 4 — Drift injection
Built three independent drift injectors plus a scheduler that places events per segment, respecting a configurable clean baseline window before any drift begins.
- **Abrupt**: step-function multiplier on a component from a start date onward
- **Gradual**: linear ramp from 1.0 to a target multiplier over a configurable duration, then holds
- **Volatility**: added multiplicative noise from a start date onward, changing variance rather than mean
- Scheduler assigns each event a segment, component, magnitude, and start date placed only in the post-baseline window, then applies all events sequentially so later drift compounds on earlier drift

**Known simplification**: overlapping events on the same segment aren't explicitly prevented. With the default one event per type per segment, collision risk is low but non-zero over a ~2-year post-baseline window.

### Phase 1, Stage 5 — Anomaly injection and dataset assembly
Built all six anomaly types with ground-truth cause labels, a counterfactual snapshot, and the combined `generate_dataset()` entry point.
- **Point events**: `large_unplanned_payment`, `duplicate_or_suspicious_debit` (outflow spikes), `payroll_delay`, `client_payment_delay` (inflow zeroed and shifted to a later real inflow day)
- **Span events**: `business_closure` (inflow zeroed across a date range), `data_gap` (both components set to `NaN` across a range, distinct from zero-activity)
- Segment eligibility enforced at the scheduling layer (e.g. `payroll_delay` only for salaried, `client_payment_delay` only for freelancer)
- Counterfactual captured after drift but before anomalies, isolating anomaly effects specifically

**Debugging notes**
- Delay-based anomalies (`payroll_delay`, `client_payment_delay`) need a real inflow day to delay — a naively chosen random date often lands on a zero-inflow day, silently doing nothing. Fixed with a `_nearest_inflow_day` lookup that finds the closest actual inflow day on or after the target date before applying the delay.
- Actual injected anomaly count can come in slightly under the configured `target_rate`, since eligible-segment and eligible-day skips reduce the effective count. The manifest's `n_anomaly_events` is the source of truth for evaluation, not the configured rate.

## Known limitations
- All data is synthetic. No claim is made about real-world forecasting accuracy.
- VAT settlement dates are simplified to a fixed day-of-month rather than exact HMRC quarter-specific rules.
- Drift and anomaly events aren't checked for overlap on the same segment; low but non-zero collision risk with default settings.
- Injected anomaly count can fall slightly under the configured target rate due to eligibility and inflow-day constraints.

## Setup

pip install -r requirements.txt
pip install -e .

## Testing

pytest -q