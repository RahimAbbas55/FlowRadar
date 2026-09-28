import pandas as pd
from flowradar.config import GeneratorConfig
from flowradar.segments.freelancer import generate_freelancer
from flowradar.segments.salaried import generate_salaried
from flowradar.segments.small_business import generate_small_business

# maps segment name to its generator function
_GENERATORS = {
    "salaried": generate_salaried,
    "freelancer": generate_freelancer,
    "small_business": generate_small_business,
}

   # builds the combined pre-drift, pre-anomaly series for every configured segment
def generate_base_series(config: GeneratorConfig) -> pd.DataFrame:
    frames = []
    for i, segment in enumerate(config.segments):
        if segment not in _GENERATORS:
            raise ValueError(f"unknown segment: {segment}")
        gen_fn = _GENERATORS[segment]
        df = gen_fn(
            start_date=config.start_date,
            n_days=config.n_days,
            seed=config.seed + i,
            noise_scale=config.noise_scale,
        )
        frames.append(df)
    
    combined = pd.concat(frames, ignore_index=True)
    combined["net"] = (combined["inflow"] - combined["outflow"]).round(2)
    return combined.sort_values(["segment", "date"]).reset_index(drop=True)