import polars as pl
import datetime as dt
from research.signals import get_signal, construct_signals
from research.filters import get_filter, apply_filters
from research.alpha_constructors import get_alpha_constructor, construct_alphas
from pathlib import Path

def get_alphas(signal_name: str):
    start = dt.date(2000, 1, 1)
    end = dt.date(2024, 12, 31)

    filter_names = ['low-price-stocks']
    filters = [get_filter(filter_name) for filter_name in filter_names]

    alpha_constructor_name = "cross-sectional-z-score"
    alpha_constructor = get_alpha_constructor(alpha_constructor_name, signal_name=signal_name)


    signal = get_signal(signal_name, id_col='barrid')

    print("Loading data...")
    barra = pl.scan_parquet("data/barra/barra_*parquet")
    ff3 = pl.scan_parquet("data/fama_french/ff3_factors.parquet")
    barra_ff3_betas = pl.scan_parquet("data/barra_ff3_betas/barra_ff3_betas_*.parquet")

    data = (
        barra
        .join(
            other=ff3,
            on=['date'],
            how='left'
        )
        .join(
            other=barra_ff3_betas,
            on=['date', 'barrid'],
            how='left'
        )
        .filter(pl.col('date').is_between(start, end))
        .collect()
    )

    print("Constructing signals...")
    signals = construct_signals(data=data, signal=signal)

    print("Applying filters...")
    filtered = apply_filters(signals=signals, filters=filters)

    print("Constructing alphas...")
    alphas = construct_alphas(filtered, alpha_constructor=alpha_constructor).select('date', 'barrid', 'alpha')

    print("Saving alphas...")    
    output_path = Path(f"alphas/{signal_name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    alphas.write_parquet(output_path.with_suffix(".parquet"))

if __name__ == '__main__':
    signal_names = ["momentum", "idiosyncratic_momentum_fama_french_3", "volatility_scaled_idiosyncratic_momentum_fama_french_3"]
    
    for signal_name in signal_names:
        get_alphas(signal_name=signal_name)