import polars as pl
import datetime as dt
from research.signals import get_signal, construct_signals
from research.filters import get_filter, apply_filters
from research.alpha_constructors import get_alpha_constructor, construct_alphas
from pathlib import Path


def alphas_flow(start: dt.date, end: dt.date):
    signal_names = [
        "momentum",
        "idiosyncratic_momentum_fama_french_3",
        "volatility_scaled_idiosyncratic_momentum_fama_french_3",
    ]
    
    filter_names = ["low-price-stocks"]
    filters = [get_filter(filter_name) for filter_name in filter_names]

    print("Loading data...")
    barra = pl.scan_parquet("data/barra/barra_*parquet")
    ff3 = pl.scan_parquet("data/fama_french_factors/fama_french_factors.parquet")
    barra_ff3_betas = pl.scan_parquet("data/barra_ff3_betas/barra_ff3_betas_*.parquet")

    data = (
        barra.join(other=ff3, on=["date"], how="left")
        .join(other=barra_ff3_betas, on=["date", "barrid"], how="left")
        .filter(pl.col("date").is_between(start, end))
        .collect()
    )

    for signal_name in signal_names:
        signal = get_signal(signal_name, id_col="barrid")

        alpha_constructor_name = "cross-sectional-z-score"
        alpha_constructor = get_alpha_constructor(
            alpha_constructor_name, signal_name=signal_name
        )

        print("Constructing signals...")
        signals = construct_signals(data=data, signal=signal)

        print("Applying filters...")
        filtered = apply_filters(signals=signals, filters=filters)

        print("Constructing alphas...")
        alphas = construct_alphas(filtered, alpha_constructor=alpha_constructor).select(
            "date", "barrid", "alpha"
        )

        print("Saving alphas...")
        output_path = Path(f"data/alphas/{signal_name}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        alphas.write_parquet(output_path.with_suffix(".parquet"))

