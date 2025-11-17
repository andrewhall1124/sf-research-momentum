import polars as pl
import datetime as dt
from research.signals import get_signal, construct_signals
from research.filters import get_filter, apply_filters
from research.portfolios import construct_quantile_portfolios
from research.returns import construct_returns
from research.evaluations import (
    create_quantile_summary_table,
    create_quantile_returns_chart,
)
from pathlib import Path

sample = "in-sample"
start = end = None
n_bins = 10
weighting_scheme = "equal"
rebalance_frequency = "monthly"
annualize_results = False

signal_names = ["momentum", "volatility_scaled_idiosyncratic_momentum_fama_french_3"]

filter_names = [
    "penny-stocks",
    "micro-caps",
    "null-signal",
    "null-idiosyncratic-momentum",
]

match sample:
    case "in-sample":
        start = dt.date(1963, 7, 31)
        end = dt.date(2015, 12, 31)
    case _:
        raise ValueError(f"Sample not supported: {sample}")

crsp = pl.scan_parquet("data/crsp/crsp_*.parquet")
ff3 = pl.scan_parquet("data/fama_french/ff3_factors.parquet")
crsp_ff3_betas = pl.scan_parquet("data/crsp_ff3_betas/crsp_ff3_betas_*.parquet")

data = (
    crsp.join(other=ff3, on="date", how="left")
    .join(other=crsp_ff3_betas, on=["date", "permno"], how="left")
    .filter(pl.col("date").is_between(start, end))
    .sort("permno", "date")
    .collect()
)

for signal_name in signal_names:
    print(f"Running experiment for {signal_name}...")
    signal = get_signal(signal_name, id_col="permno")

    print("Constructing signals...")
    signals = construct_signals(data=data, signal=signal)

    print("Applying filters...")
    filters = [
        get_filter(filter_name, signal_name=signal_name) for filter_name in filter_names
    ]
    filtered = apply_filters(signals=signals, filters=filters)

    print("Constructing portfolios...")
    portfolios = construct_quantile_portfolios(
        data=filtered, n_bins=n_bins, signal=signal, weighting_scheme=weighting_scheme
    )

    print("Constructing returns...")
    returns = construct_returns(
        data=portfolios, n_bins=n_bins, rebalance_frequency=rebalance_frequency
    )

    title = signal.name.replace("_", " ").title()
    file_path = "results/experiment_4/" + signal_name + "-" + sample
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    print("Saving results...")
    create_quantile_summary_table(
        returns=returns,
        file_path=file_path,
        annualize_results=annualize_results,
        rebalance_frequency=rebalance_frequency,
        n_bins=n_bins,
        start=start,
        end=end,
        title=title,
    )
