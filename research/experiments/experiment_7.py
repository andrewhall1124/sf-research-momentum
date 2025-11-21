import polars as pl
import datetime as dt
from research.signals import get_signal, construct_signals
from research.filters import get_filter, apply_filters
from research.portfolios import construct_quantile_portfolios
from research.returns import construct_returns
from research.evaluations import (
    create_quantile_summary_table, create_quantile_returns_chart
)
from pathlib import Path

def experiment_7():
    start = dt.date(1963, 7, 31)
    end = dt.date(2015, 12, 31)    
    n_bins = 10
    weighting_scheme = "market_cap"
    rebalance_frequency = "monthly"
    annualize_results = True

    signal_names = ["momentum", "constant_volatility_scaled_momentum", "semi_volatility_scaled_momentum", "dynamic_volatility_scaled_momentum"]
    filter_names = [
        "low-price-stocks",
        "null-signal",
    ]

    print("Loading data...")
    crsp = pl.scan_parquet("data/crsp/crsp_*.parquet")
    dmom_coefficients = pl.scan_parquet("data/dmom_coefficients/dmom_coefficients.parquet")
    market_data = pl.scan_parquet("data/dmom_coefficients/market_data.parquet")

    data = (
        crsp
        .join(dmom_coefficients, on='date', how='left')
        .join(market_data, on='date', how='left')
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

        print("Saving results...")
        title = signal.name.replace("_", " ").title()
        file_path = "results/experiment_7/" + signal_name
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        create_quantile_returns_chart(
            returns=returns, n_bins=n_bins, title=title, file_path=file_path
        )
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

if __name__ == "__main__":
    experiment_7()