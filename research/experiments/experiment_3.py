import polars as pl
import datetime as dt
from research.signals import get_signal, construct_signals
from research.filters import get_filter, apply_filters
from research.alpha_constructors import get_alpha_constructor, construct_alphas
from research.portfolios import construct_quantile_portfolios
from research.returns import construct_returns_from_weights
import great_tables as gt
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

sample = "out-of-sample"
start = end = None
rebalance_frequency = "monthly"
annualize_results = True

signal_names = [
    "momentum",
    "idiosyncratic_momentum_fama_french_3",
    "volatility_scaled_idiosyncratic_momentum_fama_french_3",
]

weights_name_map = {
    "momentum": "momentum",
    "idiosyncratic_momentum_fama_french_3": "idio_mom_ff3",
    "volatility_scaled_idiosyncratic_momentum_fama_french_3": "idio_mom_vol_scaled_ff3",
}

filter_names = ["low-price-stocks"]

alpha_constructor_name = "cross_sectional_z_score"

match sample:
    case "in-sample":
        start = dt.date(1963, 7, 31)
        end = dt.date(2015, 12, 31)
    case "out-of-sample":
        start = dt.date(2016, 1, 1)
        end = dt.date(2024, 12, 31)
    case _:
        raise ValueError(f"Sample not supported: {sample}")

returns_list = []
for signal_name in signal_names:
    signal_weights_name = weights_name_map[signal_name]
    weights = (
        pl.scan_parquet(
            f"weights/{signal_weights_name}/{signal_weights_name}_*.parquet"
        )
        .filter(pl.col("date").is_between(start, end))
        # .with_columns(
        #     pl.col('weight').truediv(pl.col('weight').sum()).over('date') # unit leverage
        # )
        .collect()
    )

    returns = construct_returns_from_weights(
        weights=weights, rebalance_frequency=rebalance_frequency
    ).select("date", pl.lit(signal_name).alias("signal"), pl.col("return"))
    returns_list.append(returns)


print("Combining results...")
all_returns = pl.concat(returns_list)

annual_factor = 1
match rebalance_frequency:
    case "monthly":
        annual_factor = 12
    case _:
        raise ValueError(f"Rebalance frequency not implemented: {rebalance_frequency}")

print("Saving summary table...")
summary_table = (
    all_returns.group_by("signal")
    .agg(
        pl.col("return").mean().mul(annual_factor * 100).alias("mean_return"),
        pl.col("return")
        .std()
        .mul(pl.lit(annual_factor).sqrt() * 100)
        .alias("volatility"),
    )
    .with_columns(pl.col("mean_return").truediv("volatility").alias("sharpe"))
    .with_columns(pl.exclude("signal").round(2))
    .with_columns(pl.col("signal").str.replace_all("_", " ").str.to_titlecase())
    .rename(
        {
            "signal": "Signal",
            "mean_return": "Mean Return",
            "volatility": "Volatility",
            "sharpe": "Sharpe",
        }
    )
)

title = "Momentum Variations"
file_path = "results/experiment_3/combined"
Path(file_path).parent.mkdir(parents=True, exist_ok=True)

table = (
    gt.GT(summary_table)
    .tab_header(title=title)
    .opt_stylize(style=5, color="gray")
    .tab_source_note(source_note=f"Period: {start} to {end}")
    .tab_source_note(source_note=f"Rebalance Frequency: {rebalance_frequency.title()}")
    .tab_source_note(source_note=f"Annualized: {'Yes' if annualize_results else 'No'}")
    .tab_options(source_notes_padding=gt.px(10))  # Add padding
)

table.save(Path(file_path + "_table").with_suffix(".png"))

print("Saving returns chart...")
cumulative_returns = (
    all_returns.sort("signal", "date")
    .with_columns(pl.col("signal").str.replace_all("_", " ").str.to_titlecase())
    .with_columns(pl.col("return").log1p().cum_sum().mul(100).over("signal"))
    .rename({"date": "Date", "return": "Return", "signal": "Signal"})
)
print(cumulative_returns)
plt.figure(figsize=(10, 6))

sns.lineplot(cumulative_returns, x="Date", y="Return", hue="Signal")

plt.title("Momentum Variations")
plt.xlabel(None)
plt.ylabel("Cumulative Log Returns (%)")

plt.savefig(Path(file_path + "_chart").with_suffix(".png"))
