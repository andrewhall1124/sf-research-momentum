import sf_quant.data as sfd
import datetime as dt
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from signals import (
    cross_sectional_momentum,
    time_series_momentum,
    fifty_two_week_high_momentum,
    frog_in_the_pan_momentum,
    volatillity_scaled_momentum,
    idiosyncratic_momentum,
)


# Utility functions
def generate_decile_chart(
    df_cumulative_returns: pl.DataFrame, name: str, n_bins: int
) -> None:
    plt.figure(figsize=(10, 6))

    colors = sns.color_palette(palette="coolwarm", n_colors=n_bins)
    labels = [str(i) for i in range(n_bins)]

    for label, color in zip(labels, colors):
        sns.lineplot(df_cumulative_returns, x="date", y=label, color=color)

    sns.lineplot(df_cumulative_returns, x="date", y="spread", color="green")
    # sns.lineplot(df_cumulative_returns, x='date', y='null', color='gray')

    plt.title(name.replace("_", " ").title())
    plt.xlabel(None)
    plt.ylabel("Cumulative Log Return (%)")

    os.makedirs(exist_ok=True, name="results")
    plt.savefig(f"results/{name}.png", dpi=300)


# Parameters
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
n_bins = 10
rebalance = "monthly"  # daily or monthly
weighting = "equal"  # equal or market_cap

signals = [
    cross_sectional_momentum,
    time_series_momentum,
    fifty_two_week_high_momentum,
    frog_in_the_pan_momentum,
    volatillity_scaled_momentum,
    idiosyncratic_momentum,
]

# Configure parameters
match rebalance:
    case "daily":
        annual_factor = 252
        fwd_return = "fwd_return"
    case "monthly":
        annual_factor = 12
        fwd_return = "fwd_return_22d"

# Pull data
columns = [
    "date",
    "barrid",
    "ticker",
    "price",
    "return",
    "specific_return",
    "market_cap",
]
assets = (
    sfd.load_assets(start=start, end=end, columns=columns, in_universe=True)
    .sort("barrid", "date")
    .with_columns(pl.col("return", "specific_return").truediv(100))
    .with_columns(
        pl.col("return").shift(-1).over("barrid").alias("fwd_return"),
        pl.col("return")
        .log1p()
        .rolling_sum(window_size=22)
        .exp()
        .sub(1)
        .shift(-22)
        .alias("fwd_return_22d"),
    )
)

# Compute signals
df_signals = (
    assets.sort("barrid", "date")
    .with_columns([signal() for signal in signals])
    .filter(
        pl.col("price").gt(5),  # Price filter
    )
)

# Generate signal backtests
labels = [str(i) for i in range(n_bins)]
long, short = labels[-1], labels[0]
summary_stats = []
for signal in signals:
    df_portfolios = df_signals.with_columns(
        pl.col(signal.__name__)
        .rank(method="random", seed=42)
        .qcut(n_bins, labels=labels)
        .cast(pl.String)
        .over("date")
        .alias("portfolio")
    )

    if weighting == "equal":
        df_returns = df_portfolios.group_by("date", "portfolio").agg(
            pl.col(fwd_return).mul(pl.lit(1).truediv(pl.len())).sum().alias("return")
        )

    elif weighting == "market_cap":
        df_returns = df_portfolios.group_by("date", "portfolio").agg(
            pl.col(fwd_return)
            .mul(pl.col("market_cap").truediv(pl.col("market_cap").sum()))
            .sum()
            .alias("return"),
        )

    df_returns = (
        df_returns.sort("date", "portfolio")
        .pivot(index="date", on="portfolio", values="return")
        .with_columns(pl.col(long).sub(pl.col(short)).alias("spread"))
    )

    year_month = pl.col("date").dt.strftime("%y-%m")

    if rebalance == "monthly":
        df_returns = df_returns.filter(
            pl.col("date").eq(pl.col("date").max().over(year_month))
        )

    mean_return = df_returns["spread"].mean() * 100 * annual_factor
    volatility = df_returns["spread"].std() * 100 * np.sqrt(annual_factor)
    sharpe = mean_return / volatility

    summary_stats.append(
        {
            "signal": signal.__name__,
            "mean_return": mean_return,
            "volatility": volatility,
            "sharpe": sharpe,
        }
    )

    df_cumulative_returns = df_returns.sort("date").with_columns(
        pl.exclude("date").log1p().cum_sum().mul(100)
    )

    # Save backtest charts
    generate_decile_chart(df_cumulative_returns, name=signal.__name__, n_bins=n_bins)

# Print summary table
summary_table = pl.from_dicts(summary_stats).with_columns(pl.exclude("signal").round(2))
print(summary_table)
