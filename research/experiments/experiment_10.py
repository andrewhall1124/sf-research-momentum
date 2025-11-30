import polars as pl
import datetime as dt
from research.returns import construct_returns_from_weights
import great_tables as gt
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

def experiment_10():
    rebalance_frequency = "monthly"
    annualize_results = True
    gamma = 60.0
    start = dt.date(2016, 1, 1)
    end = dt.date(2024, 12, 31)

    signal_names = [
        "momentum",
        "constant_volatility_scaled_momentum",
        "semi_volatility_scaled_momentum",
    ]

    returns_list = []
    for signal_name in signal_names:
        base_scan = pl.scan_parquet(f"weights/{signal_name}/gamma_{gamma}/{signal_name}_*.parquet")
        
        match rebalance_frequency:
            case 'daily':
                weights = base_scan
            
            case 'monthly':
                month_end_dates = (
                    base_scan
                    .with_columns(
                        pl.col('date').dt.strftime("%Y%m").alias('year_month')
                    )
                    .group_by('year_month')
                    .agg(pl.col('date').max())
                    .select('date')
                )
                weights = base_scan.join(month_end_dates, on=['date'], how='inner')
        
        weights = weights.filter(pl.col("date").is_between(start, end)).collect()

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
        case "daily":
            annual_factor = 252
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
        .sort('signal')
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
    file_path = f"results/experiment_10/{rebalance_frequency}_combined"
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

    plt.figure(figsize=(10, 6))

    sns.lineplot(cumulative_returns, x="Date", y="Return", hue="Signal")

    plt.title("Momentum Variations")
    plt.xlabel(None)
    plt.ylabel("Cumulative Log Returns (%)")

    plt.savefig(Path(file_path + "_chart").with_suffix(".png"))

if __name__ == '__main__':
    experiment_10()
