import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import statsmodels.formula.api as smf

from research.models import QuantileBacktestConfig

pl.Config.set_tbl_rows(n=11)
pl.Config.set_tbl_cols(n=10)


def create_quantile_summary_table(
    returns: pl.DataFrame, config: QuantileBacktestConfig, file_path: Path
) -> pl.DataFrame:
    factors = pl.read_parquet("data/fama_french/ff5_factors.parquet")

    annual_factor = 1

    if config.annualize_results:
        match config.rebalance_frequency:
            case "daily":
                annual_factor = 252
            case "monthly":
                annual_factor = 12
            case _:
                raise ValueError(f"{config.rebalance_frequency} is not supported!")

    bins = [str(i) for i in range(config.n_bins)] + ["spread"]
    merged = (
        returns.unpivot(index="date", variable_name="bin", value_name="return")
        .join(other=factors, on="date", how="left")
        .with_columns(pl.col("return").sub("rf").alias("return_rf"))
    )

    # Define regression specifications
    regressions = [
        {"name": "CAPM", "formula": "return_rf ~ mkt_rf"},
        {"name": "3FM", "formula": "return_rf ~ mkt_rf + smb + hml"},
        {"name": "5FM", "formula": "return_rf ~ mkt_rf + smb + hml + rmw + cma"},
    ]

    bins = [str(i) for i in range(config.n_bins)] + ["spread"]

    # Run regressions for each model and bin
    regression_results_list = []
    for bin_value in bins:
        # Filter once per bin
        bin_data = merged.filter(pl.col("bin") == bin_value).to_pandas()

        result_row = {"bin": bin_value}
        for regression in regressions:
            model = smf.ols(formula=regression["formula"], data=bin_data)
            fitted = model.fit()

            # Extract alpha and t-stat
            result_row[f"{regression['name']}_alpha"] = fitted.params["Intercept"] * 100
            result_row[f"{regression['name']}_tstat"] = fitted.tvalues["Intercept"]

        regression_results_list.append(result_row)

    # Convert to polars and pivot for better readability
    regression_results = pl.DataFrame(regression_results_list)

    summary_table = (
        merged.group_by("bin")
        .agg(
            pl.col("return_rf").mean().mul(100 * annual_factor).alias("excess_return"),
            pl.col("return")
            .std()
            .mul(100 * np.sqrt(annual_factor))
            .alias("volatility"),
        )
        .with_columns(pl.col("excess_return").truediv("volatility").alias("sharpe"))
        .join(other=regression_results, on="bin", how="left")
        .with_columns(pl.exclude("bin").round(2))
        .sort("bin")
    )

    # Save the DataFrame as a string to the file
    output_file = (
        file_path.with_suffix(".txt")
        if isinstance(file_path, Path)
        else Path(file_path).with_suffix(".txt")
    )
    with open(output_file, "w") as f:
        f.write(f"{config.name}\n")
        f.write(f"Period: {config.start} to {config.end}\n")
        f.write(f"Annualized: {'Yes' if config.annualize_results else 'No'}\n\n")
        f.write(str(summary_table))

    return summary_table


def create_quantile_returns_chart(
    returns: pl.DataFrame, config: QuantileBacktestConfig, file_path: str
) -> None:
    df_cumulative_returns = returns.sort("date").with_columns(
        pl.exclude("date").log1p().cum_sum()
    )

    plt.figure(figsize=(10, 6))

    labels = [str(i) for i in range(config.n_bins)]
    colors = sns.color_palette(palette="coolwarm", n_colors=config.n_bins)

    for i, (label, color) in enumerate(zip(labels, colors)):
        sns.lineplot(
            df_cumulative_returns,
            x="date",
            y=label,
            color=color,
            label=f"D{i + 1}",
        )

    sns.lineplot(
        df_cumulative_returns,
        x="date",
        y="spread",
        color="green",
        label="Spread (D10-D1)",
    )

    plt.title(config.name)
    plt.xlabel(None)
    plt.ylabel("Cumulative Log Return (%)")
    plt.legend(loc="best")

    output_file = Path(file_path).with_suffix(".png")
    plt.savefig(output_file, dpi=300)


def create_mve_summary_table(
    returns: pl.DataFrame,
    file_path: Path,
    annualize_results: bool,
    rebalance_frequency: str,
    name: str,
    start: dt.date,
    end: dt.date,
) -> pl.DataFrame:
    annual_factor = 1
    if annualize_results:
        match rebalance_frequency:
            case "daily":
                annual_factor = 252
            case "monthly":
                annual_factor = 12
            case _:
                raise ValueError(f"{rebalance_frequency} is not supported!")

    summary_table = (
        returns.select(
            pl.lit("total").alias("portfolio"),
            pl.col("return").mean().mul(100 * annual_factor).alias("mean_return"),
            pl.col("return")
            .std()
            .mul(100 * np.sqrt(annual_factor))
            .alias("volatility"),
        )
        .with_columns(pl.col("mean_return").truediv("volatility").alias("sharpe"))
        .with_columns(pl.exclude("portfolio").round(2))
    )

    # Save the DataFrame as a string to the file
    output_file = (
        file_path.with_suffix(".txt")
        if isinstance(file_path, Path)
        else Path(file_path).with_suffix(".txt")
    )
    with open(output_file, "w") as f:
        f.write(f"{name}\n")
        f.write(f"Period: {start} to {end}\n")
        f.write(f"Rebalance Frequency: {rebalance_frequency}\n")
        f.write(f"Annualized: {'Yes' if annualize_results else 'No'}\n\n")
        f.write(str(summary_table))

    return summary_table


def create_mve_returns_chart(returns: pl.DataFrame, name: str, file_path: Path) -> None:
    cumulative_returns = returns.sort("date").with_columns(
        pl.col("return").log1p().cum_sum()
    )

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        cumulative_returns,
        x="date",
        y="return",
    )

    plt.title(name)
    plt.xlabel(None)
    plt.ylabel("Cumulative Log Return (%)")

    output_file = Path(file_path).with_suffix(".png")
    plt.savefig(output_file, dpi=300)
