import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import statsmodels.formula.api as smf
import great_tables as gt

pl.Config.set_tbl_rows(n=11)
pl.Config.set_tbl_cols(n=10)


def create_quantile_summary_table(
    returns: pl.DataFrame,
    file_path: str,
    annualize_results: bool,
    rebalance_frequency: str,
    n_bins: int,
    start: dt.date,
    end: dt.date,
    title: str,
) -> pl.DataFrame:
    factors = pl.read_parquet("data/fama_french_factors/ff5.parquet")

    annual_factor = 1

    if annualize_results:
        match rebalance_frequency:
            case "daily":
                annual_factor = 252
            case "monthly":
                annual_factor = 12
            case _:
                raise ValueError(f"{rebalance_frequency} is not supported!")

    bins = [str(i) for i in range(n_bins)] + ["spread"]
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

    bins = [str(i) for i in range(n_bins)] + ["spread"]

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
        .with_columns(
            pl.col('bin').replace({
                '0': 'D1',
                '1': 'D2',
                '2': 'D3',
                '3': 'D4',
                '4': 'D5',
                '5': 'D6',
                '6': 'D7',
                '7': 'D8',
                '8': 'D9',
                '9': 'D10',
                'spread': 'Spread'
            })
        )
        .rename({
            'bin': 'Portfolio',
            'excess_return': 'Excess Return',
            'volatility': 'Volatility',
            'sharpe': 'Sharpe',
            'CAPM_alpha': 'CAPM Alpha',
            'CAPM_tstat': 'CAPM T-stat',
            '3FM_alpha': '3FM Alpha',
            '3FM_tstat': '3FM T-stat',
            '5FM_alpha': '5FM Alpha',
            '5FM_tstat': '5FM T-stat'
        })
    )

    # Save the DataFrame as a string to the file
    output_file = Path(file_path + "_table").with_suffix(".png")


    table = (
        gt.GT(summary_table)
        .tab_header(title=title)
        .opt_stylize(style=5, color="gray")
        .tab_source_note(source_note=f"Period: {start} to {end}")
        .tab_source_note(
            source_note=f"Rebalance Frequency: {rebalance_frequency.title()}"
        )
        .tab_source_note(
            source_note=f"Annualized: {'Yes' if annualize_results else 'No'}"
        )
        .tab_options(source_notes_padding=gt.px(10))  # Add padding

    )

    table.save(output_file, web_driver='chrome')

    return summary_table


def create_quantile_returns_chart(
    returns: pl.DataFrame, n_bins: int, title: str, file_path: str
) -> None:
    df_cumulative_returns = returns.sort("date").with_columns(
        pl.exclude("date").log1p().cum_sum()
    )

    plt.figure(figsize=(10, 6))

    labels = [str(i) for i in range(n_bins)]
    colors = sns.color_palette(palette="coolwarm", n_colors=n_bins)

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

    plt.title(title)
    plt.xlabel(None)
    plt.ylabel("Cumulative Log Return (%)")
    plt.legend(loc="best")

    output_file = Path(file_path + "_chart").with_suffix(".png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300)


def create_mve_summary_table(
    returns: pl.DataFrame,
    file_path: Path,
    annualize_results: bool,
    rebalance_frequency: str,
    name: str,
    start: dt.date,
    end: dt.date,
) -> None:
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
        .with_columns(pl.col("portfolio").str.to_titlecase())
    )

    # Save the DataFrame as a string to the file
    output_file = Path(file_path + "_table").with_suffix(".png")

    table = (
        gt.GT(summary_table)
        .tab_header(title=name)
        .fmt_number(
            columns=[
                "mean_return",
                "volatility",
                "sharpe",
            ],
            decimals=2,
        )
        .cols_label(
            portfolio="Portfolio",
            mean_return="Mean Return",
            volatility="Volatility",
            sharpe="Sharpe",
        )
        .opt_stylize(style=5, color="gray")
        .tab_options(
            table_width="500px", container_height="auto", container_overflow_y="visible"
        )
        .tab_source_note(source_note=f"Period: {start} to {end}")
        .tab_source_note(
            source_note=f"Rebalance Frequency: {rebalance_frequency.title()}"
        )
        .tab_source_note(
            source_note=f"Annualized: {'Yes' if annualize_results else 'No'}"
        )
    )

    table.save(output_file, web_driver='chrome')


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

    output_file = Path(file_path + "_chart").with_suffix(".png")
    plt.savefig(output_file, dpi=300)
