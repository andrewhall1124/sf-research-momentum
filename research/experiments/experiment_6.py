import polars as pl
import datetime as dt
import statsmodels.formula.api as smf
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import great_tables as gt

# Constants
START_DATE = dt.date(
    2016, 1, 1
)  # instead of 2018 so that we get the whole sample of observations starting in 2018
END_DATE = dt.date(2024, 12, 31)
VOL_TARGET = 4.35 / 100
LAMBDA = 1  #  .25
LOOKBACK_DAYS = 126
MONTHLY_DAYS = 21


def load_month_end_dates(parquet_path: str) -> pl.DataFrame:
    """Extract month-end dates from momentum factor returns."""
    return (
        pl.scan_parquet(parquet_path)
        .with_columns(pl.col("date").dt.strftime("%Y%m").alias("year_month"))
        .group_by("year_month")
        .agg(pl.col("date").max())
        .select("date")
        .collect()
    )


def load_daily_momentum_returns(
    parquet_path: str, start: dt.date, end: dt.date
) -> pl.DataFrame:
    """Load and filter daily momentum returns."""
    return (
        pl.scan_parquet(parquet_path)
        .filter(pl.col("date").is_between(start, end))
        .sort("date")
        .select("date", pl.col("mom").alias("return"))
        .collect()
    )


def calculate_monthly_return() -> pl.Expr:
    """Create expression for calculating monthly returns from daily data."""
    return (
        pl.col("return")
        .log1p()
        .rolling_sum(window_size=MONTHLY_DAYS)
        .exp()
        .sub(1)
        .shift(-MONTHLY_DAYS)
        .alias("return_monthly")
    )


def calculate_volatility_forecast() -> pl.Expr:
    """Calculate rolling volatility forecast scaled to monthly."""
    return (
        pl.col("return")
        .pow(2)
        .truediv(LOOKBACK_DAYS)
        .rolling_sum(window_size=LOOKBACK_DAYS)
        .mul(MONTHLY_DAYS)
        .alias("vol_forecast")
    )


def calculate_mom_strategy(
    daily_returns: pl.DataFrame, month_end_dates: pl.DataFrame
) -> pl.DataFrame:
    """Calculate basic momentum strategy returns."""
    return (
        daily_returns.with_columns(calculate_monthly_return())
        .with_columns(pl.lit("mom").alias("signal"))
        .select("date", "signal", "return_monthly")
        .join(month_end_dates, on="date", how="inner")
        .sort("date")
    )


def calculate_cmom_strategy(
    daily_returns: pl.DataFrame, month_end_dates: pl.DataFrame
) -> pl.DataFrame:
    """Calculate constant volatility momentum strategy."""
    return (
        daily_returns.sort("date")
        .with_columns(calculate_volatility_forecast())
        .with_columns(
            pl.lit(VOL_TARGET).truediv(pl.col("vol_forecast").sqrt()).alias("weight")
        )
        .with_columns(calculate_monthly_return())
        .with_columns(pl.col("return_monthly").mul(pl.col("weight")))
        .with_columns(pl.lit("cmom").alias("signal"))
        .select("date", "signal", "return_monthly")
        .join(month_end_dates, on="date", how="inner")
        .sort("date")
    )


def calculate_smom_strategy(
    daily_returns: pl.DataFrame, month_end_dates: pl.DataFrame
) -> pl.DataFrame:
    """Calculate semi-variance momentum strategy."""
    return (
        daily_returns.with_columns(
            pl.when(pl.col("return") < 0)
            .then(pl.col("return").pow(2))
            .otherwise(0)
            .alias("return_squared_neg")
        )
        .with_columns(
            pl.col("return_squared_neg")
            .rolling_sum(window_size=LOOKBACK_DAYS)
            .mul(MONTHLY_DAYS / LOOKBACK_DAYS)
            .sqrt()
            .alias("semi_vol_forecast")
        )
        .with_columns(
            pl.lit(VOL_TARGET).truediv(pl.col("semi_vol_forecast")).alias("weight")
        )
        .with_columns(calculate_monthly_return())
        .with_columns(pl.col("return_monthly").mul(pl.col("weight")))
        .with_columns(pl.lit("smom").alias("signal"))
        .select("date", "signal", "return_monthly")
        .join(month_end_dates, on="date", how="inner")
        .sort("date")
    )


def load_market_data(start: dt.date, end: dt.date) -> pl.DataFrame:
    """Load and prepare market data with bear indicator and interaction term."""
    return (
        pl.scan_parquet("data/fama_french_factors/ff3.parquet")
        .join(
            pl.scan_parquet(
                "data/momentum_factor_returns/momentum_factor_returns.parquet"
            ).select("date", pl.col("mom").alias("r_mom")),
            on="date",
            how="left",
        )
        .select(
            "date",
            "r_mom",
            pl.col("mkt_rf").add(pl.col("rf")).alias("rm"),
            pl.col("mkt_rf").alias("rmrf"),
        )
        .filter(pl.col("date").is_between(start, end))
        .sort("date")
        .with_columns(
            pl.col("rm").log1p().rolling_sum(252 * 2).exp().sub(1).alias("rm_2y")
        )
        .with_columns(
            pl.when(pl.col("rm_2y").lt(0))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("bear_indicator")
        )
        .with_columns(pl.col("rmrf").rolling_var(LOOKBACK_DAYS).alias("rmrf_variance"))
        .sort("date")
        .with_columns(pl.col("r_mom").shift(-1))
        .select("date", "r_mom", "bear_indicator", "rmrf_variance")
        .collect()
    )


def get_month_dates(market_data: pl.DataFrame) -> list:
    """Extract unique month-end dates from market data."""
    return (
        market_data.with_columns(pl.col("date").dt.strftime("%Y%m").alias("year_month"))
        .group_by("year_month")
        .agg(pl.col("date").max())["date"]
        .unique()
        .sort()
        .to_list()
    )


def estimate_coefficients(data: pl.DataFrame) -> dict[str, None | float]:
    """Estimate regression coefficients for dynamic momentum strategy."""
    data = data.drop_nulls().with_columns(pl.col('bear_indicator').mul('rmrf_variance').alias('interaction'))

    if len(data) == 0:
        return {"gamma_0": None, "gamma_1": None}

    model = smf.ols(formula="r_mom ~ interaction", data=data)
    results = model.fit()

    return {
        "gamma_0": results.params["Intercept"],
        "gamma_1": results.params["interaction"],
    }


def calculate_rolling_coefficients(
    market_data: pl.DataFrame, month_dates: list
) -> pl.DataFrame:
    """Calculate rolling regression coefficients for each month."""
    coefficients_list = []
    for month_date in tqdm(month_dates, desc="Computing coefficients"):
        regression_data = market_data.filter(pl.col("date").le(month_date))
        coefficients = estimate_coefficients(regression_data)
        coefficients_list.append({"date": month_date} | coefficients)

    return pl.DataFrame(coefficients_list)


def calculate_dmom_strategy(
    daily_returns: pl.DataFrame,
    coefficients: pl.DataFrame,
    market_data: pl.DataFrame,
    month_end_dates: pl.DataFrame,
) -> pl.DataFrame:
    """Calculate dynamic momentum strategy."""
    return (
        daily_returns.join(coefficients, on="date", how="left")
        .join(market_data, on="date", how="left")
        .sort("date")
        .with_columns(calculate_volatility_forecast())
        .with_columns(
            pl.col("gamma_0")
            .add(pl.col("gamma_1").mul(pl.col('bear_indicator').mul('rmrf_variance')))
            .alias("return_forecast")
        )
        .with_columns(
            pl.col("return_forecast")
            .truediv(pl.col("vol_forecast"))
            .mul(1 / (2 * LAMBDA))
            .alias("weight")
        )
        .with_columns(calculate_monthly_return())
        .with_columns(pl.col("return_monthly").mul("weight"))
        .with_columns(pl.lit("dmom").alias("signal"))
        .select("date", "signal", "return_monthly")
        .join(month_end_dates, on="date", how="inner")
        .sort("date")
    )


def create_summary_table(
    returns_monthly: pl.DataFrame, vol_scale: bool = False
) -> pl.DataFrame:
    """Calculate summary statistics for all strategies."""
    annual_factor = 12
    sort_key = {"mom": 1, "cmom": 2, "smom": 3, "dmom": 4}

    if vol_scale:
        vol_scaled_returns_monthly = (
            returns_monthly
            # Vol scale returns ex post for visuals
            .with_columns(
                pl.col("return_monthly")
                .mul(VOL_TARGET)
                .truediv(pl.col("return_monthly").std())
                .over("signal")
            )
        )
    else:
        vol_scaled_returns_monthly = returns_monthly

    summary_table = (
        vol_scaled_returns_monthly.group_by("signal")
        .agg(
            pl.col("return_monthly").mean().mul(100).alias("mean_return"),
            pl.col("return_monthly").std().mul(100).alias("volatility"),
        )
        .with_columns(
            pl.col("mean_return")
            .truediv(pl.col("volatility"))
            .mul(pl.lit(annual_factor).sqrt())
            .alias("sharpe")
        )
        .with_columns(pl.exclude("signal").round(2))
        .sort(by=pl.col("signal").replace(sort_key))
        .rename(
            {
                "signal": "Signal",
                "mean_return": "Mean Return",
                "volatility": "Volatility",
                "sharpe": "Sharpe (Annualized)",
            }
        )
    )

    output_file = Path(
        f"results/experiment_6/{'vol_scaled_' if vol_scale else ''}combined_table"
    ).with_suffix(".png")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    table = (
        gt.GT(summary_table)
        .tab_header(title="Volatility Scaled Momentum Variations")
        .opt_stylize(style=5, color="gray")
        .tab_source_note(source_note=f"Vol Scaled: {'Yes' if vol_scale else 'No'}")
        .tab_source_note(source_note=f"Period: {START_DATE} to {END_DATE}")
        .tab_source_note(source_note="Rebalance Frequency: Monthly")
        .tab_options(source_notes_padding=gt.px(10))  # Add padding
    )

    table.save(output_file, web_driver="chrome")


def create_returns_chart(
    returns_monthly: pl.DataFrame, vol_scale: bool = False, log_scale: bool = True
) -> None:
    if vol_scale:
        vol_scaled_returns_monthly = (
            returns_monthly
            # Vol scale returns ex post for visuals
            .with_columns(
                pl.col("return_monthly")
                .mul(VOL_TARGET)
                .truediv(pl.col("return_monthly").std())
                .over("signal")
            )
        )
    else:
        vol_scaled_returns_monthly = returns_monthly

    cumulative_returns = vol_scaled_returns_monthly.sort("signal", "date").with_columns(
        pl.col("return_monthly").log1p().cum_sum().mul(100).over("signal")
    )

    if not log_scale:
        cumulative_returns = cumulative_returns.with_columns(
            pl.col("return_monthly").truediv(100).exp().sub(1).mul(100)
        )

    file_path = Path(
        f"results/experiment_6/{'vol_scaled_' if vol_scale else ''}combined_chart"
    )
    file_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.lineplot(cumulative_returns, x="date", y="return_monthly", hue="signal")

    return_type = "Log" if log_scale else "Product"
    plt.title("Volatility Scaling Variations")
    plt.xlabel(None)
    plt.ylabel(f"Cumulative {return_type} Returns (%)")
    plt.legend()

    plt.savefig(file_path.with_suffix(".png"), dpi=300)


def experiment_6():
    """Main execution function."""
    # Load data
    month_end_dates = load_month_end_dates(
        "data/momentum_factor_returns/momentum_factor_returns.parquet"
    )
    daily_returns = load_daily_momentum_returns(
        "data/momentum_factor_returns/momentum_factor_returns.parquet",
        START_DATE,
        END_DATE,
    )
    market_data = load_market_data(START_DATE, END_DATE)

    # Calculate strategies
    mom_monthly = calculate_mom_strategy(daily_returns, month_end_dates)
    cmom_monthly = calculate_cmom_strategy(daily_returns, month_end_dates)
    smom_monthly = calculate_smom_strategy(daily_returns, month_end_dates)

    # Calculate dynamic momentum (requires regression coefficients)
    month_dates = get_month_dates(market_data)
    coefficients = calculate_rolling_coefficients(market_data, month_dates)
    dmom_monthly = calculate_dmom_strategy(
        daily_returns, coefficients, market_data, month_end_dates
    )

    # Combine and analyze
    returns_monthly = pl.concat(
        [mom_monthly, cmom_monthly, smom_monthly, dmom_monthly]
    ).sort("date")
    create_summary_table(returns_monthly, vol_scale=True)
    create_summary_table(returns_monthly, vol_scale=False)
    create_returns_chart(returns_monthly, vol_scale=True, log_scale=True)
    create_returns_chart(returns_monthly, vol_scale=False, log_scale=True)


if __name__ == "__main__":
    experiment_6()
