import polars as pl
import datetime as dt
import statsmodels.formula.api as smf
from tqdm import tqdm
from pathlib import Path

# Constants
LAMBDA = 1
LOOKBACK_DAYS = 126
MONTHLY_DAYS = 21

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

def dmom_coefficents_history_flow(start: dt.date, end: dt.date) -> None:
    market_data = load_market_data(start, end)
    print(market_data)
    month_dates = get_month_dates(market_data)
    coefficients = calculate_rolling_coefficients(market_data, month_dates)

    # Create output directory
    coefficients_path = Path("data/dmom_coefficients/dmom_coefficients.parquet")
    market_data_path = Path("data/dmom_coefficients/market_data.parquet")

    coefficients_path.parent.mkdir(parents=True, exist_ok=True)

    coefficients.write_parquet(coefficients_path)
    market_data.write_parquet(market_data_path)