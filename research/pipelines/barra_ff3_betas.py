import datetime as dt
from pathlib import Path

import pandas as pd
import polars as pl
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from tqdm import tqdm


def barra_ff3_betas_flow(
    start: dt.date, end: dt.date
) -> None:
    df_barra = pl.read_parquet("data/barra/barra_*.parquet").sort("barrid", "date")

    df_ff3fm = pl.read_parquet("data/fama_french_factors/fama_french_factors.parquet")

    df_merge = (
        df_barra.join(other=df_ff3fm, on="date", how="left")
        .with_columns(pl.col("return").sub("rf").alias("return_rf"))
        .filter(pl.col("date").is_between(start, end))
        .sort("barrid", "date")
    )

    # Regression model
    def rolling_ff3_regression(group: pd.DataFrame, window: int):
        """
        Rolling 3-factor Fama-French regression for a single BARRID
        window=36 for 3 years of monthly data
        """
        # Sort by date
        group = group.sort_values("date").reset_index(drop=True)

        # Check if we have enough observations
        if len(group) < window:
            return group

        # Prepare variables
        y = group["return_rf"]  # Stock excess return
        X = group[["mkt_rf", "smb", "hml"]]
        X = sm.add_constant(X)  # Add intercept

        # Run rolling OLS
        model = RollingOLS(y, X, window=window, min_nobs=window)
        results = model.fit()

        # Extract results
        group["alpha"] = results.params["const"]
        group["beta_mkt"] = results.params["mkt_rf"]
        group["beta_smb"] = results.params["smb"]
        group["beta_hml"] = results.params["hml"]

        return group

    # Compute regression coefficients
    tqdm.pandas(desc="Computing model coefficients")
    df_betas: pl.DataFrame = pl.from_pandas(
        df_merge.to_pandas()
        .groupby(by="barrid")
        .progress_apply(
            lambda x: rolling_ff3_regression(
                x[["date", "return_rf", "mkt_rf", "smb", "hml"]], window=36 * 21
            ),
            include_groups=False,
        )
        .reset_index(level=0)
        .reset_index(drop=True)
    )

    df_clean = df_betas.select(
        pl.col("date").dt.date(), "barrid", "alpha", "beta_mkt", "beta_smb", "beta_hml"
    )

    min_year = df_clean["date"].min().year
    max_year = df_clean["date"].max().year
    years = list(range(min_year, max_year + 1))

    for year in tqdm(years, "Writing Barra FF3 Betas"):
        df_year = df_clean.filter(pl.col("date").dt.year().eq(year))

        # Create output directory
        file_path = Path(f"data/barra_ff3_betas/barra_ff3_betas_{year}.parquet")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        df_year.write_parquet(file_path)
