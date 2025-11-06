import polars as pl
from models import Config


def construct_returns(data: pl.DataFrame, config: Config) -> pl.DataFrame:
    labels = [str(i) for i in range(config.n_bins)]
    top_bin, bottom_bin = labels[-1], labels[0]
    if config.rebalance_frequency == "daily":
        result = (
            data.group_by("date", "bin")
            .agg(pl.col("fwd_return").mul("weight").sum().alias("return"))
            .pivot(index="date", on="bin", values="return")
            .with_columns(pl.col(top_bin).sub(bottom_bin).alias("spread"))
            .sort("date")
        )

    elif config.rebalance_frequency == "monthly":
        year_months = (
            data.with_columns(pl.col("date").dt.strftime("%Y%m").alias("year_month"))
            .group_by("year_month")
            .agg(pl.col("date").max())["date"]
            .unique()
            .sort()
            .to_list()
        )

        result = (
            data.filter(pl.col("date").is_in(year_months))
            .group_by("date", "bin")
            .agg(pl.col("fwd_return").mul("weight").sum().alias("return"))
            .pivot(index="date", on="bin", values="return")
            .with_columns(pl.col(top_bin).sub(bottom_bin).alias("spread"))
            .sort("date")
        )
    else:
        raise ValueError(
            f"Unsupported rebalance frequency: {config.rebalance_frequency}. Supported values are 'daily' or 'monthly'."
        )

    return result
