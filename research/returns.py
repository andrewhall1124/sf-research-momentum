import polars as pl
import sf_quant.data as sfd

from research.models import QuantileBacktestConfig


def construct_returns(
    data: pl.DataFrame, config: QuantileBacktestConfig
) -> pl.DataFrame:
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


def construct_returns_from_weights(
    weights: pl.DataFrame, rebalance_frequency: str
) -> pl.DataFrame:
    holding_period = 1 # TODO: Use the paradigm that weights are already in the correct frequency.
    # match rebalance_frequency:
    #     case "daily":
    #         holding_period = 1
    #     case "monthly":
    #         holding_period = 21
    #     case _:
    #         raise ValueError(
    #             f"Rebalance frequency not implemented: {rebalance_frequency}"
    #         )

    forward_returns = (
        pl.scan_parquet("data/barra/barra_*.parquet")
        .sort("barrid", "date")
        .select(
            "date",
            "barrid",
            pl.col("return")
            .log1p()
            .rolling_sum(window_size=holding_period)
            .shift(-holding_period)
            .exp()
            .sub(1)
            .alias("fwd_return"),
        )
    )

    returns = (
        weights.lazy()
        .join(other=forward_returns, on=["date", "barrid"], how="left")
        .group_by("date")
        .agg(pl.col("weight").mul(pl.col("fwd_return")).sum().alias("return"))
        .collect()
    )

    if rebalance_frequency == "daily":
        return returns

    elif rebalance_frequency == "monthly":
        month_end_dates = (
            returns.with_columns(pl.col("date").dt.strftime("%Y%m").alias("year_month"))
            .group_by("year_month")
            .agg(pl.col("date").max())["date"]
            .unique()
            .sort()
            .to_list()
        )

        return returns.filter(pl.col("date").is_in(month_end_dates)).sort("date")

    else:
        raise ValueError(f"Rebalance frequency not implemented: {rebalance_frequency}")
