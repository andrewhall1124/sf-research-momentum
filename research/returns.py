import polars as pl

def construct_returns(
    data: pl.DataFrame, n_bins: int, rebalance_frequency: str
) -> pl.DataFrame:
    labels = [str(i) for i in range(n_bins)]
    top_bin, bottom_bin = labels[-1], labels[0]

    holding_period = 1 # TODO: Is this always 1?
    match rebalance_frequency:
        case "daily":
            holding_period = 1
        case "monthly":
            holding_period = 21
        case _:
            raise ValueError(
                f"Rebalance frequency not implemented: {rebalance_frequency}"
            )

    forward_returns = (
        pl.scan_parquet("data/crsp/crsp_*.parquet")
        .sort("permno", "date")
        .select(
            "date",
            "permno",
            pl.col("return")
            .log1p()
            .rolling_sum(window_size=holding_period)
            .shift(-holding_period)
            .exp()
            .sub(1)
            .over('permno')
            .alias("fwd_return"),
        )
    )

    data = (
        data.lazy()
        .join(
            other=forward_returns,
            on=['date', 'permno'],
            how='left'
        )
        .collect()
    )


    if rebalance_frequency == "daily":
        result = (
            data.group_by("date", "bin")
            .agg(pl.col("fwd_return").mul("weight").sum().alias("return"))
            .pivot(index="date", on="bin", values="return")
            .with_columns(pl.col(top_bin).sub(bottom_bin).alias("spread"))
            .sort("date")
        )

    elif rebalance_frequency == "monthly":
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
            f"Unsupported rebalance frequency: {rebalance_frequency}. Supported values are 'daily' or 'monthly'."
        )

    return result


def construct_returns_from_weights(
    weights: pl.DataFrame, rebalance_frequency: str
) -> pl.DataFrame:
    holding_period = None
    match rebalance_frequency:
        case "daily":
            holding_period = 1
        case "monthly":
            holding_period = 21
        case _:
            raise ValueError(
                f"Rebalance frequency not implemented: {rebalance_frequency}"
            )

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
            .over('barrid')
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

    return returns
