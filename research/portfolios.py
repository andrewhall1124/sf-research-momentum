import polars as pl
import sf_quant.backtester as sfb

from research.models import Constraint, Signal


def construct_quantile_portfolios(
    data: pl.DataFrame, n_bins: int, signal: Signal, weighting_scheme: str, drop_null: bool = True
) -> pl.DataFrame:
    labels = [str(i) for i in range(n_bins)]
    portfolios = data.with_columns(
        pl.col(signal.name)
        # .rank(method='random', seed=47)
        .qcut(quantiles=n_bins, labels=labels)
        .cast(pl.String)
        .over("date")
        .alias("bin")
    )

    if drop_null:
        portfolios = portfolios.filter(pl.col('bin').ne('null'))

    if weighting_scheme == "equal":
        return portfolios.with_columns(
            pl.lit(1).truediv(pl.len()).over("bin", "date").alias("weight")
        )

    elif weighting_scheme == "market_cap":
        return portfolios.with_columns(
            pl.col("market_cap")
            .truediv(pl.col("market_cap").sum())
            .over("bin", "date")
            .alias("weight")
        )

    else:
        raise ValueError(f"{weighting_scheme} not supported!")


def construct_mve_portfolios(
    alphas: pl.DataFrame,
    rebalance_frequency: str,
    constraints: list[Constraint],
    gamma: float,
) -> pl.DataFrame:
    constraints = [c.constraint for c in constraints]

    if rebalance_frequency == "daily":
        pass

    elif rebalance_frequency == "monthly":
        month_end_dates = (
            alphas.with_columns(pl.col("date").dt.strftime("%Y%m").alias("year_month"))
            .group_by("year_month")
            .agg(pl.col("date").max())["date"]
            .unique()
            .sort()
            .to_list()
        )

        alphas = alphas.filter(pl.col("date").is_in(month_end_dates)).sort("date")

    else:
        raise ValueError(f"Rebalance frequency not implemented: {rebalance_frequency}")

    return sfb.backtest_parallel(data=alphas, constraints=constraints, gamma=gamma)
