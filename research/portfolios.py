import polars as pl
from models import Config


def construct_portfolios(data: pl.DataFrame, config: Config) -> pl.DataFrame:
    labels = [str(i) for i in range(config.n_bins)]
    portfolios = data.with_columns(
        pl.col(config.signal.name)
        .qcut(quantiles=config.n_bins, labels=labels)
        .cast(pl.String)
        .over("date")
        .alias("bin")
    )

    if config.weighting_scheme == "equal":
        return portfolios.with_columns(
            pl.lit(1).truediv(pl.len()).over("bin", "date").alias("weight")
        )

    elif config.weighting_scheme == "market_cap":
        return portfolios.with_columns(
            pl.col("market_cap")
            .truediv(pl.col("market_cap").sum())
            .over("bin", "date")
            .alias("weight")
        )

    else:
        raise ValueError(f"{config.weighting_scheme} not supported!")
