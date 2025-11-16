import datetime as dt

import polars as pl

from research.models import AlphaConstructor, Constraint, Filter, Signal


def _get_columns(
    signal: Signal,
    filters: list[Filter],
    constraints: list[Constraint],
    alpha_constructor: AlphaConstructor,
    id_col: str,
) -> list[str]:
    # Get signal columns
    signal_columns = signal.columns

    # Get filter columns
    filter_columns = []
    for filter_ in filters:
        filter_columns.extend(filter_.columns)

    # Get constraint columns
    constraint_columns = []
    for constraint in constraints:
        constraint_columns.extend(constraint.columns)

    # Get alpha constructor columns
    alpha_constructor_columns = alpha_constructor.columns

    # Always include 'date' and id_col for the forward return calculation
    all_columns = list(
        set(
            signal_columns
            + filter_columns
            + constraint_columns
            + alpha_constructor_columns
            + ["date", id_col]
        )
    )

    return all_columns


def _get_fwd_return_expr(rebalance_frequency: str, id_col: str) -> pl.Expr:
    holding_period = None
    match rebalance_frequency:
        case "daily":
            holding_period = 1
        case "monthly":
            holding_period = 21
        case _:
            raise ValueError(f"{rebalance_frequency} is not supported!")

    return (
        pl.col("return")
        .log1p()
        .rolling_sum(window_size=holding_period)
        .exp()
        .sub(1)
        .shift(-holding_period)
        .over(id_col)
        .alias("fwd_return")
    )


def load_data(
    start: dt.date,
    end: dt.date,
    rebalance_frequency: str,
    datasets: list[str],
    signal: Signal,
    filters: list[Filter] | None = None,
    constraints: list[Constraint] | None = None,
    alpha_constructor: AlphaConstructor | None = None,
) -> pl.DataFrame:
    # Create base dataframe
    data = None
    id_col = None
    if "crsp" in datasets:
        data = pl.scan_parquet("data/crsp/crsp_*.parquet")
        id_col = "permno"

    if "barra" in datasets:
        data = pl.scan_parquet("data/barra/barra_*.parquet")
        id_col = "barrid"

    if data is None:
        raise ValueError("No base dataset provided")

    if "ff3" in datasets:
        data = data.join(
            pl.scan_parquet("data/fama_french/ff3_factors.parquet"),
            on="date",
            how="left",
        )

    if "crsp_ff3_betas" in datasets:
        data = data.join(
            pl.scan_parquet("data/crsp_ff3_betas/crsp_ff3_betas_*.parquet"),
            on=["date", "permno"],
            how="left",
        )

    if "barra_ff3_betas" in datasets:
        data = data.join(
            pl.scan_parquet("data/barra_ff3_betas/barra_ff3_betas_*.parquet"),
            on=["date", "barrid"],
            how="left",
        )

    columns = _get_columns(
        signal=signal,
        filters=filters,
        constraints=constraints,
        alpha_constructor=alpha_constructor,
        id_col=id_col,
    )

    data = (
        data.select(columns)
        .sort(id_col, "date")
        .with_columns(
            _get_fwd_return_expr(rebalance_frequency=rebalance_frequency, id_col=id_col)
        )
        .filter(pl.col("date").is_between(start, end))
        .sort(id_col, "date")
    )

    return data.collect()
