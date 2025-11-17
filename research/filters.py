import polars as pl

from research.models import Filter


def penny_stocks() -> Filter:
    return Filter(
        name=penny_stocks.__name__, expr=pl.col("price").gt(1), columns=["price"]
    )


def micro_caps() -> Filter:
    return Filter(
        name=micro_caps.__name__,
        expr=pl.col("market_cap").gt(pl.col("market_cap").quantile(0.20).over("date")),
        columns=["date", "market_cap"],
    )


def null_signal(signal_name: str) -> Filter:
    return Filter(
        name=null_signal.__name__, expr=pl.col(signal_name).is_not_null(), columns=[]
    )


def low_price_stocks() -> Filter:
    return Filter(
        name=low_price_stocks.__name__, expr=pl.col("price").gt(5), columns=["price"]
    )


def null_idiosyncratic_momentum() -> Filter:
    residual = (
        pl.col("return")
        .sub("rf")
        .sub(pl.col("alpha"))
        .sub(pl.col("beta_mkt").mul("mkt_rf"))
        .sub(pl.col("beta_smb").mul("smb"))
        .sub(pl.col("beta_hml").mul("hml"))
    )
    idiosyncratic_momentum = (
        residual.rolling_sum(230)
        .truediv(residual.rolling_std(230))
        .shift(22)
        .over("permno")
    )

    return Filter(
        name=null_idiosyncratic_momentum.__name__,
        expr=idiosyncratic_momentum.is_not_null(),
        columns=[
            "alpha",
            "beta_mkt",
            "beta_hml",
            "beta_smb",
            "mkt_rf",
            "hml",
            "smb",
            "rf",
        ],
    )


def get_filter(name: str, **kwargs) -> Filter:
    signal_name = kwargs.get("signal_name")
    match name:
        case "penny-stocks":
            return penny_stocks()
        case "micro-caps":
            return micro_caps()
        case "null-signal":
            return null_signal(signal_name)
        case "low-price-stocks":
            return low_price_stocks()
        case "null-idiosyncratic-momentum":
            return null_idiosyncratic_momentum()
        case _:
            raise ValueError


def apply_filters(signals: pl.DataFrame, filters: list[Filter]) -> pl.DataFrame:
    return signals.filter([filter_.expr for filter_ in filters])
