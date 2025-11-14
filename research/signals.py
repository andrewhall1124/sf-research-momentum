import polars as pl

from research.models import Signal


def momentum(id_col: str) -> Signal:
    return Signal(
        name=momentum.__name__,
        expr=pl.col("return")
        .log1p()
        .rolling_sum(window_size=230)
        .shift(22)
        .over(id_col)
        .alias(momentum.__name__),
        columns=[id_col, "return"],
        lookback_days=252,
    )


def idio_mom_vol_scaled_ff3() -> Signal:
    residual = (
        pl.col("return")
        .sub("rf")
        .sub(pl.col("alpha"))
        .sub(pl.col("beta_mkt").mul("mkt_rf"))
        .sub(pl.col("beta_smb").mul("smb"))
        .sub(pl.col("beta_hml").mul("hml"))
        .alias("residual")
    )

    return Signal(
        name=idio_mom_vol_scaled_ff3.__name__,
        expr=(
            residual.rolling_sum(230)
            .truediv(residual.rolling_std(230))
            .shift(22)
            .over("permno")
            .alias(idio_mom_vol_scaled_ff3.__name__)
        ),
        columns=[
            "permno",
            "return",
            "rf",
            "alpha",
            "mkt_rf",
            "smb",
            "hml",
            "beta_mkt",
            "beta_smb",
            "beta_hml",
        ],
        lookback_days=252,
    )


def idio_mom_ff3() -> Signal:
    residual = (
        pl.col("return")
        .sub("rf")
        .sub(pl.col("alpha"))
        .sub(pl.col("beta_mkt").mul("mkt_rf"))
        .sub(pl.col("beta_smb").mul("smb"))
        .sub(pl.col("beta_hml").mul("hml"))
        .alias("residual")
    )

    return Signal(
        name=idio_mom_ff3.__name__,
        expr=(
            residual.rolling_sum(230)
            .shift(22)
            .over("permno")
            .alias(idio_mom_ff3.__name__)
        ),
        columns=[
            "permno",
            "return",
            "rf",
            "alpha",
            "mkt_rf",
            "smb",
            "hml",
            "beta_mkt",
            "beta_smb",
            "beta_hml",
        ],
        lookback_days=252,
    )


def get_signal(name: str, id_col: str) -> Signal:
    match name:
        case "momentum":
            return momentum(id_col)
        case "idio_mom_vol_scaled_ff3":
            return idio_mom_vol_scaled_ff3()
        case "idio_mom_ff3":
            return idio_mom_ff3()
        case _:
            raise ValueError(f"{name} not implemented")


def construct_signals(data: pl.DataFrame, signal: Signal) -> pl.DataFrame:
    """Data is assumed to have already been sorted by id_col and date."""
    return data.with_columns(signal.expr)
