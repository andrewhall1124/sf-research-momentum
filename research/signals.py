import polars as pl

from research.models import Signal


def momentum(id_col: str) -> Signal:
    return Signal(
        name="momentum",
        expr=pl.col("return")
        .log1p()
        .rolling_sum(window_size=230)
        .shift(22)
        .over(id_col)
        .alias("momentum"),
        columns=[id_col, "return"],
        lookback_days=252,
    )


def idio_mom_vol_scaled_ff3(id_col: str) -> Signal:
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
        name="volatility_scaled_idiosyncratic_momentum_fama_french_3",
        expr=(
            residual.rolling_sum(230)
            .truediv(residual.rolling_std(230))
            .shift(22)
            .over(id_col)
            .alias("volatility_scaled_idiosyncratic_momentum_fama_french_3")
        ),
        columns=[
            id_col,
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


def idio_mom_ff3(id_col: str) -> Signal:
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
        name="idiosyncratic_momentum_fama_french_3",
        expr=(
            residual.rolling_sum(230)
            .shift(22)
            .over(id_col)
            .alias("idiosyncratic_momentum_fama_french_3")
        ),
        columns=[
            id_col,
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

def cmom(id_col: str) -> Signal:
    momentum = (
        pl.col("return")
        .log1p()
        .rolling_sum(window_size=230) # 11 month momentum
        .shift(22)
        .over(id_col)
    )

    volatility_forecast = (
        pl.col("return")
        .pow(2)
        .truediv(126)
        .rolling_sum(window_size=126) # 6 month volatility
        .mul(21)
        .over(id_col)
    )

    vol_scaled_momentum = momentum / volatility_forecast

    clean_vol_scaled_momentum = (
        pl.when(vol_scaled_momentum.is_infinite())
        .then(pl.lit(None))
        .otherwise(vol_scaled_momentum)
    )

    return Signal(
        name="constant_volatility_scaled_momentum",
        expr=clean_vol_scaled_momentum.alias('constant_volatility_scaled_momentum'),
        columns=['return', id_col],
        lookback_days=252
    )

def smom(id_col: str) -> Signal:
    momentum = (
        pl.col("return")
        .log1p()
        .rolling_sum(window_size=230) # 11 month momentum
        .shift(22)
        .over(id_col)
    )

    return_squared_neg = (
        pl.when(pl.col("return") < 0)
        .then(pl.col("return").pow(2))
        .otherwise(0)
    )

    volatility_forecast = (
        return_squared_neg
        .rolling_sum(window_size=126)
        .mul(21 / 126)
        .sqrt()
        .over(id_col)
    )

    vol_scaled_momentum = momentum / volatility_forecast

    clean_vol_scaled_momentum = (
        pl.when(vol_scaled_momentum.is_infinite())
        .then(pl.lit(None))
        .otherwise(vol_scaled_momentum)
    )

    return Signal(
        name="semi_volatility_scaled_momentum",
        expr=clean_vol_scaled_momentum.alias('semi_volatility_scaled_momentum'),
        columns=['return', id_col],
        lookback_days=252
    )

def dmom(id_col: str) -> Signal:
    momentum = (
        pl.col("return")
        .log1p()
        .rolling_sum(window_size=230) # 11 month momentum
        .shift(22)
        .over(id_col)
    )
    volatility_forecast = (
        pl.col("return")
        .pow(2)
        .truediv(126)
        .rolling_sum(window_size=126) # 6 month volatility
        .mul(21)
        .over(id_col)
    )

    return_forecast = (
        pl.col('gamma_0').add(pl.col('gamma_1').mul(pl.col('bear_indicator').mul('rmrf_variance')))
    )
        
    return Signal(
        name="dynamic_volatility_scaled_momentum",
        expr=momentum.mul(return_forecast).truediv(volatility_forecast).alias('dynamic_volatility_scaled_momentum'),
        columns=['return', id_col],
        lookback_days=252
    )


def get_signal(name: str, id_col: str) -> Signal:
    match name:
        case "momentum":
            return momentum(id_col)
        case "volatility_scaled_idiosyncratic_momentum_fama_french_3":
            return idio_mom_vol_scaled_ff3(id_col)
        case "idiosyncratic_momentum_fama_french_3":
            return idio_mom_ff3(id_col)
        case "constant_volatility_scaled_momentum":
            return cmom(id_col)
        case "semi_volatility_scaled_momentum":
            return smom(id_col)
        case "dynamic_volatility_scaled_momentum":
            return dmom(id_col)
        case _:
            raise ValueError(f"{name} not implemented")


def construct_signals(data: pl.DataFrame, signal: Signal) -> pl.DataFrame:
    """Data is assumed to have already been sorted by id_col and date."""
    return data.with_columns(signal.expr)
