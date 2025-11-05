from src.models import Signal, Config
import polars as pl

def momentum() -> Signal:
    return Signal(
        name='momentum',
        expr=pl.col('return').log1p().rolling_sum(window_size=230).shift(22).over('permno').alias(momentum.__name__),
        columns=['permno', 'return'],
        lookback_days=252
    )

def idiosyncratic_momentum_ff3() -> Signal:
    residual = (
        pl.col("return").sub('rf')
        .sub(pl.col('alpha'))
        .sub(pl.col("beta_mkt_rf").mul("mkt_rf"))
        .sub(pl.col("beta_smb").mul("smb"))
        .sub(pl.col("beta_hml").mul("hml"))
        .alias("residual")
    )

    return Signal(
        name=idiosyncratic_momentum_ff3.__name__,
        expr=(
            residual
            .rolling_sum(230)
            .truediv(residual.rolling_std(230))
            .shift(22)
            .over('permno')
            .alias(idiosyncratic_momentum_ff3.__name__)
        ),
        columns=['permno', 'return', 'rf', 'alpha', 'mkt_rf', 'smb', 'hml', 'beta_mkt_rf', "beta_smb", "beta_hml"],
        lookback_days=252
    )

def get_signal(name: str) -> Signal:
    match name:
        case 'momentum':
            return momentum()
        case 'idiosyncratic-momentum-ff3':
            return idiosyncratic_momentum_ff3()
        case _:
            raise ValueError
        
def construct_signals(data: pl.DataFrame, config: Config) -> pl.DataFrame:
    return (
        data
        .sort('permno', 'date')
        .with_columns(
            config.signal.expr
        )
    )