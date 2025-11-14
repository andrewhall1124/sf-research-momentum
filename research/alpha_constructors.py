import polars as pl

from research.models import AlphaConstructor


def cross_sectional_z_score(signal_name: str) -> AlphaConstructor:
    signal = pl.col(signal_name)
    score = signal.sub(signal.mean()).truediv(signal.std()).over("date")
    return AlphaConstructor(
        name=cross_sectional_z_score.__name__,
        expr=pl.lit(0.05).mul(score).mul(pl.col("specific_risk")).alias("alpha"),
        columns=["specific_risk"],
    )


def get_alpha_constructor(alpha_constructor_name: str, signal_name) -> AlphaConstructor:
    match alpha_constructor_name:
        case "cross-sectional-z-score":
            return cross_sectional_z_score(signal_name=signal_name)
        case _:
            raise ValueError(
                f"Alpha constructor not implemented: {alpha_constructor_name}"
            )


def construct_alphas(
    signals: pl.DataFrame, alpha_constructor: AlphaConstructor
) -> pl.DataFrame:
    return signals.with_columns(alpha_constructor.expr).with_columns(
        pl.col("alpha").fill_null(0)
    )
