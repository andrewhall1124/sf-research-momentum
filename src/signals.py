import polars as pl


def cross_sectional_momentum() -> pl.Expr:
    return (
        pl.col("return")
        .log1p()
        .rolling_sum(window_size=230)
        .shift(22)
        .over("barrid")
        .alias(cross_sectional_momentum.__name__)
    )


def time_series_momentum() -> pl.Expr:
    momentum = pl.col("return").log1p().rolling_sum(window_size=230).shift(22)
    z_score_window = 252 * 3
    rolling_mean = momentum.rolling_mean(window_size=z_score_window)
    rolling_std = momentum.rolling_std(window_size=z_score_window)

    return (
        momentum.sub(rolling_mean)
        .truediv(rolling_std)
        .over("barrid")
        .alias(time_series_momentum.__name__)
    )


def fifty_two_week_high_momentum() -> pl.Expr:
    # I use the cumulative return as a proxy to adjust for splits
    price = pl.col("return").add(1).cum_prod().sub(1)
    fifty_two_week_high = price.rolling_max(window_size=252)
    return (
        price.truediv(fifty_two_week_high)
        .shift(22)
        .over("barrid")
        .alias(fifty_two_week_high_momentum.__name__)
    )


def frog_in_the_pan_momentum() -> pl.Expr:
    pos_days = (pl.col("return") > 0).cast(pl.Int32).rolling_sum(window_size=252)
    neg_days = (pl.col("return") < 0).cast(pl.Int32).rolling_sum(window_size=252)
    zero_days = (pl.col("return") == 0).cast(pl.Int32).rolling_sum(window_size=252)

    total_days = pos_days + neg_days + zero_days

    # Discreteness measure: (N+ - N-) / (N+ + N- + N0)
    # Higher discreteness means more jumpy/discrete price movements
    discreteness = (pos_days - neg_days) / total_days
    continuousness = 1 - discreteness

    momentum = pl.col("return").log1p().rolling_sum(window_size=252).shift(22)

    return (
        (momentum * continuousness)
        .over("barrid")
        .alias(frog_in_the_pan_momentum.__name__)
    )


def volatillity_scaled_momentum() -> pl.Expr:
    momentum = pl.col("return").log1p().rolling_sum(window_size=230).shift(22)
    volatillity = pl.col("return").log1p().rolling_std(window_size=230).shift(22)
    return (
        momentum.truediv(volatillity)
        .over("barrid")
        .alias(volatillity_scaled_momentum.__name__)
    )


def idiosyncratic_momentum() -> pl.Expr:
    return (
        pl.col("specific_return")
        .log1p()
        .rolling_sum(window_size=230)
        .shift(22)
        .over("barrid")
        .alias(idiosyncratic_momentum.__name__)
    )
