import polars as pl
from models import Config
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

pl.Config.set_tbl_rows(n=11)

def create_summary_table(returns: pl.DataFrame, config: Config, annualize: bool, file_path: Path) -> pl.DataFrame:
    annual_factor = 1

    if annualize:
        match config.rebalance_frequency:
            case 'daily':
                annual_factor = 252
            case 'monthly':
                annual_factor = 12
            case _:
                raise ValueError(f"{config.rebalance_frequency} is not supported!")

    summary_table = (
        returns
        .unpivot(index='date', variable_name='bin', value_name='return')
        .group_by('bin')
        .agg(
            pl.col('return').mean().mul(100 * annual_factor).alias('mean_return'),
            pl.col('return').std().mul(100 * np.sqrt(annual_factor)).alias('volatility'),
        )
        .with_columns(
            pl.col('mean_return').truediv('volatility').alias('sharpe')
        )
        .with_columns(
            pl.exclude('bin').round(2)
        )
        .sort('bin')
    )

    # Save the DataFrame as a string to the file
    output_file = file_path.with_suffix('.txt') if isinstance(file_path, Path) else Path(file_path).with_suffix('.txt')
    with open(output_file, 'w') as f:
        f.write(f"{config.name}\n")
        f.write(f"Period: {config.start} to {config.end}\n\n")
        f.write(str(summary_table))

    return summary_table


def create_quantile_returns_chart(returns: pl.DataFrame, config: Config, file_path: str) -> None:
    df_cumulative_returns = (
        returns
        .sort('date')
        .with_columns(
            pl.exclude('date').log1p().cum_sum()
        )
    )


    plt.figure(figsize=(10, 6))

    labels = [str(i) for i in range(config.n_bins)]
    colors = sns.color_palette(palette="coolwarm", n_colors=10)

    for label, color in zip(labels, colors):
        sns.lineplot(df_cumulative_returns, x="date", y=label, color=color)

    sns.lineplot(df_cumulative_returns, x="date", y="spread", color="green")

    plt.title(config.name)
    plt.xlabel(None)
    plt.ylabel("Cumulative Log Return (%)")

    output_file = Path(file_path).with_suffix('.png')
    plt.savefig(output_file, dpi=300)