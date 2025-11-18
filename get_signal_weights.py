import polars as pl
import sf_quant.backtester as sfb
import sf_quant.optimizer as sfo
from pathlib import Path
import click
import os

@click.command()
@click.argument('signal_name', type=str)
@click.argument('year', type=int)
@click.argument('gamma', type=float)
@click.option('--n-cpus', type=int, default=None, help='Number of CPUs to use')
def main(signal_name: str, year: int, gamma: float, n_cpus: int | None):
    """
    Backtest a signal for a specific year and generate portfolio weights.
    
    SIGNAL_NAME: Name of the signal to backtest
    YEAR: Year to process
    """
    # Get n_cpus from option, SLURM environment, or default to 8
    if n_cpus is None:
        n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
    
    click.echo(f"Processing signal={signal_name} for year={year} with {n_cpus} CPUs")
    
    alphas = (
        pl.scan_parquet(f"alphas/{signal_name}.parquet")
        .join(
            pl.scan_parquet("data/barra/barra_*.parquet").select('date', 'barrid', 'predicted_beta'),
            on=['date', 'barrid'],
            how='left'
        )
        .filter(pl.col("date").dt.year().eq(year))
        .collect()
    )
    
    click.echo(f"Alpha data shape: {alphas.shape}")
    
    constraints = [
        sfo.ZeroBeta()
    ]
    
    weights = sfb.backtest_parallel(
        data=alphas,
        constraints=constraints,
        gamma=gamma,
        n_cpus=n_cpus
    )
    
    output_path = Path(f"weights/{signal_name}/gamma_{gamma}/{signal_name}_{year}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weights.write_parquet(output_path.with_suffix(".parquet"))
    
    click.echo(f"✓ Saved weights to {output_path.with_suffix('.parquet')}")

if __name__ == '__main__':
    main()