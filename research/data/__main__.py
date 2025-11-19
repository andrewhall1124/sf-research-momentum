from crsp import crsp_history_flow
from barra import barra_history_flow
from fama_french_factors import fama_french_factors_history_flow
from crsp_ff3_betas import crsp_ff3_betas_flow
from barra_ff3_betas import barra_ff3_betas_flow
from momentum_factor_returns import momentum_factor_returns_flow
from alphas import alphas_flow
import datetime as dt

def main():
    blitz_start = dt.date(1963, 7, 31)
    hanauer_start = dt.date(1930, 1, 1)
    barra_start = dt.date(1995, 7, 31)
    end = dt.date(2024, 12, 31)

    # Base datasets
    crsp_history_flow(hanauer_start, end)
    barra_history_flow(barra_start, end)

    # Factor datasets
    fama_french_factors_history_flow()

    # Betas
    crsp_ff3_betas_flow(blitz_start, end)
    barra_ff3_betas_flow(barra_start, end)

    # Alphas
    alphas_flow(barra_start, end)

    # Momentum factor returns
    momentum_factor_returns_flow()


if __name__ == '__main__':
    main()