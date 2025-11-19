from crsp import crsp_history_flow
from barra import barra_history_flow
from fama_french_factors import fama_french_factors_history_flow
from crsp_ff3_betas import crsp_ff3_betas_flow
from barra_ff3_betas import barra_ff3_betas_flow
from alphas import alphas_flow
import datetime as dt

def main():
    start = dt.date(1963, 7, 31)
    end = dt.date(2024, 12, 31)

    # Base datasets
    crsp_history_flow(start, end)
    barra_history_flow(start, end)

    # Factor datasets
    fama_french_factors_history_flow()

    # Betas
    crsp_ff3_betas_flow(start, end)
    barra_ff3_betas_flow(start, end)

    # Alphas
    alphas_flow(start, end)


if __name__ == '__main__':
    main()