# Momentum Variations Research
This repository contains the code for researching and testing various momentum trading strategies within the context of BYU Silver Fund's Quantitative Investment Team.

## Setup
We use `uv` for Python package management.

Sync your virtual environment with the `pyproject.toml` file.
```bash
uv sync
```

## Data
Load all of the necessary data for the project.

```bash
python research/data
```

This script will:
- Compiles CRSP daily dataset
- Compiles Barra daily dataset
- Fetches fama french daily factors
- Computes CRSP fama french 3 factor model betas
- Computes Barra fama french 3 factor model betas

## Experiments
Run all of the existing experiments.

```bash
python research/experiments
```

This script will create the results for each experiment in the results folder.

## Components
This repository makes extensive use of the following component files:
- `alpha_constructors.py`: Abstraction for taking a signal (i.e. momentum) and creating an alpha.
- `constratints.py`: Constraints for mean variance optimization.
- `filters.py`: Filters are applied to a dataset of assets to reduce it according to some attribute (i.e. price).
- `signals.py`: Abstraction for signal computation and requirements.

## Utilities
The following files contain utitlities that aid in the experimentation process.
- `evaluations.py`: Functions for generating various tables and charts.
- `models.py`: Various Python `dataclasses` and `dataframely` schemas.
- `portfolios.py`: Functions for computing quantile portfolios.
- `returns.py`: Functions for generating returns from MVO weights and quantile portfolios.
