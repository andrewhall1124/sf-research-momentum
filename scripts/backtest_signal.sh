#!/bin/bash
#SBATCH --job-name=backtest_signals
#SBATCH --output=logs/backtest_%A_%a.out
#SBATCH --error=logs/backtest_%A_%a.err
#SBATCH --array=0-74%30   # 75 tasks total (3 signals × 25 years), max 30 running at once
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --mail-user=amh1124@byu.edu
#SBATCH --mail-type=BEGIN,END,FAIL 

# Create logs directory if it doesn't exist
mkdir -p logs

# Parameters
gamma=60

# Define signals
signals=("momentum" "volatility_scaled_idiosyncratic_momentum_fama_french_3" "idiosyncratic_momentum_fama_french_3")

# Define years from 2000 to 2024 (25 years)
years=($(seq 2000 2024))

# Calculate dimensions
num_signals=${#signals[@]}
num_years=${#years[@]}
total_tasks=$((num_signals * num_years))

# Validate task ID
if [ $SLURM_ARRAY_TASK_ID -ge $total_tasks ]; then
  echo "Task ID $SLURM_ARRAY_TASK_ID is out of range (max $((total_tasks-1)))."
  exit 1
fi

# Map task ID to signal and year
signal_index=$(( SLURM_ARRAY_TASK_ID / num_years ))
year_index=$(( SLURM_ARRAY_TASK_ID % num_years ))

signal=${signals[$signal_index]}
year=${years[$year_index]}

# Activate virtual environment
source .venv/bin/activate

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Signal: $signal"
echo "Year: $year"
echo "Gamma: $gamma"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 20G"
echo "=========================================="

# Run the Python script (no dataset parameter needed)
srun python backtester "$signal" "$year" "$gamma"

# Check exit status
if [ $? -eq 0 ]; then
    echo "✓ Task completed successfully: signal=$signal, year=$year gamma=$gamma"
else
    echo "✗ Task failed: signal=$signal, year=$year gamma=$gamma"
    exit 1
fi