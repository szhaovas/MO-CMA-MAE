#!/bin/bash
#SBATCH --job-name=overcooked_nsga2
#SBATCH --output=experiment_output_%A_%a.log
#SBATCH --error=experiment_error_%A_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=48:00:00
#SBATCH --tasks-per-node=1
#SBATCH --account=nikolaid_548
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load usc
module load gcc/12.3.0
module load cuda/12.2.1

singularity exec --nv -H $PWD:/home --env MALLOC_TRIM_THRESHOLD_=0,XLA_PYTHON_CLIENT_PREALLOCATE=false gpu_container.sif python -m src.main num_trials=1 env=overcooked dask=32_cpu alg=nsga2 seed=42
