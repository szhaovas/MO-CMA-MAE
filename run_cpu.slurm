#!/bin/bash
#SBATCH --job-name=sphere_mome
#SBATCH --output=experiment_output_%A_%a.log
#SBATCH --error=experiment_error_%A_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=48:00:00
#SBATCH --tasks-per-node=1
#SBATCH --account=nikolaid_548

module load usc

singularity exec --cleanenv -H $PWD:/home cpu_container.sif python -m src.main env=sphere dask=basic alg=mome seed=42
