#!/bin/bash
#SBATCH --job-name=overcooked_mocmamae
#SBATCH --output=experiment_output_%A_%a.log
#SBATCH --error=experiment_error_%A_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=48:00:00
#SBATCH --tasks-per-node=1
#SBATCH --account=nikolaid_548

module load usc

singularity exec --cleanenv -H $PWD:/home --env MALLOC_TRIM_THRESHOLD_=0 cpu_container.sif python -m src.main num_trials=1 dask=16_cpu batch_size=16 alg._emitter_dict.e1.num=1 env=overcooked alg=mo_cma_mae alg._emitter_dict.e1.selection_rule=filter alg._emitter_dict.e1.restart_rule=basic seed=46
