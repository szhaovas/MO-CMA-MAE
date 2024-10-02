# Multi-Objective Covariance Matrix Adaptation MAP-Annealing

## Installation
This repository requires a Nvidia GPU with CUDA version >= 12.2 to run.

### Download the pre-built Singularity container
- Clone this repository.
- Download the pre-built container from [here](https://drive.google.com/file/d/15LbKdQi_bWUOLD_-8iTx1TPV-kL8tfh9/view?usp=sharing) and put it into the cloned repository.

### Or build the container manually
- Clone this repository.
- Follow the instructions [here](https://docs.sylabs.io/guides/3.0/user-guide/installation.html) to install Singularity, recommended version is 3.8.5.
- Install the academic version of CPLEX Optimization Studio by following the instructions [here](https://www.ibm.com/academic/).
    - Put the CPLEX Studio folder under this repository.
- Build the Singularity container by running
```
sudo singularity build container.sif container.def
```

## Runninng
- To start an experiment on domain `env`, and algorithm `alg`, run
```
singularity exec --nv -H $PWD:/home container.sif python -m src.main env=<env> alg=<alg>
```
- Logged metrics, pickled archives, and heatmaps generated throughout the experiment can be found at `outputs/<env>/<alg>/%Y-%m-%d_%H%M%S/`.
- Experiment configurations are located under `config`.
- Default parameters may be overwritten through the command line. For example, to run the default algorithm (MO-CMA-MAE) on the default env (sphere) but with `alpha=0.5`, run:
```
singularity exec --nv -H $PWD:/home container.sif python -m src.main env=<env> alg=<alg> alg.archive.alpha=0.5
```
- You can sweep over multiple parameters like this:
```
singularity exec --nv -H $PWD:/home container.sif python -m src.main --multirun env=<env> alg=<alg> alg.archive.alpha=0.2,0.3,0.4
```