import os
import csv
import copy
import pickle
import logging
import hydra
import numpy as np
import pickle as pkl
from pathlib import Path
from functools import partial
from hydra.utils import instantiate
from hydra.core.utils import setup_globals, configure_log
from omegaconf import OmegaConf, DictConfig, read_write
from dask.distributed import Client
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
from src.moribs.schedulers import Scheduler
from src.moribs.archives._pf_utils import cvt_archive_heatmap

logger = logging.getLogger(__name__)


def define_resolvers():

    if not OmegaConf.has_resolver("plus"):
        OmegaConf.register_new_resolver("plus", lambda x, y: x + y)

    if not OmegaConf.has_resolver("minus"):
        OmegaConf.register_new_resolver("minus", lambda x, y: x - y)

    if not OmegaConf.has_resolver("multiply"):
        OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)

    if not OmegaConf.has_resolver("divide"):
        OmegaConf.register_new_resolver("divide", lambda x, y: x / y)

    if not OmegaConf.has_resolver("listfill"):
        OmegaConf.register_new_resolver("listfill", lambda item, len: [item] * len)

    if not OmegaConf.has_resolver("hydra"):
        setup_globals()


def save_heatmap(archive, heatmap_path):
    """Saves a heatmap of the archive to the given path.

    Args:
        archive (GridArchive or CVTArchive): The archive to save.
        heatmap_path: Image path for the heatmap.
    """
    plt.figure(figsize=(8, 6))
    # grid_archive_heatmap(archive, vmin=0, vmax=10000, cmap='viridis')
    cvt_archive_heatmap(archive, lw=0.1, vmin=0, vmax=10000, cmap="viridis")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close(plt.gcf())


def exp_func(
    cfg: DictConfig,
    runtime_env: str,
    runtime_alg: str,
    runtime_dir: str,
    env_manager,
    ranges,
    starting_scheduler,
    starting_itr,
    hydra_cfg: DictConfig,
    trial_id: int,
):
    # Configure logging to each trial
    trial_outdir = os.path.join(
        runtime_dir,
        f"trial_{trial_id}",
    )

    trial_path = Path(trial_outdir)
    if not trial_path.is_dir():
        trial_path.mkdir()

    log_config = copy.deepcopy(hydra_cfg.job_logging)
    with read_write(log_config):
        log_config["handlers"]["file"]["filename"] = f"{trial_outdir}/trial_log.log"
    configure_log(log_config)

    logger.info(f"----- Alg: {runtime_alg}; Env: {runtime_env} -----")
    logger.info(f"Running trial {trial_id}...")

    solution_dim = cfg["env"]["solution_dim"]
    batch_size = cfg["batch_size"]
    trial_seed = cfg["seed"] + trial_id

    # Instantiate archive
    if starting_scheduler is None:
        archive = instantiate(
            cfg["alg"]["archive"],
            solution_dim=solution_dim,
            ranges=ranges,
            seed=trial_seed,
        )

        # Instantiate emitters according to alg
        emitters = []
        for e in cfg["alg"]["emitters"]:
            emitter_seeds = list(
                range(
                    trial_seed,
                    trial_seed + e["num"],
                )
            )

            extra_args = {k: e[k] for k in e.keys() - ["type", "num"]}
            extra_args["x0"] = np.zeros(solution_dim)

            bounds = {
                'sphere': [(-10.24, 10.24)] * solution_dim,
                'rastrigin': [(-10.24, 10.24)] * solution_dim,
                'arm': [(-np.pi, np.pi)] * solution_dim,
                'overcooked': None
            }

            emitters.extend(
                [
                    instantiate(
                        e["type"],
                        archive=archive,
                        batch_size=batch_size,
                        seed=s,
                        bounds=bounds[runtime_env],
                        **extra_args,
                    )
                    for s in emitter_seeds
                ]
            )

        # Create a scheduler
        # No need to randomize emitter query order because data queried from all emitters
        #   are added together.
        scheduler = Scheduler(archive, emitters)
    else:
        scheduler = copy.deepcopy(starting_scheduler)

    logger.info(f"Archive: {scheduler.archive}")
    logger.info(f"Emitters: {scheduler.emitters}")
    logger.info(f"Scheduler: {scheduler}")

    summary_filename = os.path.join(trial_outdir, f"summary.csv")
    if os.path.exists(summary_filename):
        os.remove(summary_filename)
    with open(summary_filename, "w") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(["Iteration", "MOQD-Score", "Coverage", "Maximum", "Average"])

    # For MOME and NSGA2, warmup archive with a random batch
    if runtime_alg in ["mome", "nsga2"] and scheduler.archive.empty:
        init_sols = np.random.default_rng(trial_seed).normal(
            loc=0, scale=0.5, size=(batch_size, solution_dim)
        )

        if runtime_env in ["overcooked"]:
            objs, measures = env_manager.evaluate(init_sols, trial_outdir=trial_outdir)
        else:
            objs, measures = env_manager.evaluate(init_sols)
        success_mask = np.all(~np.isnan(objs), axis=1)
        
        scheduler.archive.add(init_sols[success_mask], objs[success_mask], measures[success_mask])

    # The experiment main loop
    # Runs QD optimization for cfg["itrs"] iterations while saving heatmaps every
    #   cfg["draw_arch_freq"] iterations, and pickling the final archive.
    for itr in range(starting_itr, cfg["itrs"] + 1):
        logger.info(
            f"----------------------- Starting itr{itr} -----------------------"
        )

        sols = scheduler.ask()
        if runtime_env in ["overcooked"]:
            objs, measures = env_manager.evaluate(sols, trial_outdir=trial_outdir)
        else:
            objs, measures = env_manager.evaluate(sols)
        success_mask = np.all(~np.isnan(objs), axis=1)
        
        try:
            scheduler.tell(objs, measures, success_mask)
        except ValueError as e:
            logger.error(e)

        has_passive = runtime_alg in ["nsga2", "como_cma_es", "mo_cma_mae"]
        log_arch_itr = itr > 0 and itr % cfg["log_arch_freq"] == 0
        final_itr = itr == cfg["itrs"]
        # Save heatmap and log QD metrics every cfg["draw_arch_freq"] iterations, and on final iteration.
        if log_arch_itr or final_itr:
            # For NSGA2 and COMO-CMA-ES, update the passive archive before accessing.
            if has_passive:
                scheduler.archive.qd_update()

            save_heatmap(
                scheduler.archive,
                os.path.join(trial_outdir, f"heatmap_{itr:08d}.png"),
            )

            with open(summary_filename, "a") as summary_file:
                writer = csv.writer(summary_file)
                data = [
                    itr,
                    scheduler.archive.stats.qd_score,
                    scheduler.archive.stats.coverage,
                    scheduler.archive.stats.obj_max,
                    scheduler.archive.stats.obj_mean,
                ]
                writer.writerow(data)

                logger.info(f"QD Metrics: {data}")
            
            # if np.isclose(scheduler.archive.stats.coverage, 0):
            #     __import__("pdb").set_trace()

            # pickle.dump(
            #     scheduler.archive,
            #     open(os.path.join(trial_outdir, f"archive_{itr:08d}.pkl"), "wb")
            # )

            pickle.dump(
                scheduler,
                open(os.path.join(trial_outdir, f"scheduler_{itr:08d}.pkl"), "wb")
            )
	    
            # np.savetxt(os.path.join(trial_outdir, f"sols_{itr:08d}.txt"), sols, delimiter=',')
            # np.savetxt(os.path.join(trial_outdir, f"objs_{itr:08d}.txt"), objs, delimiter=',')
            # np.savetxt(os.path.join(trial_outdir, f"meas_{itr:08d}.txt"), measures, delimiter=',')


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    define_resolvers()

    runtime_env = hydra.core.hydra_config.HydraConfig.get().runtime.choices["env"]
    runtime_alg = hydra.core.hydra_config.HydraConfig.get().runtime.choices["alg"]
    runtime_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # client = Client(**cfg["dask"])

    # Fill ranges at runtime according to env and/or solution_dim, and instantiate env.
    solution_dim = cfg["env"]["solution_dim"]
    if runtime_env in ["sphere", "rastrigin"]:
        max_bound = solution_dim / 2 * 5.12
        ranges = [(-max_bound, max_bound), (-max_bound, max_bound)]
        env_manager = instantiate(cfg["env"])
    elif runtime_env in ["arm"]:
        ranges = [
            (-solution_dim, solution_dim),
            (-solution_dim, solution_dim),
        ]
        env_manager = instantiate(cfg["env"])
    elif runtime_env in ["overcooked"]:
        ranges = [[-5, 5], [-5, 5]]  # (diff_num_ingre_held, diff_num_plate_held)
        # ranges = [[0.4, 1], [0, 0.4]]  # (concurr_active, stuck_time)
        # Only overcooked env needs client for parallelizing evaluations, and random seed
        env_manager = instantiate(cfg["env"], client=client, seed=cfg["seed"])
    else:
        raise TypeError(f"Unrecognized env {runtime_env}.")

    if cfg["starting_scheduler_path"] is None:
        pickled_scheduler = None
    else:
        pickled_scheduler = pkl.load(open(file=cfg["starting_scheduler_path"], mode="rb"))

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    configured_exp_func = partial(
        exp_func,
        cfg,
        runtime_env,
        runtime_alg,
        runtime_dir,
        env_manager,
        ranges,
        pickled_scheduler,
        cfg["starting_itr"],
        hydra_cfg,
    )
    if runtime_env in ["sphere", "rastrigin", "arm"]:
        # Parallelize trials for simple envs
        trial_ids = list(range(cfg["num_trials"]))
        # futures = client.map(configured_exp_func, trial_ids)
        # results = client.gather(futures)
        for trial_id in range(cfg["num_trials"]):
            configured_exp_func(trial_id)
    elif runtime_env in ["overcooked"]:
        # Serialize trials and parallelize evaluations for overcooked
        for id in range(cfg["num_trials"]):
            configured_exp_func(id)
            client.restart()
    else:
        raise TypeError(f"Unrecognized env {runtime_env}.")


if __name__ == "__main__":
    main()  # pylint: disable = no-value-for-parameter
