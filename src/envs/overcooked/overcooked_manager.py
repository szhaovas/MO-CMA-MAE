"""Provides OvercookedManager."""

import logging
import random
import time
from typing import Tuple, List

import numpy as np
import torch
from dask.distributed import Client

from .gan import OvercookedGenerator
from .milp_repair import RepairModule
from .level import OvercookedLevel
from .agents.agent import MediumQMdpPlanningAgent, GreedyHumanModel
from .planning.planners import MediumLevelPlanner, HumanSubtaskQMDPPlanner
from .mdp.overcooked_env import OvercookedEnv
from .mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.search import NotConnectedError

import copy
from pathlib import Path
from omegaconf import read_write
import hydra
from hydra.core.utils import configure_log

logger = logging.getLogger(__name__)

MIN_SCORE = 0
# max sparse reward = 3 * 20
# max reward = (60 + 1) * 100
MAX_SCORE = int(61e6)

mlp_params = {
    "start_orientations": False,
    "wait_allowed": False,
    "counter_goals": [],
    "counter_drop": [],
    "counter_pickup": [],
    "same_motion_goals": True,
}

grid_world_config = {
    "start_order_list": ["onion", "onion", "onion"],
    "cook_time": 10,
    "num_items_for_soup": 3,
    "delivery_reward": 20,
    "rew_shaping_params": None,
}

generator_config = {
    "i_size": 16,
    "nz": 32,
    "nc": 8,
    "ngf": 64,
    "n_extra_layers": 0,
    "lvl_width": 15,
    "lvl_height": 10,
    "model_file": "generator_15x10.pth",
}

repair_config = {
    "cost_type": "flow",
    "use_cont": False,
    "discard_suboptimal": False,
    "time_limit": 1500,
}


class OvercookedManager:
    """Manager for the overcooked environments.

    Args:
        client: Dask client for distributed compute.
        n_evals: Number of times to evaluate each solution during real evaluation.
        generator_config: Configurations for the GAN used to generate overcooked levels.
        repair_config: Configurations for the CPLEX repair module used to repair unfeasible
            levels generated by GAN.
        seed: Workers will use this master seed during evaluation.
    """

    def __init__(
        self,
        client: Client,
        solution_dim: int,
        n_evals: int,
        seed: int = None,
    ):
        self.client = client
        self.n_evals = n_evals

        self.generator = (
            OvercookedGenerator(**generator_config)
            .load_from_saved_weights()
            .to("cpu")
            .eval()
        )

        self.repair_module = RepairModule(**repair_config)

        self.rng = np.random.default_rng(seed)

        self.hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    def evaluate(
        self, sols: np.ndarray, trial_outdir: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Create agents from solutions to pass them for evaluation
        unrepaired_lvls = self.generator.levels_from_latent(sols)

        # Make each solution evaluation have a different seed. Note that we
        # assign seeds to solutions rather than workers, which means that we
        # are agnostic to worker configuration.
        # FIXME: Why have different seeds for each worker? Shouldn't this be
        # different seeds for each eval?
        evaluation_seeds = self.rng.integers(
            np.iinfo(np.int32).max / 2, size=len(sols), endpoint=True
        )

        futures = [
            self.client.submit(
                run,
                lvl,
                self.n_evals,
                seed,
                self.repair_module,
                self.hydra_cfg,
                trial_outdir,
                lvl_id,
                pure=False,
            )
            for lvl_id, (lvl, seed) in enumerate(zip(unrepaired_lvls, evaluation_seeds))
        ]
        results = self.client.gather(futures)
        print(results)

        objs, meas = [], []
        for r in results:
            objs.append(r[0])
            meas.append([r[1], r[2]])

        return np.array(objs), np.array(meas)


def run(
    level: np.ndarray,
    n_evals: int,
    seed: int,
    repair_module,
    hydra_cfg,
    trial_outdir,
    lvl_id,
    render: bool = False,
):
    """Evaluates a generated level map by first repairing it and then running
    games in it for n_evals times.

    Args:
        level (np.ndarray): Unrepaired level generated by GAN from a solution within
            latent space.
        n_evals (int): The number of games the level map should be run on.
        seed (int): _description_
        render (bool, optional): Renders the game play if true. Defaults to False.

    Returns:

    """
    # Configure logging to each trial
    if not hydra_cfg is None:
        trial_path = Path(trial_outdir)
        if not trial_path.is_dir():
            trial_path.mkdir()

        log_config = copy.deepcopy(hydra_cfg.job_logging)
        with read_write(log_config):
            log_config["handlers"]["file"]["filename"] = f"{trial_outdir}/trial_log.log"
        configure_log(log_config)

    # start = time.time()

    logger.info(
        f"----------------------- Evaluating {lvl_id}th level -----------------------"
    )

    # logger.info("seeding global randomness")
    np.random.seed(seed // np.int32(4))
    random.seed(seed // np.int32(2))

    # logger.info("repairing level with MIP")
    repaired_lvl = repair_module.repair_lvl(level, seed=seed)
    if repaired_lvl is None:
        logger.warning(f"repairing level failed")
        logger.info(f"The unrepaired level was {level}")

        return np.full((2,), np.nan), np.nan, np.nan

    try:
        grid = OvercookedLevel(repaired_lvl).to_str_grid()
        mdp = OvercookedGridworld.from_grid(grid, grid_world_config)
        env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=100)

        # logger.info("Pre-constructing graph...")
        mlp_planner = MediumLevelPlanner(mdp, mlp_params)

        # logger.info("Planning...")
        agent2 = GreedyHumanModel(mlp_planner, auto_unstuck=True)

        # logger.info("Pre-constructing qmdp plan...")
        qmdp_planner = HumanSubtaskQMDPPlanner.from_pickle_or_compute(
            mdp, mlp_params, force_compute_all=True
        )

        # logger.info("QMDP x planning...")
        agent1 = MediumQMdpPlanningAgent(qmdp_planner, greedy=False, auto_unstuck=True)

        agent1.set_mdp(mdp)
        agent2.set_mdp(mdp)

        agent1.set_agent_index(0)
        agent2.set_agent_index(1)

        # logger.info("Preprocess took %f seconds", time.time() - start)

        fitnesses = []
        mea1s, mea2s = [], []
        np.random.seed(seed)

        aug_level = np.zeros((2, *repaired_lvl.shape))
        for i in range(n_evals):
            done = False
            total_sparse_reward = 0
            last_state = None
            timestep = 0

            # Saves when each soup (order) was delivered
            checkpoints = [env.horizon - 1] * env.num_orders
            cur_order = 0

            if render:
                env.render()
                time.sleep(0.1)

            while not done:
                (pos1x, pos1y), (pos2x, pos2y) = env.state.player_positions
                aug_level[0][pos1y, pos1x] += 1
                aug_level[1][pos2y, pos2x] += 1
                joint_action = (
                    agent1.action(env.state)[0],
                    agent2.action(env.state)[0],
                )

                next_state, timestep_sparse_reward, done, info = env.step(joint_action)
                total_sparse_reward += timestep_sparse_reward

                if timestep_sparse_reward > 0:
                    checkpoints[cur_order] = timestep
                    cur_order += 1

                last_state = next_state
                timestep += 1

                if render:
                    env.render()
                    time.sleep(0.1)

            workloads = last_state.get_player_workload()
            mea1 = workloads[0]["num_ingre_held"] - workloads[1]["num_ingre_held"]
            mea2 = workloads[0]["num_plate_held"] - workloads[1]["num_plate_held"]
            # mea1 = last_state.cal_concurrent_active_sum()
            # mea2 = last_state.cal_total_stuck_time()

            # Smooth fitness is the total reward tie-broken by soup delivery
            # times.
            # Later soup deliveries are higher priority.
            fitness = total_sparse_reward + 1
            for timestep in reversed(checkpoints):
                fitness *= env.horizon
                fitness -= timestep

            fitnesses.append(fitness / MAX_SCORE)
            mea1s.append(mea1)
            mea2s.append(mea2)
            logger.info(
                f"Finished {i}th eval; Fitness: {fitness}; Measures: {[mea1, mea2]}"
            )

            env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=100)

        # logger.info("run_overcooked done after %f sec", time.time() - start)

        # Overcooked evaluation returns 2 objectives, both mapped to [0, 100]
        #   - the agents' game score (the lower the better)
        #   - the number of wall cells (the fewer the better)
        obj1 = (1 - np.median(fitnesses, axis=0)) * 100
        if np.any((obj1 > 100) | (obj1 < 0)):
            logger.warning(
                f"Fitness ouside of the [0, 100] range: {fitnesses}"
                "The evaluated level is:"
                f"{level}"
                f"Seed: {seed}"
            )

        logger.info(
            f"Proportion of wall cells: {np.sum(repaired_lvl == 2) / repaired_lvl.size}"
        )
        obj2 = (1 - np.sum(repaired_lvl == 2) / repaired_lvl.size) * 100
        return (
            np.array([obj1, obj2]),
            np.mean(mea1s, axis=0),
            np.mean(mea2s, axis=0),
        )
    except (TimeoutError, NotConnectedError) as e:
        logger.warning(f"evaluate failed")
        logger.info(f"The repaired level was {repaired_lvl}")

        return np.full((2,), np.nan), np.nan, np.nan
