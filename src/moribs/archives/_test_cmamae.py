import os
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ribs.archives import CVTArchive
from src.moribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from ribs.visualize import cvt_archive_heatmap


def calc_sphere(sol):
    
    dim = sol.shape[1]

    # Shift the Sphere function so that the optimal value is at x_i = 2.048.
    target_shift = 4

    # Normalize the objective to the range [0, 100] where 100 is optimal.
    best_obj = 0.0
    worst_obj = (-5.12 - target_shift)**2 * dim
    raw_obj = np.sum(np.square(sol - target_shift), axis=1)
    objs = (raw_obj - worst_obj) / (best_obj - worst_obj) * 100

    clipped = sol.copy()
    clip_indices = np.where(np.logical_or(clipped > 5.12, clipped < -5.12))
    clipped[clip_indices] = 5.12 / clipped[clip_indices]
    measures = np.concatenate(
        (
            np.sum(clipped[:, : dim // 2], axis=1, keepdims=True),
            np.sum(clipped[:, dim // 2 :], axis=1, keepdims=True),
        ),
        axis=1,
    )

    return objs**2, measures


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


if __name__ == "__main__":
    # Configure logging to each trial
    trial_outdir = os.path.join(
        os.getcwd(),
        f"trial_0",
    )

    trial_path = Path(trial_outdir)
    if not trial_path.is_dir():
        trial_path.mkdir()

    itrs = 5000
    solution_dim = 100
    cells = 10000
    batch_size = 36
    trial_seed = 42
    alpha = 0.16
    max_bound = solution_dim / 2 * 5.12
    ranges = [(-max_bound, max_bound), (-max_bound, max_bound)]
    
    archive = CVTArchive(
        solution_dim=solution_dim,
        cells=cells,
        ranges=ranges,
        learning_rate=alpha,
        threshold_min=0,
        seed=trial_seed,
        samples=50000
    )

    passive_archive = CVTArchive(
        solution_dim=solution_dim,
        cells=cells,
        ranges=ranges,
        seed=trial_seed,
        custom_centroids=np.copy(archive.centroids)
    )

    # Instantiate emitters according to alg
    emitter_seeds = list(
        range(
            trial_seed,
            trial_seed + 5,
        )
    )
    
    emitters = [
        EvolutionStrategyEmitter(
            x0=np.zeros(solution_dim),
            sigma0=0.5,
            ranker="imp",
            selection_rule="mu",
            restart_rule="basic",
            archive=archive,
            batch_size=batch_size,
            seed=s,
            bounds=[(-10.24, 10.24)] * solution_dim
        ) for s in emitter_seeds
    ]

    # Create a scheduler
    # No need to randomize emitter query order because data queried from all emitters
    #   are added together.
    scheduler = Scheduler(archive, emitters)

    summary_filename = os.path.join(trial_outdir, f"summary.csv")
    if os.path.exists(summary_filename):
        os.remove(summary_filename)
    with open(summary_filename, "w") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(["Iteration", "MOQD-Score", "Coverage", "Maximum", "Average"])

    # The experiment main loop
    # Runs QD optimization for cfg["itrs"] iterations while saving heatmaps every
    #   cfg["draw_arch_freq"] iterations, and pickling the final archive.
    for itr in range(itrs + 1):
        sols = scheduler.ask()
        objs, measures = calc_sphere(sols)
            
        scheduler.tell(objs, measures)
        passive_archive.add(sols, objs, measures)

        log_arch_itr = itr > 0 and itr % 10 == 0
        final_itr = itr == itrs
        # Save heatmap and log QD metrics every cfg["draw_arch_freq"] iterations, and on final iteration.
        if log_arch_itr or final_itr:
            save_heatmap(
                passive_archive,
                os.path.join(trial_outdir, f"heatmap_{itr:08d}.png"),
            )

            with open(summary_filename, "a") as summary_file:
                writer = csv.writer(summary_file)
                data = [
                    itr,
                    passive_archive.stats.qd_score,
                    passive_archive.stats.coverage,
                    passive_archive.stats.obj_max,
                    passive_archive.stats.obj_mean,
                ]
                writer.writerow(data)