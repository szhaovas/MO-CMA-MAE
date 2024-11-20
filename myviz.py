import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import torch
from src.envs.overcooked.gan import OvercookedGenerator
from src.envs.overcooked.milp_repair import RepairModule
from src.moribs.archives._pf_utils import cvt_archive_heatmap
from src.envs.overcooked.overcooked_manager import run, generator_config, repair_config


rollout_seed = 42


def show_rollout(sol, seed):
    generator = (
        OvercookedGenerator(**generator_config)
        .load_from_saved_weights()
        .to("cuda" if torch.cuda.is_available() else "cpu")
        .eval()
    )
    unrepaired_lvl = generator.levels_from_latent(np.expand_dims(sol, axis=0))[0]

    repair_module = RepairModule(**repair_config)

    run(
        level=unrepaired_lvl,
        n_evals=1,
        seed=seed,
        repair_module=repair_module,
        hydra_cfg=None,
        trial_outdir=os.getcwd(),
        lvl_id=0,
        render=True,
    )


def show_arm_links(sol):
    plt.figure(figsize=(6, 6))

    link_lengths = np.ones(sol.size)
    lim = 1.05 * np.sum(link_lengths)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)

    # Plot each link and joint
    pos = np.array([0, 0])
    cum_thetas = np.cumsum(sol)

    for link_length, cum_theta in zip(link_lengths, cum_thetas):
        next_pos = pos + link_length * np.array([np.cos(cum_theta), np.sin(cum_theta)])
        plt.plot([pos[0], next_pos[0]], [pos[1], next_pos[1]], "-ko", ms=3)
        pos = next_pos

    plt.plot(0, 0, "ro", ms=6)
    plt.plot(pos[0], pos[1], "go", ms=6)

    plt.show()


def show_interactive_archive(archive, objective_dim=2):
    fig = plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(
        archive, lw=0.1, vmin=0, vmax=100**objective_dim, cmap="viridis"
    )
    plt.tight_layout()

    def onclick(event):
        occupied, data = archive.retrieve_single([event.xdata, event.ydata])
        pf = data["pf"]

        if occupied:
            print("Objective values within this PF:")
            print(pf.objectives)
            print("Measures within this PF:")
            print(f"{pf.measures}\n")
            show_interactive_pf(archive, event.xdata, event.ydata, plot_tpf=True)

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()


def show_interactive_pf(
    archive, x, y, plot_tpf=False, datapoint_freq=1, show_discount=False
):
    _, data = archive.retrieve_single([x, y])
    pf = data["pf"]
    fig = plt.figure(figsize=(8, 6))
    objs = np.array(sorted(pf.objectives, key=lambda objs: objs[0]))
    plt.step(objs[:, 0], objs[:, 1], "b-", where="pre")
    print(f"Passive archive numvisits: {pf.numvisits}")

    if plot_tpf:
        _, t_data = archive.main.retrieve_single([x, y])
        t_pf = t_data["pf"]
        t_objs = np.array(sorted(t_pf.objectives, key=lambda objs: objs[0]))
        plt.step(t_objs[:, 0], t_objs[:, 1], "b-", alpha=0.2, where="pre")
        plt.scatter(t_objs[::datapoint_freq, 0], t_objs[::datapoint_freq, 1], c="blue")
        if show_discount:
            for i, (x, y) in enumerate(zip(t_objs[:, 0], t_objs[:, 1])):
                if i % datapoint_freq == 0:
                    plt.text(x, y, f"$d_{{{i}}}={t_pf.discount_factors[i]}$")
        print(f"Main archive numvisits: {t_pf.numvisits}")

    def onclick(event):
        # retrieves the solution with the most similar objectives as the clicked position
        min_distance = np.inf
        idx2show = None

        all_objs = pf.objectives
        all_meas = pf.measures
        all_sols = pf.solutions
        all_discounts = pf._discount_factors

        if plot_tpf:
            all_objs += t_pf.objectives
            all_meas += t_pf.measures
            all_sols += t_pf.solutions
            all_discounts += t_pf._discount_factors

        for i, objs in enumerate(all_objs):
            distance = (event.xdata - objs[0]) ** 2 + (event.ydata - objs[1]) ** 2
            if distance < min_distance:
                min_distance = distance
                idx2show = i

        print("Showing solution with")
        print(f"\t objective values {all_objs[idx2show]}")
        print(f"\t measures {all_meas[idx2show]}")
        print(f"\t discount {all_discounts[idx2show]}")

        # show_rollout(all_sols[idx2show], rollout_seed)
        show_arm_links(all_sols[idx2show])

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()


if __name__ == "__main__":
    with open(
        file="/home/shihanzh/Desktop/moqd_dev-main/multirun/2024-11-19/13-52-49/0/trial_0/scheduler_00001200.pkl",
        mode="rb",
    ) as f:
        archive = pkl.load(f).archive
        show_interactive_archive(archive)
