import glob
from os import path, chdir
import pickle as pkl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

have_legend = True
exp_logdir = "/home/shihanzh/Desktop/moqd_dev-main/multirun/2024-11-19/19-05-07"

lst = []
for dirname in glob.glob(f"{exp_logdir}/*/trial*"):
    head, _ = path.split(dirname)
    _, alpha = path.split(head)

    chdir(dirname)

    for scheduler_name in sorted(glob.glob("*.pkl")):
        itr = int(scheduler_name[-12:-4])

        with open(scheduler_name, mode="rb") as f:
            # Count numsols for main instead of passive archive
            archive = pkl.load(f).archive.main
            numsols = 0

            for cell in archive:
                numsols += len(cell["pf"])

            avg_numsols = numsols / archive.cells

            lst.append([alpha, itr, avg_numsols])

df = pd.DataFrame(lst, columns=["Alpha", "Iteration", "Avg. Numsols"])

sns.set(font_scale=2.4)
with sns.axes_style("white"):
    sns.set_style("white", {"font.family": "serif", "font.serif": "Palatino"})

    p1 = sns.lineplot(
        data=df, x="Iteration", y="Avg. Numsols", hue="Alpha", legend=have_legend
    )

    for line in p1.get_lines():
        line.set_linewidth(4.0)

    plt.xticks([0, 5000])
    # plt.yticks([0, 5000])

    if have_legend:
        legend = plt.legend(loc="lower right", frameon=False, prop={"size": 20})
        for line in legend.get_lines():
            line.set_linewidth(4.0)

        frame = legend.get_frame()
        frame.set_facecolor("white")

    plt.tight_layout()

    chdir(exp_logdir)

    p1.figure.savefig("avg_numsols.pdf", bbox_inches="tight", dpi=100)
