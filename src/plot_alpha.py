import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

have_legend = False

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

data = pd.read_csv("total_summary.csv")

# y_label = "Best Solution"
y_label = "Coverage"

plt.figure(figsize=(12, 12))

palette = {
    "unlimited": "C1",
    "alpha=0.001": "C2",
    "alpha=0.01": "C3",
    "alpha=0.1": "C4",
    "alpha=1": "C5"
}

sns.set(font_scale=4)
with sns.axes_style("white"):
    sns.set_style("white", {"font.family": "serif", "font.serif": "Palatino"})
    sns.set_palette("colorblind")

    sns_plot = sns.lineplot(
        x="Iteration",
        y=y_label,
        linewidth=3.0,
        hue="Algorithm",
        data=data,
        legend=have_legend,
        palette=palette,
    )
    plt.xticks([0, 2500, 5000])
    # plt.yticks([3000, 6000, 9000], ["$3000$", "$6000$", "$9000$"])
    # plt.yticks([0.1, 0.5, 0.9], ["$0.1$", "$0.5$", "$0.9$"])
    plt.xlabel("Iterations")
    # plt.ylabel("Largest HV")
    plt.ylabel("Coverage")

    if have_legend:
        legend = plt.legend(loc="lower right", frameon=False, prop={"size": 45})
        for line in legend.get_lines():
            line.set_linewidth(4.0)

        frame = legend.get_frame()
        frame.set_facecolor("white")

    # sns_plot.figure.savefig("largest.pdf", bbox_inches="tight", dpi=100)
    sns_plot.figure.savefig("coverage.pdf", bbox_inches="tight", dpi=100)