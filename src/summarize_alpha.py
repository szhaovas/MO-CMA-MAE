import re
import csv
import glob
from os import path

total_summary_filename = "total_summary.csv"
skip_len = 1

name_mapping = {
    'mo_cma_mae': 'MO-CMA-MAE',
    'mome': 'MOME',
    'nsga2': 'NSGA-II',
    'como_cma_es': 'COMO-CMA-ES',
}

# name_mapping = {
#     "alpha=0.001": "alpha=0.001",
#     "alpha=0.01": "alpha=0.01",
#     "alpha=0.1": "alpha=0.1",
#     "alpha=1": "alpha=1",
#     "dynamic_archive": "dynamic archive",
#     "single_discount": "single discount"
# }

# algo_order = ["alpha=0.001", "alpha=0.01", "alpha=0.1", "alpha=1", "dynamic archive", "single discount"]
algo_order = ["MO-CMA-MAE", "MOME", "NSGA-II", "COMO-CMA-ES"]


def order_func(datum):
    return algo_order.index(datum[0])


all_data = []
for p in glob.glob("*/*/trial*/summary.csv"):
    head, _ = path.split(p)
    head, _ = path.split(head)
    algo_name, _ = path.split(head)

    if algo_name not in name_mapping:
        continue

    algo_name = name_mapping[algo_name]

    with open(p) as summary_file:
        all_lines = list(csv.reader(summary_file))
        for cur_line in all_lines[skip_len::skip_len]:
            datum = [algo_name] + cur_line
            all_data.append(datum)

# Sort the data by the names in the given order.
all_data.sort(key=order_func)
all_data.insert(
    0,
    [
        "Algorithm",
        "Iteration",
        "QD-Score",
        "Coverage",
        "Best Solution",
        "Average",
    ],
)

# Output the summary of summary files.
with open(total_summary_filename, "w") as summary_file:
    writer = csv.writer(summary_file)
    for datum in all_data:
        writer.writerow(datum)