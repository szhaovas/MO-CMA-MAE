import re
import csv
import glob
from os import path

total_summary_filename = "total_summary.csv"
skip_len = 1

# name_mapping = {
#     'me_moo_logs': 'MAP-Elites',
#     'mome_logs': 'MOME',
#     'nsga2_logs': 'NSGA-II',
#     'spea2_logs': 'SPEA2',
#     'old_tbd_logs': 'TBD',
#     'tbd_logs': 'TBD(mu+mu)'
# }

# algo_order = [
#     'TBD(mu+mu)',
#     'TBD',
#     'MOME',
#     'MAP-Elites',
#     'NSGA-II',
#     'SPEA2'
# ]

name_mapping = {
    "0.001": "alpha=0.001",
    "0.01": "alpha=0.01",
    "0.1": "alpha=0.1",
    "1": "alpha=1",
    "unlimited": "unlimited"
}

algo_order = ["unlimited", "alpha=0.001", "alpha=0.01", "alpha=0.1", "alpha=1"]


def order_func(datum):
    return algo_order.index(datum[0])


all_data = []
for p in glob.glob("*/summary.csv"):
    algo_name, _ = path.split(p)

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