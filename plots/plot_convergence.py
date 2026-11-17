# Stage 2 plot: convergence curves (mean best-so-far error +- standard error over the seeds)


import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np

from plots.style import PROBLEM_TITLE, clean_axes, style_for

import os

# ---------------------------------SETTINGS------------------------------------------------------

# IMPORTANT: update the folder
RESULTS_DIR = "results/stage2"   # where stage2/runner.py saved the results
OUT_DIR = "plots/output"         # where the figures are saved
PROBLEMS = ["bbob", "real"]

LOWEST_ERROR = 1e-8 


# -----------------------------------READING THE RESULTS---------------------------

def load_error_curves(folder):
    # Read all seed_*.json files in a folder
    # Returns an array with one row per seed and one column per evaluation
    curves = []
    for path in sorted(folder.glob("seed_*.json")):
        with open(path) as f:
            result = json.load(f)
        curves.append(result["best_so_far_error"])
    return np.array(curves)


def mean_and_standard_error(curves):
    n_seeds = curves.shape[0]
    mean = curves.mean(axis=0)               # average over the seeds, for every evaluation
    if n_seeds > 1:
        standard_error = curves.std(axis=0, ddof=1) / np.sqrt(n_seeds)
    else:
        standard_error = np.zeros_like(mean)
    return mean, standard_error


# --------------------------ONE FIGURE PER PROBLEM----------------------------------------------------

def plot_problem(problem, summary_rows):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # every sub-folder of RESULTS_DIR is one algorithm
    for algorithm_folder in sorted(Path(RESULTS_DIR).iterdir()):
        folder = algorithm_folder / problem
        if not folder.is_dir():
            continue

        curves = load_error_curves(folder)
        if len(curves) == 0:
            continue
        curves = np.clip(curves, LOWEST_ERROR, None)

        mean, standard_error = mean_and_standard_error(curves)
        n_seeds = curves.shape[0]
        evaluations = np.arange(1, curves.shape[1] + 1)   # 1, 2, ..., budget

        style = style_for(algorithm_folder.name)
        label = style["label"] + " (n = " + str(n_seeds) + ")"
        ax.plot(evaluations, mean, color=style["color"], linestyle=style["linestyle"],
                linewidth=2, label=label)
        lower = np.clip(mean - standard_error, LOWEST_ERROR, None)
        upper = mean + standard_error
        ax.fill_between(evaluations, lower, upper, color=style["color"], alpha=0.15, linewidth=0)

        # remember the final errors for the summary table
        final_errors = curves[:, -1]
        if n_seeds > 1:
            final_se = final_errors.std(ddof=1) / np.sqrt(n_seeds)
        else:
            final_se = 0.0
        summary_rows.append({
            "problem": problem,
            "algorithm": algorithm_folder.name,
            "n_seeds": n_seeds,
            "final_error_mean": float(final_errors.mean()),
            "final_error_se": float(final_se),
            "final_error_median": float(np.median(final_errors)),
        })

    ax.set_yscale("log")
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Best-so-far error  f(x) - f*")
    ax.set_title(PROBLEM_TITLE.get(problem, problem) + ": mean ± standard error")
    clean_axes(ax)
    ax.legend(facecolor="white", edgecolor="none", framealpha=0.9)
    fig.tight_layout()

    for extension in ["png", "pdf"]:
        path = Path(OUT_DIR) / ("convergence_" + problem + "." + extension)
        fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved:", Path(OUT_DIR) / ("convergence_" + problem + ".png"))


# --------------------------------------MAIN------------------------------------------------------

if __name__ == "__main__":
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for problem in PROBLEMS:
        plot_problem(problem, summary_rows)

    # summary table of the final errors
    table_path = Path(OUT_DIR) / "summary_final_error.csv"
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print("saved:", table_path)

    for row in summary_rows:
        print(" ", row["problem"], row["algorithm"],
              "| final error =", round(row["final_error_mean"], 4),
              "±", round(row["final_error_se"], 4),
              "| median", round(row["final_error_median"], 4))
