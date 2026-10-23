#!/usr/bin/env python3
"""Plot the degree distribution produced by the Barabási–Albert run.

The C++ model writes one CSV per checkpoint into ``results/`` named
``degree_distribution_step_<n>.csv`` with columns ``degree,count,pk``.
This script loads the latest (largest-step) file by default and plots the
degree distribution on log--log axes together with a reference k^{-3} line,
which is the theoretical BA exponent for the m = 1 model.

Usage:
    python plot_degree_distribution.py [results_dir] [--step N] [--save out.png]
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt


def find_csv(results_dir: str, step: int | None) -> str:
    files = glob.glob(os.path.join(results_dir, "degree_distribution_step_*.csv"))
    if not files:
        sys.exit(f"No degree_distribution_step_*.csv files found in '{results_dir}'. "
                 f"Run the parallel_walking_graph executable first.")

    def step_of(path: str) -> int:
        m = re.search(r"step_(\d+)\.csv$", path)
        return int(m.group(1)) if m else -1

    if step is not None:
        for f in files:
            if step_of(f) == step:
                return f
        sys.exit(f"Step {step} not found. Available: "
                 f"{sorted(step_of(f) for f in files)}")
    return max(files, key=step_of)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", nargs="?", default="results",
                    help="directory containing the CSV files (default: results)")
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step to plot (default: latest)")
    ap.add_argument("--save", default=None,
                    help="save the figure to this path instead of showing it")
    args = ap.parse_args()

    path = find_csv(args.results_dir, args.step)
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    degree = data[:, 0]
    count = data[:, 1]
    pk = data[:, 2]
    n_nodes = int(count.sum())

    # Keep degree >= 1 and non-zero probability for the log--log plot.
    mask = (degree >= 1) & (pk > 0)
    degree, pk = degree[mask], pk[mask]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.loglog(degree, pk, "o", ms=5, color="#1f77b4",
              label=f"BA simulation (N = {n_nodes})")

    # Reference power law P(k) ~ k^{-3}, normalised to the first data point.
    k_ref = np.logspace(np.log10(degree.min()), np.log10(degree.max()), 100)
    c = pk[0] * degree[0] ** 3.0
    ax.loglog(k_ref, c * k_ref ** -3.0, "--", color="crimson",
              label=r"reference $P(k)\propto k^{-3}$")

    ax.set_xlabel("degree $k$")
    ax.set_ylabel("$P(k)$")
    ax.set_title(f"Barabási–Albert degree distribution\n({os.path.basename(path)})")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
