"""Run the reduced ICBTA 2026 conference experiment matrix.

Default final grid = 70 runs:
- S0 benign, Dir(0.5): C0,C1,C3,C4 x 5 seeds = 20
- S1 20% label-flip, Dir(0.5): C0..C4 x 5 seeds = 25
- S1 20% label-flip, Dir(0.1): C0..C4 x 5 seeds = 25

The script is resume-safe: a run is skipped only when all three expected output
files exist (round_metrics.csv, client_metrics.csv, metadata.json).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
SCRIPT = Path(__file__).with_name("run_conference_experiment.py")


def is_complete(run_dir: Path) -> bool:
    return all((run_dir / f).exists() for f in ["round_metrics.csv", "client_metrics.csv", "metadata.json"])


def build_grid(profile: str):
    seeds = [0, 1, 2, 3, 4]
    jobs = []

    if profile == "smoke":
        for method in ["C0", "C4"]:
            jobs.append(dict(scenario="S0", alpha=0.5, method=method, seed=0, rounds=10))
        return jobs

    if profile == "pilot":
        for scenario, alpha, methods in [
            ("S0", 0.5, ["C0", "C1", "C3", "C4"]),
            ("S1", 0.5, ["C0", "C1", "C2", "C3", "C4"]),
        ]:
            for method in methods:
                for seed in [0, 1]:
                    jobs.append(dict(scenario=scenario, alpha=alpha, method=method, seed=seed, rounds=50))
        return jobs

    # final: 70 runs, 100 rounds each
    for method in ["C0", "C1", "C3", "C4"]:
        for seed in seeds:
            jobs.append(dict(scenario="S0", alpha=0.5, method=method, seed=seed, rounds=100))

    for alpha in [0.5, 0.1]:
        for method in ["C0", "C1", "C2", "C3", "C4"]:
            for seed in seeds:
                jobs.append(dict(scenario="S1", alpha=alpha, method=method, seed=seed, rounds=100))

    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["smoke", "pilot", "final"], default="final")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--output_root", default="outputs_conference")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--force", action="store_true", help="rerun completed jobs")
    args = ap.parse_args()

    jobs = build_grid(args.profile)
    print(f"Profile={args.profile}; total jobs={len(jobs)}")

    for idx, job in enumerate(jobs, start=1):
        partition = f"dir{job['alpha']:g}"
        run_dir = Path(args.output_root) / job["scenario"] / partition / job["method"] / f"seed{job['seed']}"
        if is_complete(run_dir) and not args.force:
            print(f"[{idx}/{len(jobs)}] SKIP complete: {run_dir}")
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            PYTHON,
            str(SCRIPT),
            "--method", job["method"],
            "--scenario", job["scenario"],
            "--seed", str(job["seed"]),
            "--dir_alpha", str(job["alpha"]),
            "--attack_frac", "0.0" if job["scenario"] == "S0" else "0.20",
            "--rounds", str(job["rounds"]),
            "--device", args.device,
            "--data_root", args.data_root,
            "--out_dir", str(run_dir),
        ]
        print(f"[{idx}/{len(jobs)}] RUN: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    print("Grid complete.")


if __name__ == "__main__":
    main()
