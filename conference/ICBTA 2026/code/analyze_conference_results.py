"""Aggregate ICBTA 2026 experiment outputs into publication-ready tables/figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import ks_2samp, ttest_rel, wilcoxon
except Exception:
    ks_2samp = None
    ttest_rel = None
    wilcoxon = None

METHOD_LABELS = {
    "C0": "FedAvg",
    "C1": "Trimmed Mean",
    "C2": "Static Reputation",
    "C3": "Dynamic Trust",
    "C4": "Proposed Trust+Screening",
}


def discover_runs(root: Path):
    return sorted(root.glob("**/round_metrics.csv"))


def load_all(root: Path):
    round_frames = []
    client_frames = []
    for round_file in discover_runs(root):
        run_dir = round_file.parent
        client_file = run_dir / "client_metrics.csv"
        if not client_file.exists():
            continue
        round_frames.append(pd.read_csv(round_file))
        client_frames.append(pd.read_csv(client_file))
    if not round_frames:
        raise RuntimeError(f"No completed runs found under {root}")
    return pd.concat(round_frames, ignore_index=True), pd.concat(client_frames, ignore_index=True)


def last_non_nan(series: pd.Series) -> float:
    x = series.dropna()
    return float(x.iloc[-1]) if len(x) else np.nan


def trapezoid_auc(rounds: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(values)
    x = rounds[mask]
    y = values[mask]
    if len(x) < 2:
        return np.nan
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def build_run_summary(round_df: pd.DataFrame, client_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "method", "scenario", "partition"]
    rows = []
    for key_vals, g in round_df.groupby(keys):
        key = dict(zip(keys, key_vals))
        g = g.sort_values("round")
        cg = client_df
        for k, v in key.items():
            cg = cg[cg[k] == v]

        final_test = last_non_nan(g["test_acc"])
        aulc = trapezoid_auc(g["round"].to_numpy(), g["test_acc"].to_numpy())
        selected_cg = cg[cg["selected"] == 1]
        adv_accept = selected_cg.loc[selected_cg["is_adversarial"] == 1, "accepted"].mean() if (selected_cg["is_adversarial"] == 1).any() else np.nan
        benign_accept = selected_cg.loc[selected_cg["is_adversarial"] == 0, "accepted"].mean()

        # Final-round trust separation over the full client population (not only selected clients).
        final_round = int(cg["round"].max()) if len(cg) else 0
        cf = cg[cg["round"] == final_round]
        adv_trust = cf.loc[cf["is_adversarial"] == 1, "trust"].dropna().to_numpy()
        ben_trust = cf.loc[cf["is_adversarial"] == 0, "trust"].dropna().to_numpy()
        ks = np.nan
        if ks_2samp is not None and len(adv_trust) > 0 and len(ben_trust) > 0:
            ks = float(ks_2samp(ben_trust, adv_trust, alternative="two-sided").statistic)

        rows.append({
            **key,
            "final_test_acc": final_test,
            "aulc": aulc,
            "malicious_accept_rate": adv_accept,
            "benign_accept_rate": benign_accept,
            "final_trust_ks": ks,
            "mean_round_runtime_sec": g["round_runtime_sec"].mean(),
        })
    return pd.DataFrame(rows)


def aggregate_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "final_test_acc", "aulc", "malicious_accept_rate",
        "benign_accept_rate", "final_trust_ks", "mean_round_runtime_sec"
    ]
    grouped = run_summary.groupby(["scenario", "partition", "method"])[metrics].agg(["mean", "std", "count"])
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
    return grouped.reset_index()


def paired_tests(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Paired seed-wise comparisons of C4 against available baselines."""
    rows = []
    for (scenario, partition), gp in run_summary.groupby(["scenario", "partition"]):
        proposed = gp[gp["method"] == "C4"]
        if proposed.empty:
            continue
        for baseline in ["C0", "C1", "C2", "C3"]:
            base = gp[gp["method"] == baseline]
            if base.empty:
                continue
            merged = proposed.merge(base, on=["seed", "scenario", "partition"], suffixes=("_C4", f"_{baseline}"))
            for metric in ["final_test_acc", "aulc"]:
                a = merged[f"{metric}_C4"].to_numpy(dtype=float)
                b = merged[f"{metric}_{baseline}"].to_numpy(dtype=float)
                mask = np.isfinite(a) & np.isfinite(b)
                a, b = a[mask], b[mask]
                if len(a) == 0:
                    continue
                diff = a - b
                t_p = np.nan
                w_p = np.nan
                if ttest_rel is not None and len(a) >= 2:
                    try:
                        t_p = float(ttest_rel(a, b, nan_policy="omit").pvalue)
                    except Exception:
                        pass
                if wilcoxon is not None and len(a) >= 2 and not np.allclose(diff, 0):
                    try:
                        w_p = float(wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue)
                    except Exception:
                        pass
                rows.append({
                    "scenario": scenario,
                    "partition": partition,
                    "comparison": f"C4_vs_{baseline}",
                    "metric": metric,
                    "n_paired_seeds": int(len(a)),
                    "mean_C4": float(np.mean(a)),
                    "mean_baseline": float(np.mean(b)),
                    "mean_paired_difference": float(np.mean(diff)),
                    "sd_paired_difference": float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan,
                    "paired_t_p": t_p,
                    "wilcoxon_p": w_p,
                })
    return pd.DataFrame(rows)


def plot_accuracy(round_df: pd.DataFrame, out_dir: Path, scenario: str, partition: str) -> None:
    sub = round_df[(round_df["scenario"] == scenario) & (round_df["partition"] == partition)].copy()
    sub = sub.dropna(subset=["test_acc"])
    if sub.empty:
        return
    agg = sub.groupby(["method", "round"])["test_acc"].agg(["mean", "std"]).reset_index()

    plt.figure(figsize=(7.2, 4.8))
    for method in ["C0", "C1", "C2", "C3", "C4"]:
        a = agg[agg["method"] == method]
        if a.empty:
            continue
        plt.plot(a["round"], a["mean"], label=METHOD_LABELS[method])
        std = a["std"].fillna(0.0)
        plt.fill_between(a["round"], a["mean"] - std, a["mean"] + std, alpha=0.12)
    plt.xlabel("Communication round")
    plt.ylabel("Test accuracy")
    plt.title(f"{scenario}, {partition}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"accuracy_{scenario}_{partition}.png", dpi=300)
    plt.savefig(out_dir / f"accuracy_{scenario}_{partition}.pdf")
    plt.close()


def plot_trust(client_df: pd.DataFrame, out_dir: Path, partition: str) -> None:
    sub = client_df[
        (client_df["scenario"] == "S1") &
        (client_df["partition"] == partition) &
        (client_df["method"] == "C4")
    ].copy()
    if sub.empty:
        return
    agg = sub.groupby(["round", "is_adversarial"])["trust"].agg(["mean", "std"]).reset_index()

    plt.figure(figsize=(7.2, 4.8))
    for flag, label in [(0, "Benign clients"), (1, "Label-flip attackers")]:
        a = agg[agg["is_adversarial"] == flag]
        if a.empty:
            continue
        plt.plot(a["round"], a["mean"], label=label)
        std = a["std"].fillna(0.0)
        plt.fill_between(a["round"], a["mean"] - std, a["mean"] + std, alpha=0.12)
    plt.xlabel("Communication round")
    plt.ylabel("Trust score")
    plt.title(f"Trust trajectories under poisoning ({partition})")
    plt.legend()
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"trust_C4_S1_{partition}.png", dpi=300)
    plt.savefig(out_dir / f"trust_C4_S1_{partition}.pdf")
    plt.close()


def plot_acceptance(client_df: pd.DataFrame, out_dir: Path, partition: str) -> None:
    sub = client_df[
        (client_df["scenario"] == "S1") &
        (client_df["partition"] == partition) &
        (client_df["method"] == "C4") &
        (client_df["selected"] == 1)
    ].copy()
    if sub.empty:
        return
    agg = sub.groupby(["round", "is_adversarial"])["accepted"].mean().reset_index()

    plt.figure(figsize=(7.2, 4.8))
    for flag, label in [(0, "Benign updates"), (1, "Malicious updates")]:
        a = agg[agg["is_adversarial"] == flag]
        if a.empty:
            continue
        plt.plot(a["round"], a["accepted"], label=label)
    plt.xlabel("Communication round")
    plt.ylabel("Acceptance rate")
    plt.ylim(-0.02, 1.02)
    plt.title(f"Update acceptance under poisoning ({partition})")
    plt.legend()
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"acceptance_C4_S1_{partition}.png", dpi=300)
    plt.savefig(out_dir / f"acceptance_C4_S1_{partition}.pdf")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs_conference")
    ap.add_argument("--out", default="outputs_conference/analysis")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    round_df, client_df = load_all(root)
    run_summary = build_run_summary(round_df, client_df)
    aggregate = aggregate_summary(run_summary)
    tests = paired_tests(run_summary)

    run_summary.to_csv(out / "run_summary.csv", index=False)
    aggregate.to_csv(out / "aggregate_summary.csv", index=False)
    tests.to_csv(out / "paired_tests.csv", index=False)

    for scenario, partition in [("S0", "dir0.5"), ("S1", "dir0.5"), ("S1", "dir0.1")]:
        plot_accuracy(round_df, out, scenario, partition)
    for partition in ["dir0.5", "dir0.1"]:
        plot_trust(client_df, out, partition)
        plot_acceptance(client_df, out, partition)

    print(f"Loaded {run_summary.shape[0]} completed runs")
    print(f"Saved analysis to {out.resolve()}")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
