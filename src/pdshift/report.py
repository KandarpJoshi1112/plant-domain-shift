"""Aggregate runs across seeds and compare experiments statistically.

Example:
    python -m pdshift.report --results results \
        --compare e1_custom_cnn/seed0 e2_resnet18/seed0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pdshift.metrics import mcnemar_test

logger = logging.getLogger("pdshift.report")


def summarise(results_dir: Path) -> pd.DataFrame:
    """Mean and standard deviation of key metrics per experiment (and few-shot k) across seeds."""
    rows = []
    for metrics_file in sorted(results_dir.glob("*/*/metrics.json")):
        m = json.loads(metrics_file.read_text(encoding="utf-8"))
        rows.append({
            "experiment": m["experiment"], "fewshot_k": m["fewshot_k"], "seed": m["seed"],
            "pv_acc": m["plantvillage_test"]["accuracy"], "pd_acc": m["plantdoc_test"]["accuracy"],
            "pd_macro_f1": m["plantdoc_test"]["macro_f1"], "gap": m["generalisation_gap_accuracy"],
            "pd_ece": m["plantdoc_test"]["ece"], "pd_ece_temp": m["plantdoc_test"]["ece_after_temperature"],
        })
    if not rows:
        raise FileNotFoundError(f"No metrics.json files under {results_dir}")
    df = pd.DataFrame(rows)
    summary = df.drop(columns="seed").groupby(["experiment", "fewshot_k"]).agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    return summary.reset_index()


def compare(results_dir: Path, run_a: str, run_b: str, domain: str = "plantdoc_test") -> dict[str, float]:
    """McNemar test between two runs evaluated on the same images."""
    a = np.load(results_dir / run_a / f"preds_{domain}.npz")
    b = np.load(results_dir / run_b / f"preds_{domain}.npz")
    if not np.array_equal(a["paths"], b["paths"]):
        raise ValueError("Runs were evaluated on different images; McNemar is not valid")
    return mcnemar_test(a["labels"], a["logits"].argmax(1), b["logits"].argmax(1))


def plot_summary(summary: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of lab vs field accuracy per experiment (mean ± std across seeds)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{e}\n(k={k})" if k else e for e, k in zip(summary["experiment"], summary["fewshot_k"])]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(labels)), 4))
    ax.bar(x - 0.2, summary["pv_acc_mean"], 0.4, yerr=summary["pv_acc_std"].fillna(0), label="PlantVillage test (lab)")
    ax.bar(x + 0.2, summary["pd_acc_mean"], 0.4, yerr=summary["pd_acc_std"].fillna(0), label="PlantDoc test (field)")
    for xi, (pv, pdv) in enumerate(zip(summary["pv_acc_mean"], summary["pd_acc_mean"])):
        ax.text(xi - 0.2, pv + 0.02, f"{pv:.1%}", ha="center", fontsize=8)
        ax.text(xi + 0.2, pdv + 0.02, f"{pdv:.1%}", ha="center", fontsize=8)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results_dir = Path(args.results)
    summary = summarise(results_dir)
    out = results_dir / "summary.csv"
    summary.to_csv(out, index=False)
    logger.info("Wrote %s\n%s", out, summary.round(4).to_string(index=False))
    plot_summary(summary, results_dir / "lab_vs_field_accuracy.png")
    if args.compare:
        res = compare(results_dir, *args.compare)
        logger.info("McNemar %s vs %s on PlantDoc: %s", *args.compare, res)


if __name__ == "__main__":
    main()
