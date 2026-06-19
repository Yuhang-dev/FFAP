from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def read_csv(path: str | Path, source: str) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row = dict(row)
            row["source_file"] = str(path)
            row["source"] = source
            rows.append(row)
    return rows


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        method = row.get("method", "")
        sparsity = as_float(row.get("target_sparsity")) or as_float(row.get("sparsity")) or 0.0
        ppl = as_float(row.get("ppl"))
        dense = method == "dense"
        out.append(
            {
                **row,
                "method": method,
                "sparsity_float": sparsity,
                "ppl_float": ppl,
                "active_jaccard_float": as_float(row.get("active_jaccard")),
                "decoded_activation_cosine_delta_float": as_float(
                    row.get("decoded_activation_cosine_delta")
                ),
                "reconstruction_mse_delta_float": as_float(
                    row.get("reconstruction_mse_delta")
                ),
                "firing_rate_l1_float": as_float(row.get("firing_rate_l1")),
                "is_dense": dense,
            }
        )
    return out


def write_combined(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "method",
        "target_sparsity",
        "sparsity",
        "ppl",
        "ppl_relative_increase",
        "active_jaccard",
        "decoded_activation_cosine_delta",
        "reconstruction_mse_delta",
        "firing_rate_l1",
        "feature_l0",
        "active_features_count",
        "source_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    method_labels = {
        "local_magnitude_unstructured": "Magnitude",
        "wanda_unstructured": "Wanda",
    }
    colors = {
        "local_magnitude_unstructured": "#c43c39",
        "wanda_unstructured": "#1f77b4",
    }
    methods = ["local_magnitude_unstructured", "wanda_unstructured"]
    dense_ppl = next((row["ppl_float"] for row in rows if row["is_dense"]), None)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for method in methods:
        method_rows = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: row["sparsity_float"],
        )
        x = [row["sparsity_float"] * 100 for row in method_rows]
        label = method_labels.get(method, method)
        color = colors.get(method)
        axes[0].plot(
            x,
            [row["ppl_float"] for row in method_rows],
            marker="o",
            label=label,
            color=color,
        )
        axes[1].plot(
            x,
            [row["active_jaccard_float"] for row in method_rows],
            marker="o",
            label=label,
            color=color,
        )
        axes[2].plot(
            x,
            [-row["decoded_activation_cosine_delta_float"] for row in method_rows],
            marker="o",
            label=label,
            color=color,
        )

    if dense_ppl is not None:
        axes[0].axhline(dense_ppl, color="#555555", linestyle="--", linewidth=1)
        axes[0].text(20, dense_ppl, "dense", va="bottom", fontsize=8)

    axes[0].set_title("WikiText-2 PPL")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Sparsity (%)")
    axes[0].set_ylabel("PPL (log scale)")
    axes[1].set_title("Active Feature Jaccard")
    axes[1].set_xlabel("Sparsity (%)")
    axes[1].set_ylabel("Jaccard vs dense")
    axes[2].set_title("Decoded Cosine Damage")
    axes[2].set_xlabel("Sparsity (%)")
    axes[2].set_ylabel("-delta cosine")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage 1 comparison figures")
    parser.add_argument("--magnitude-csv", default="results/stage1_magnitude_sweep.csv")
    parser.add_argument("--wanda-csv", default="results/stage1_wanda_sweep.csv")
    parser.add_argument("--out-csv", default="results/stage1_pruning_comparison.csv")
    parser.add_argument(
        "--out-figure", default="figures/stage1_ppl_vs_featuredamage.png"
    )
    args = parser.parse_args()

    rows = normalize_rows(
        read_csv(args.magnitude_csv, "magnitude")
        + read_csv(args.wanda_csv, "wanda")
    )
    write_combined(Path(args.out_csv), rows)
    make_plot(Path(args.out_figure), rows)
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

