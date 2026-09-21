"""
Renders docs/vs_field_benchmark.png from the raw result JSONs in results/ --
the chart form of the "poorjev vs the field" table in the main README.

Usage (from crossbench/):
    pip install matplotlib
    python plot_comparison.py
"""
import json
import matplotlib.pyplot as plt

SYSTEMS = ["Jev", "von", "Laya", "poorjev"]
COLORS = {"Jev": "#9aa0a6", "von": "#9aa0a6", "Laya": "#9aa0a6", "poorjev": "#d9534f"}

BANKING77 = {
    "Jev": json.load(open("results/banking77_jev_results.json"))["overall"],
    "von": json.load(open("results/banking77_von_results.json"))["overall"],
    "Laya": json.load(open("results/banking77_laya_results.json"))["overall"],
    "poorjev": json.load(open("results/banking77_poorjev_results.json"))["overall"],
}

MULTI = {
    "Jev": json.load(open("results/typed_decisions_jev_results.json"))["overall"],
    "von": json.load(open("results/typed_decisions_von_results.json"))["overall"],
    "Laya": json.load(open("results/typed_decisions_laya_results.json"))["overall"],
    "poorjev": json.load(open("results/typed_decisions_poorjev_results.json"))["calibrated_5fold_cv"],
}


def bar(ax, data, metric, title, ylabel, lower_is_better=False):
    vals = [data[s][metric] for s in SYSTEMS]
    colors = [COLORS[s] for s in SYSTEMS]
    bars = ax.bar(SYSTEMS, vals, color=colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(vals) * 1.25)
    suffix = " (lower is better)" if lower_is_better else ""
    ax.set_xlabel(suffix, fontsize=8, color="#666")


fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.suptitle("poorjev vs the field -- independently measured, same inputs, same metrics code",
             fontsize=12, fontweight="bold")

bar(axes[0, 0], BANKING77, "accuracy", "Banking77 (77-way choice, n=154) -- Accuracy", "accuracy")
bar(axes[0, 1], BANKING77, "ece", "Banking77 -- ECE", "ECE", lower_is_better=True)
bar(axes[1, 0], MULTI, "accuracy", "Multi-primitive set (n=160) -- Accuracy", "accuracy")
bar(axes[1, 1], MULTI, "ece", "Multi-primitive set -- ECE", "ECE", lower_is_better=True)

fig.text(0.5, 0.01,
         "poorjev highlighted in red. Full methodology, fairness notes, and raw results: crossbench/",
         ha="center", fontsize=8, color="#666")
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig("../docs/vs_field_benchmark.png", dpi=150)
print("wrote ../docs/vs_field_benchmark.png")
