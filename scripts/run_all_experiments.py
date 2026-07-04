#!/usr/bin/env python3
"""Run every experiment in the project and save results/figures to disk.

Usage (from the project root):
    python scripts/run_all_experiments.py

Outputs:
    results/*.csv   - one file per experiment
    figures/*.png   - class balance, correlation heatmap, threshold plot,
                      original-vs-grouped-vs-dedup comparison
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe; script also works fine interactively
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, data_loading, evaluation, experiments, preprocessing


def ensure_dirs():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    ensure_dirs()

    section("1. Loading the article dataset")
    df_raw, source = data_loading.load_article_dataset()
    print("Data source:", source)
    print("Raw shape:", df_raw.shape)

    dup_info = preprocessing.duplicate_summary(df_raw)
    print("Duplicate summary:", dup_info)
    pd.DataFrame([dup_info]).to_csv(
        os.path.join(config.RESULTS_DIR, "dataset_summary.csv"), index=False
    )

    # ---------------------------------------------------------------
    section("2. Original-style split (random, duplicates included)")
    original_results, original_fitted = experiments.run_original_style_split(df_raw)
    print(original_results[["Model", "Accuracy", "Precision", "Recall", "F1", "PR_AUC"]])
    original_results.to_csv(
        os.path.join(config.RESULTS_DIR, "results_original_style_split.csv"), index=False
    )

    # ---------------------------------------------------------------
    section("3. Deduplicated split (random, duplicates removed first)")
    dedup_results, dedup_fitted = experiments.run_deduplicated_split(df_raw)
    print(dedup_results[["Model", "Accuracy", "Precision", "Recall", "F1", "PR_AUC"]])
    dedup_results.to_csv(
        os.path.join(config.RESULTS_DIR, "results_deduplicated_split.csv"), index=False
    )

    # ---------------------------------------------------------------
    section("4. Grouped split (reviewer-requested: duplicate-safe GroupShuffleSplit)")
    grouped_results, grouped_fitted = experiments.run_grouped_split(df_raw)
    print(grouped_results[["Model", "Accuracy", "Precision", "Recall", "F1", "PR_AUC"]])
    grouped_results.to_csv(
        os.path.join(config.RESULTS_DIR, "results_grouped_split.csv"), index=False
    )

    # Combined comparison table (used in the report + a figure below)
    comparison = pd.concat(
        [original_results, dedup_results, grouped_results], ignore_index=True
    )
    comparison = comparison[comparison["Model"] == "Random Forest"][
        ["Experiment", "Accuracy", "Precision", "Recall", "F1", "MCC", "ROC_AUC",
         "PR_AUC", "FP", "FN"]
    ]
    comparison.to_csv(
        os.path.join(config.RESULTS_DIR, "comparison_rf_across_splits.csv"), index=False
    )
    print("\nRandom Forest across split strategies:")
    print(comparison)

    # ---------------------------------------------------------------
    section("5. Cross-validation (stratified vs. grouped)")
    cv_stratified = experiments.cross_validation_summary(
        df_raw, "Original rows (StratifiedKFold)", grouped=False
    )
    cv_grouped = experiments.cross_validation_summary(
        df_raw, "Original rows (GroupKFold, duplicate-safe)", grouped=True
    )
    cv_dedup = experiments.cross_validation_summary(
        preprocessing.deduplicate(df_raw), "Deduplicated rows (StratifiedKFold)", grouped=False
    )
    cv_all = pd.concat([cv_stratified, cv_grouped, cv_dedup], ignore_index=True)
    print(cv_all[["Dataset", "Model", "f1_mean", "recall_mean", "mcc_mean", "pr_auc_mean"]])
    cv_all.to_csv(os.path.join(config.RESULTS_DIR, "cross_validation_results.csv"), index=False)

    # ---------------------------------------------------------------
    section("6. Threshold analysis (Random Forest, original-style vs. grouped)")
    rf_original = original_fitted["Random Forest"]
    rf_grouped = grouped_fitted["Random Forest"]

    threshold_original = evaluation.threshold_table(
        rf_original["y_test"], rf_original["y_score"]
    )
    threshold_grouped = evaluation.threshold_table(
        rf_grouped["y_test"], rf_grouped["y_score"]
    )
    print("Original-style split thresholds:\n", threshold_original)
    print("Grouped split thresholds:\n", threshold_grouped)
    threshold_original.to_csv(
        os.path.join(config.RESULTS_DIR, "threshold_analysis_original_style.csv"), index=False
    )
    threshold_grouped.to_csv(
        os.path.join(config.RESULTS_DIR, "threshold_analysis_grouped.csv"), index=False
    )

    # ---------------------------------------------------------------
    section("7. Low-prevalence phishing scenario (5%)")
    low_prevalence = evaluation.simulate_low_prevalence(
        rf_original["y_test"], rf_original["y_score"]
    )
    print(low_prevalence)
    low_prevalence.to_csv(
        os.path.join(config.RESULTS_DIR, "low_prevalence_scenario.csv"), index=False
    )

    # ---------------------------------------------------------------
    section("8. Newer (2020) external dataset - cross-dataset generalization check")
    try:
        df_newer = data_loading.load_newer_dataset()
        print("Newer dataset shape:", df_newer.shape)
        newer_results, _, newer_ci = experiments.run_newer_dataset_experiment(df_newer)
        print(newer_results[["Model", "Accuracy", "Precision", "Recall", "F1", "PR_AUC"]])
        newer_results.to_csv(
            os.path.join(config.RESULTS_DIR, "results_newer_dataset_2020.csv"), index=False
        )
        print("Bootstrap 95% CIs (full-dataset test split, ~14,662 rows - expect a tight band):")
        print(newer_ci)
        newer_ci.to_csv(
            os.path.join(config.RESULTS_DIR, "results_newer_dataset_2020_bootstrap_ci.csv"),
            index=False,
        )
    except FileNotFoundError as e:
        print("Skipped:", e)

    # ---------------------------------------------------------------
    section("9. Figures")
    make_figures(df_raw, comparison, threshold_original)

    section("Done")
    print(f"Results written to: {config.RESULTS_DIR}")
    print(f"Figures written to: {config.FIGURES_DIR}")


def make_figures(df_raw, comparison, threshold_original):
    df, _ = preprocessing.prepare_binary_dataset(df_raw)

    # Figure 1: class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    df[config.TARGET_COLUMN].value_counts().sort_index().plot(kind="bar", ax=ax,
                                                                color=["#4C72B0", "#DD8452"])
    ax.set_title("Target distribution: 0=legitimate, 1=phishing")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_DIR, "class_distribution.png"), dpi=150)
    plt.close(fig)

    # Figure 2: Spearman correlation heatmap (top correlated features)
    corr = df.corr(method="spearman")
    target_corr = corr[config.TARGET_COLUMN].drop(config.TARGET_COLUMN)
    top_features = target_corr.abs().sort_values(ascending=False).head(10).index.tolist()
    sub_corr = df[top_features + [config.TARGET_COLUMN]].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sub_corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(sub_corr.columns)))
    ax.set_xticklabels(sub_corr.columns, rotation=90)
    ax.set_yticks(range(len(sub_corr.columns)))
    ax.set_yticklabels(sub_corr.columns)
    ax.set_title("Spearman correlation: top features vs. phishing label")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)

    # Figure 3: threshold trade-off
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(threshold_original["Threshold"], threshold_original["Precision"], marker="o", label="Precision")
    ax.plot(threshold_original["Threshold"], threshold_original["Recall"], marker="o", label="Recall")
    ax.plot(threshold_original["Threshold"], threshold_original["F1"], marker="o", label="F1")
    ax.set_title("Random Forest (original-style split): threshold trade-off")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_DIR, "threshold_tradeoff.png"), dpi=150)
    plt.close(fig)

    # Figure 4: RF metrics across split strategies (the key critical-evaluation figure)
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_to_plot = ["Accuracy", "Recall", "F1", "MCC"]
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    for i, (_, row) in enumerate(comparison.iterrows()):
        ax.bar(x + i * width, [row[m] for m in metrics_to_plot], width, label=row["Experiment"])
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylim(0.8, 1.0)
    ax.set_title("Random Forest: original-style vs. deduplicated vs. grouped split")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_DIR, "rf_split_comparison.png"), dpi=150)
    plt.close(fig)

    print("Saved 4 figures to", config.FIGURES_DIR)


if __name__ == "__main__":
    main()
