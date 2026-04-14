"""Census Income — feature engineering visualisations.

Pure visualisation module for Phase 5 of the pipeline.
No data is transformed or persisted here — all functions produce read-only
views that communicate and validate the feature engineering decisions
documented in docs/pipeline_architecture.md.

Each function returns a ``matplotlib.figure.Figure`` so the caller controls
whether to display, save, or embed the plot.

Public API
----------
plot_missing_values          — which columns need imputation and how much
plot_numeric_distributions   — raw vs StandardScaler-transformed distributions
plot_numeric_target_split    — box plots by salary class (predictive power)
plot_categorical_target_rates — >50K rate per category (justifies keeping features)
plot_correlation_heatmap     — collinearity check on the transformed matrix
save_all_figures             — convenience: run all four + write PNGs to disk
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from census.data_loader import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# ── Visual style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
_PALETTE = {"<=50K": "#4878cf", ">50K": "#6acc65"}
_FIG_DPI = 120

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = _PROJECT_ROOT / "reports" / "figures"


# ── 1. Missing values ─────────────────────────────────────────────────────────


def plot_missing_values(
    features: pd.DataFrame,
    *,
    title: str = "Missing Values per Feature",
    figsize: tuple[float, float] = (9, 4),
) -> plt.Figure:
    """Horizontal bar chart: missing value percentage per column.

    Surfaces the three columns that require imputation before the sklearn
    pipeline runs: workclass (≈5.6%), occupation (≈5.7%),
    native_country (≈1.8%).  Columns with zero missings are omitted to keep
    the chart readable.

    Args:
        features: Feature matrix (output of split_features_target — no salary).
        title:    Chart title.
        figsize:  Figure size in inches.

    Returns:
        Figure with a single Axes.
    """
    missing_pct = (features.isnull().sum() / len(features) * 100).sort_values(
        ascending=False
    )
    missing_pct = missing_pct[missing_pct > 0]

    fig, ax = plt.subplots(figsize=figsize, dpi=_FIG_DPI)

    if missing_pct.empty:
        ax.text(
            0.5, 0.5, "No missing values found",
            ha="center", va="center", transform=ax.transAxes, fontsize=13,
        )
        ax.set_title(title)
        return fig

    colours = [
        "#e05c5c" if pct > 5 else "#f0a040" if pct > 1 else "#aaaaaa"
        for pct in missing_pct
    ]
    bars = ax.barh(
        missing_pct.index,
        missing_pct.values,
        color=colours,
        edgecolor="white",
    )

    for bar, pct in zip(bars, missing_pct.values, strict=True):
        ax.text(
            bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{pct:.2f}%",
            va="center", ha="left", fontsize=10,
        )

    ax.set_xlabel("Missing (%)")
    ax.set_title(title)
    ax.set_xlim(0, missing_pct.max() * 1.25)
    ax.invert_yaxis()

    # Imputer strategy annotation
    ax.axvline(0, color="black", linewidth=0.8)
    fig.text(
        0.99, 0.01,
        "Imputer strategy: most_frequent  (avoids unseen category at inference)",
        ha="right", va="bottom", fontsize=8, color="grey",
    )
    fig.tight_layout()
    return fig


# ── 2. Numeric distributions: raw vs scaled ───────────────────────────────────


def plot_numeric_distributions(
    features: pd.DataFrame,
    *,
    title: str = "Numeric Features — Raw vs StandardScaler",
    figsize: tuple[float, float] = (16, 8),
) -> plt.Figure:
    """Side-by-side histograms: original and StandardScaler-normalised.

    Justifies the use of StandardScaler:
    - capital_gain / capital_loss are highly right-skewed and zero-heavy.
    - After scaling, all features share a common scale (mean=0, std=1).
    - Tree-based models are scale-invariant, but linear models and distance-based
      algorithms require this normalisation.

    The scaler is fitted on *features* (the subset passed in); when used in
    context this should be the training split only to avoid leakage.

    Args:
        features: DataFrame containing at least all NUMERIC_FEATURES columns.
        title:    Figure suptitle.
        figsize:  Figure size in inches.

    Returns:
        Figure with two rows × len(NUMERIC_FEATURES) Axes.
    """
    n_cols = len(NUMERIC_FEATURES)
    fig, axes = plt.subplots(2, n_cols, figsize=figsize, dpi=_FIG_DPI)
    fig.suptitle(title, fontsize=14, y=1.01)

    # Fit scaler on the provided data (training split in practice)
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(features[NUMERIC_FEATURES].fillna(
        features[NUMERIC_FEATURES].median()
    ))
    scaled_df = pd.DataFrame(scaled_array, columns=NUMERIC_FEATURES)

    for col_idx, col in enumerate(NUMERIC_FEATURES):
        ax_raw = axes[0, col_idx]
        ax_scaled = axes[1, col_idx]

        raw_values = features[col].dropna()
        scaled_values = scaled_df[col]

        # Row 0 — raw distribution
        ax_raw.hist(raw_values, bins=40, color="#4878cf", edgecolor="white", alpha=0.85)
        ax_raw.set_title(col, fontsize=10)
        ax_raw.set_ylabel("Count" if col_idx == 0 else "")
        ax_raw.tick_params(axis="x", labelsize=8)

        # Row 1 — scaled distribution
        ax_scaled.hist(
            scaled_values,
            bins=40,
            color="#6acc65",
            edgecolor="white",
            alpha=0.85,
        )
        ax_scaled.tick_params(axis="x", labelsize=8)
        ax_scaled.set_ylabel("Count" if col_idx == 0 else "")

        # Annotate mean and std on scaled row
        ax_scaled.axvline(
            0,
            color="#e05c5c",
            linestyle="--",
            linewidth=1.2,
            label="mean=0",
        )
        ax_scaled.text(
            0.97, 0.93, f"μ={scaled_values.mean():.2f}\nσ={scaled_values.std():.2f}",
            transform=ax_scaled.transAxes, ha="right", va="top",
            fontsize=7.5, color="black",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7},
        )

    axes[0, 0].set_ylabel("Raw  —  Count", fontsize=9)
    axes[1, 0].set_ylabel("Scaled  —  Count", fontsize=9)

    fig.tight_layout()
    return fig


# ── 3. Numeric features split by target class ────────────────────────────────


def plot_numeric_target_split(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    title: str = "Numeric Features by Salary Class",
    figsize: tuple[float, float] = (16, 4),
) -> plt.Figure:
    """Box plots of each numeric feature split by <=50K / >50K.

    Reveals the predictive signal in each feature:
    - education_num: clear separation (H1 — monotonic relationship).
    - capital_gain:  extreme outliers only in >50K class (H2 — quasi-binary).
    - hours_per_week: overlapping medians (H4 — weak predictor in isolation).
    - age: moderate separation.

    Args:
        features: Feature matrix (NUMERIC_FEATURES must be present).
        target:   Binary Series (0=<=50K, 1=>50K) or string salary Series.
        title:    Figure suptitle.
        figsize:  Figure size in inches.

    Returns:
        Figure with one row × len(NUMERIC_FEATURES) Axes.
    """
    # Accept both binary-encoded and raw-string target
    if pd.api.types.is_integer_dtype(target):
        salary_label = target.map({0: "<=50K", 1: ">50K"})
    else:
        salary_label = target

    df_plot = features[NUMERIC_FEATURES].copy()
    df_plot["salary"] = salary_label.values

    fig, axes = plt.subplots(1, len(NUMERIC_FEATURES), figsize=figsize, dpi=_FIG_DPI)
    fig.suptitle(title, fontsize=13, y=1.02)

    for ax, col in zip(axes, NUMERIC_FEATURES, strict=True):
        sns.boxplot(
            data=df_plot,
            x="salary",
            y=col,
            hue="salary",
            palette=_PALETTE,
            order=["<=50K", ">50K"],
            hue_order=["<=50K", ">50K"],
            width=0.5,
            fliersize=1.5,
            legend=False,
            ax=ax,
        )
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=8)

    fig.tight_layout()
    return fig


# ── 4. Categorical features — positive rate per category ─────────────────────


def plot_categorical_target_rates(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    title: str = "Categorical Features — >50K Rate per Category",
    figsize: tuple[float, float] | None = None,
    max_categories: int = 12,
) -> plt.Figure:
    """Horizontal bar charts: >50K positive rate per category per feature.

    Surfaces the predictive value of each categorical feature and highlights
    fairness-sensitive variables:
    - marital_status: Married-civ-spouse ≈44.7% vs Never-married ≈4.6% (H3).
    - sex:            Male ≈30.6% vs Female ≈10.9% (19.6 p.p. gap).
    - race:           White ≈25.6% vs Other ≈9.2% (17.3 p.p. gap).

    Categories are sorted by positive rate (descending).  When a feature has
    more than *max_categories* unique values, only the top + bottom are shown.

    Args:
        features:       Feature matrix (CATEGORICAL_FEATURES must be present).
        target:         Binary Series (0/1) or string salary Series.
        title:          Figure suptitle.
        figsize:        Figure size; auto-computed if None.
        max_categories: Maximum number of bars to show per feature.

    Returns:
        Figure with one Axes per categorical feature.
    """
    n_features = len(CATEGORICAL_FEATURES)
    cols_per_row = 4
    n_rows = (n_features + cols_per_row - 1) // cols_per_row

    if figsize is None:
        figsize = (cols_per_row * 5, n_rows * 4.5)

    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=figsize, dpi=_FIG_DPI)
    axes_flat: list[plt.Axes] = list(axes.flatten()) if n_rows > 1 else list(axes)
    fig.suptitle(title, fontsize=13, y=1.01)

    # Resolve target to binary float
    if pd.api.types.is_integer_dtype(target) or pd.api.types.is_float_dtype(target):
        binary_target = target.astype(float)
    else:
        binary_target = (target == ">50K").astype(float)

    for ax, col in zip(axes_flat, CATEGORICAL_FEATURES, strict=False):
        rates = (
            pd.DataFrame({"cat": features[col].values, "label": binary_target.values})
            .groupby("cat", dropna=False)["label"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "positive_rate", "count": "n"})
            .sort_values("positive_rate", ascending=True)
        )

        # Limit to max_categories (keep extreme values for readability)
        if len(rates) > max_categories:
            half = max_categories // 2
            rates = pd.concat([rates.head(half), rates.tail(half)])

        bar_colours = [
            "#e05c5c" if r >= 0.30 else "#f0a040" if r >= 0.15 else "#4878cf"
            for r in rates["positive_rate"]
        ]
        ax.barh(rates.index.astype(str), rates["positive_rate"], color=bar_colours,
                edgecolor="white")
        ax.set_title(col, fontsize=10)
        ax.set_xlabel(">50K rate")
        ax.set_xlim(0, 1)
        ax.axvline(binary_target.mean(), color="black", linestyle="--",
                   linewidth=1.0, label=f"overall ({binary_target.mean():.1%})")
        ax.legend(fontsize=7, loc="lower right")

        for y_pos, (rate, n) in enumerate(
            zip(rates["positive_rate"], rates["n"], strict=True)
        ):
            ax.text(rate + 0.01, y_pos, f"{rate:.1%} (n={n:,})",
                    va="center", fontsize=7)

    # Hide unused axes
    for ax in axes_flat[n_features:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


# ── 5. Correlation heatmap on the transformed matrix ─────────────────────────


def plot_correlation_heatmap(
    train_array: np.ndarray,
    feature_names: list[str],
    *,
    title: str = "Feature Correlation Heatmap (Transformed)",
    figsize: tuple[float, float] = (11, 9),
) -> plt.Figure:
    """Pearson correlation heatmap on the ColumnTransformer output.

    Checks for multicollinearity in the transformed feature space before
    model selection.  High correlations (|r| > 0.7) are worth investigating:
    - marital_status and relationship are known proxies of sex (see architecture).
    - age and education_num have mild positive correlation.

    Args:
        train_array:   Output of fit_preprocessor() — shape (n_train, n_features).
        feature_names: Ordered list matching the columns in train_array.
                       Typically NUMERIC_FEATURES + CATEGORICAL_FEATURES.
        title:         Axes title.
        figsize:       Figure size in inches.

    Returns:
        Figure with a single heatmap Axes.
    """
    corr_matrix = pd.DataFrame(train_array, columns=feature_names).corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # upper triangle

    fig, ax = plt.subplots(figsize=figsize, dpi=_FIG_DPI)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.4,
        annot_kws={"size": 8},
        ax=ax,
    )
    ax.set_title(title, fontsize=13, pad=12)
    fig.tight_layout()
    return fig


# ── 6. Save all figures ───────────────────────────────────────────────────────


def save_all_figures(
    features: pd.DataFrame,
    target: pd.Series,
    train_array: np.ndarray,
    *,
    output_dir: Path | str = FIGURES_DIR,
) -> list[Path]:
    """Generate and save all feature engineering plots to *output_dir*.

    Calls every visualisation function in this module and writes each result
    as a PNG file.  Existing files are silently overwritten.

    Args:
        features:    Feature matrix (output of split_features_target).
        target:      Binary (0/1) or string salary Series.
        train_array: Transformed training array (output of fit_preprocessor).
        output_dir:  Destination directory.  Created if it does not exist.

    Returns:
        List of resolved Paths to the written PNG files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    plots: dict[str, plt.Figure] = {
        "01_missing_values": plot_missing_values(features),
        "02_numeric_distributions": plot_numeric_distributions(features),
        "03_numeric_target_split": plot_numeric_target_split(features, target),
        "04_categorical_target_rates": plot_categorical_target_rates(features, target),
        "05_correlation_heatmap": plot_correlation_heatmap(train_array, feature_names),
    }

    saved: list[Path] = []
    for filename, fig in plots.items():
        dest = output_dir / f"{filename}.png"
        fig.savefig(dest, bbox_inches="tight")
        plt.close(fig)
        saved.append(dest.resolve())

    return saved


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from loguru import logger

    from census.configure_logging import configure_logging
    from census.preprocessing import (
        build_preprocessor,
        fit_preprocessor,
        load_cleaned_data,
        split_features_target,
        split_train_test,
    )

    _LOG_DIR = _PROJECT_ROOT / "logs" / "feature_engineering_log"
    configure_logging(_LOG_DIR, "feature_engineering")

    logger.info("Loading cleaned data…")
    df = load_cleaned_data()
    features, target = split_features_target(df)
    features_train, features_test, target_train, _ = split_train_test(features, target)

    logger.info("Fitting preprocessor on training split…")
    preprocessor = build_preprocessor()
    train_array, _ = fit_preprocessor(preprocessor, features_train, features_test)

    logger.info("Generating and saving figures…")
    saved_paths = save_all_figures(features_train, target_train, train_array)

    for path in saved_paths:
        logger.info("Saved: {}", path)

    logger.info("Done — {} figures written to {}", len(saved_paths), FIGURES_DIR)
