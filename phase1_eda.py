"""
KDDM1 – Phase I EDA helpers.
Compact graph-focused outputs (no slide-specific CSV/Markdown exports).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUT_DIR = Path("phase1_outputs")
ID_COLS = {"international_id", "medical_id", "physician_signature", "jersey_number"}

# Domain plausibility ranges used for outlier checks and cleaner plots.
DOMAIN_RANGES: dict[str, tuple[float | None, float | None]] = {
    "age": (15, 50),
    "age_in_years": (15, 50),
    "height": (1.4, 2.3),
    "weight": (45, 150),
    "body_fat_percentage": (3, 35),
    "shooting_percentage": (0, 100),
    "save_percentage": (0, 100),
    "pass_completion_rate": (0, 100),
    "faceoff_win_percentage": (0, 100),
    "school_grade": (1, 5),
    "years_in_usa": (0, 60),
    "shoe_size": (30, 55),
}

NON_NEGATIVE = {
    "goals", "assists", "num_of_shots", "shot_attempts", "high_danger_shots",
    "medium_danger_shots", "low_danger_shots", "winning_goals", "power_play_goals",
    "power_play_time", "time_on_ice", "puck_touches", "puck_recoveries",
    "puck_possession_time", "penalty_kill_time", "penalty_minutes", "penalties_taken",
    "goals_against_total", "passes_attempted", "passes_completed",
    "games_missed_due_to_injury", "contracts_signed", "salary", "years_played",
    "years_pro", "number_of_previous_teams",
}

def _numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in ID_COLS
    ]


def _plot_missingness(df: pd.DataFrame, out_dir: Path) -> None:
    miss = df.isna().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    top = pd.DataFrame({"column": miss.index, "percent": miss_pct.values})
    top = top[top["percent"] > 0].sort_values("percent", ascending=False).head(20)
    if top.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=top, x="percent", y="column", color="steelblue", ax=ax)
    ax.set_title("Top Missing Columns")
    ax.set_xlabel("Missing (%)")
    plt.tight_layout()
    fig.savefig(out_dir / "missingness_top20.png", dpi=150)
    plt.close(fig)


def _outlier_summary(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_out = int(((s < lo) | (s > hi)).sum())

        dom_lo, dom_hi = DOMAIN_RANGES.get(col, (None, None))
        dom_out = 0
        if dom_lo is not None:
            dom_out += int((s < dom_lo).sum())
        if dom_hi is not None:
            dom_out += int((s > dom_hi).sum())
        if col in NON_NEGATIVE:
            dom_out += int((s < 0).sum())

        rows.append({
            "column": col,
            "iqr_outliers": iqr_out,
            "iqr_outlier_pct": round(iqr_out / len(s) * 100, 2),
            "domain_violations": dom_out,
            "min": float(s.min()),
            "max": float(s.max()),
        })
    if not rows:
        return pd.DataFrame(columns=["column", "iqr_outliers", "iqr_outlier_pct", "domain_violations", "min", "max"])
    return pd.DataFrame(rows).sort_values(["domain_violations", "iqr_outlier_pct"], ascending=False)


def _plot_outliers(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    numeric_cols = _numeric_feature_cols(df)
    summary = _outlier_summary(df, numeric_cols)

    focus = [
        c for c in ["age", "height", "weight", "body_fat_percentage", "shooting_percentage", "pass_completion_rate"]
        if c in df.columns
    ]
    if focus:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for ax, col in zip(axes.flat, focus):
            sns.boxplot(x=df[col].dropna(), ax=ax, color="steelblue")
            ax.set_title(col)
        for ax in axes.flat[len(focus):]:
            ax.axis("off")
        fig.suptitle("Outliers in Key Numeric Features")
        plt.tight_layout()
        fig.savefig(out_dir / "outliers_boxplots.png", dpi=150)
        plt.close(fig)

    return summary


def _relationship_corr_and_pairs(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    corr_cols = [
        c for c in [
            "goals", "assists", "num_of_shots", "shot_speed", "shooting_percentage",
            "time_on_ice", "puck_touches", "passes_completed", "pass_completion_rate",
            "penalty_minutes", "games_missed_due_to_injury", "age", "height", "weight",
            "body_fat_percentage",
        ] if c in df.columns
    ]

    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation Heatmap (Pearson)")
    plt.tight_layout()
    fig.savefig(out_dir / "relationships_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # Correct strongest-pairs extraction: one triangle only (no mirrored duplicates).
    iu = np.triu_indices_from(corr, k=1)
    pairs = pd.DataFrame(
        {
            "feature_a": corr.index.to_numpy()[iu[0]],
            "feature_b": corr.columns.to_numpy()[iu[1]],
            "abs_pearson_r": np.abs(corr.to_numpy()[iu]),
        }
    ).sort_values("abs_pearson_r", ascending=False).head(10)
    return pairs


def _plot_relationship_views(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df.copy()
    for c, (lo, hi) in DOMAIN_RANGES.items():
        if c in plot_df.columns:
            if lo is not None:
                plot_df.loc[plot_df[c] < lo, c] = np.nan
            if hi is not None:
                plot_df.loc[plot_df[c] > hi, c] = np.nan

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    if {"age", "goals"}.issubset(plot_df.columns):
        sns.scatterplot(data=plot_df, x="age", y="goals", alpha=0.3, ax=axes[0])
        axes[0].set_title("Age vs Goals")
    if {"weight", "shot_speed"}.issubset(plot_df.columns):
        sns.scatterplot(data=plot_df, x="weight", y="shot_speed", alpha=0.3, ax=axes[1])
        axes[1].set_title("Weight vs Shot Speed")
    if {"position", "time_on_ice"}.issubset(plot_df.columns):
        order = plot_df["position"].value_counts().index.tolist()
        sns.boxplot(data=plot_df, x="position", y="time_on_ice", order=order, ax=axes[2])
        axes[2].set_title("Time on Ice by Position")
        axes[2].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(out_dir / "relationships_scatter_grouped.png", dpi=150)
    plt.close(fig)


def run_phase1_eda(df: pd.DataFrame, out_dir: Path = OUT_DIR) -> None:
    """Run compact EDA and save only graph outputs."""
    out_dir.mkdir(exist_ok=True, parents=True)

    _plot_missingness(df, out_dir)
    outlier_summary = _plot_outliers(df, out_dir)
    top_pairs = _relationship_corr_and_pairs(df, out_dir)
    _plot_relationship_views(df, out_dir)

    print("\n" + "=" * 60)
    print("EDA SUMMARY")
    print("=" * 60)
    print("Saved graphs:")
    print(f"- {out_dir / 'missingness_top20.png'}")
    print(f"- {out_dir / 'outliers_boxplots.png'}")
    print(f"- {out_dir / 'relationships_correlation_heatmap.png'}")
    print(f"- {out_dir / 'relationships_scatter_grouped.png'}")

    if not outlier_summary.empty:
        print("\nTop outlier-domain issues:")
        print(outlier_summary.head(5).to_string(index=False))

    if not top_pairs.empty:
        print("\nTop absolute correlations (Pearson, linear only):")
        print(top_pairs.head(5).to_string(index=False))
