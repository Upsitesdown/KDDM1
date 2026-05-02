"""
KDDM1 – Phase I EDA helpers.
Small Phase I EDA helpers: plots, summary tables, findings, and slide plan.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUT_DIR = Path("phase1_outputs")
ID_COLS = {"international_id", "medical_id", "physician_signature", "jersey_number"}
DATE_COLS = {"return_date"}
TEXT_COLS = {
    "first_name", "last_name", "birth_city", "nationality",
    "medical_information", "scout_notes", "personal_notes",
    "stuffed_animal_name", "favourite_board_game",
    "favourite_child_info", "favourite_tv_show", "favourite_food",
}

# Estimate plausible ranges (outlier checks)s
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


def _classify_feature(df: pd.DataFrame, col: str) -> str:
    if col in ID_COLS:
        return "id_meta"
    if col in DATE_COLS or pd.api.types.is_datetime64_any_dtype(df[col]):
        return "date"
    if col in TEXT_COLS:
        return "text"
    if pd.api.types.is_numeric_dtype(df[col]):
        return "numeric"
    return "categorical" if df[col].nunique(dropna=True) <= 20 else "text"


def _missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    pct = (miss / len(df) * 100).round(2)
    out = pd.DataFrame({
        "column": miss.index,
        "missing": miss.values,
        "missing_percent": pct.values,
        "feature_type": [_classify_feature(df, c) for c in miss.index],
    })
    return out.sort_values("missing_percent", ascending=False)


def _missingness_strategy(miss_tbl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in miss_tbl.iterrows():
        c, t, p = r["column"], r["feature_type"], r["missing_percent"]
        if p == 0:
            problem, action, warn = "none", "no action", ""
        elif p == 100:
            problem, action, warn = (
                "unusable column",
                "drop column (no information)",
                "do not use in Phase II",
            )
        elif p > 80:
            problem, action, warn = (
                "mostly missing",
                "drop column or treat as rare flag",
                "high risk of leakage / overfit if imputed",
            )
        elif t == "numeric":
            problem, action, warn = (
                "missing numeric",
                "median imputation (fit on train split only)",
                "impute AFTER train/test split to avoid leakage",
            )
        elif t == "categorical":
            problem, action, warn = (
                "missing categorical",
                "add 'Unknown' category",
                "do not mode-impute blindly",
            )
        elif t == "date":
            problem, action, warn = (
                "missing date",
                "keep NaT + add `<col>_missing` flag",
                "naive date imputation is not meaningful",
            )
        elif t == "text":
            problem, action, warn = (
                "missing free-text",
                "keep NaN + optional missing flag",
                "not a direct model feature",
            )
        elif t == "id_meta":
            problem, action, warn = (
                "missing id/meta",
                "keep as-is, do not impute",
                "not a predictive feature",
            )
        else:
            problem, action, warn = "missing", "review manually", ""
        rows.append({
            "column": c,
            "missing_percent": p,
            "feature_type": t,
            "problem_type": problem,
            "recommended_phase1_handling": action,
            "phase2_warning": warn,
        })
    return pd.DataFrame(rows)


def _plot_missingness(df: pd.DataFrame, out_dir: Path) -> None:
    miss_tbl = _missingness_table(df)
    miss_tbl.to_csv(out_dir / "missingness_summary.csv", index=False)
    _missingness_strategy(miss_tbl).to_csv(out_dir / "missingness_strategy.csv", index=False)

    top = miss_tbl[miss_tbl["missing_percent"] > 0].head(20)
    if not top.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(data=top, x="missing_percent", y="column", color="steelblue", ax=ax)
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
    summary.to_csv(out_dir / "outlier_summary.csv", index=False)

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
        fig.suptitle("Outliers in Key Numeric Features (raw values)")
        plt.tight_layout()
        fig.savefig(out_dir / "outliers_boxplots.png", dpi=150)
        plt.close(fig)

    return summary


def _domain_filtered(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with domain-impossible numeric values set to NaN."""
    out = df.copy()
    for c, (lo, hi) in DOMAIN_RANGES.items():
        if c in out.columns:
            if lo is not None:
                out.loc[out[c] < lo, c] = np.nan
            if hi is not None:
                out.loc[out[c] > hi, c] = np.nan
    return out


def make_phase1_processed_data(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Replace domain-impossible values with NaN and add invalid flag columns."""
    processed = df.copy()
    summary_rows = []

    # Domain range checks
    for col, (lo, hi) in DOMAIN_RANGES.items():
        if col not in processed.columns or not pd.api.types.is_numeric_dtype(processed[col]):
            continue
        s = processed[col]
        mask = pd.Series(False, index=processed.index)
        if lo is not None:
            mask |= s < lo
        if hi is not None:
            mask |= s > hi
        mask = mask & s.notna()
        n_inv = int(mask.sum())
        if n_inv > 0:
            processed[f"{col}_invalid_flag"] = mask.astype(int)
            summary_rows.append({
                "column": col,
                "rule": f"valid range: {lo} to {hi}",
                "invalid_count": n_inv,
                "invalid_percent": round(n_inv / len(processed) * 100, 2),
                "min_before": float(s.min()),
                "max_before": float(s.max()),
            })
            processed.loc[mask, col] = np.nan

    # Non-negative checks (columns not already covered by DOMAIN_RANGES)
    for col in NON_NEGATIVE:
        if col in DOMAIN_RANGES or col not in processed.columns:
            continue
        if not pd.api.types.is_numeric_dtype(processed[col]):
            continue
        s = processed[col]
        mask = (s < 0) & s.notna()
        n_inv = int(mask.sum())
        if n_inv > 0:
            processed[f"{col}_invalid_flag"] = mask.astype(int)
            summary_rows.append({
                "column": col,
                "rule": "must be non-negative",
                "invalid_count": n_inv,
                "invalid_percent": round(n_inv / len(processed) * 100, 2),
                "min_before": float(s.min()),
                "max_before": float(s.max()),
            })
            processed.loc[mask, col] = np.nan

    # Save outputs
    processed.to_csv(out_dir / "phase1_processed_dataset.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "invalid_value_summary.csv", index=False)

    print(f"\nProcessed dataset: {len(processed)} rows, {len(processed.columns)} cols")
    print(f"Invalid values replaced with NaN: {sum(r['invalid_count'] for r in summary_rows)} across {len(summary_rows)} columns")
    return processed


def _relationship_corr_and_pairs(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    corr_cols = [
        c for c in [
            "goals", "assists", "num_of_shots", "shot_speed", "shooting_percentage",
            "time_on_ice", "puck_touches", "passes_completed", "pass_completion_rate",
            "penalty_minutes", "games_missed_due_to_injury", "age", "height", "weight",
            "body_fat_percentage",
        ] if c in df.columns and c not in ID_COLS
    ]

    # Use a domain-filtered copy: domain-impossible values -> NaN before Pearson.
    clean = _domain_filtered(df)
    corr = clean[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation Heatmap (Pearson, domain-filtered)\n"
                 "linear only; sensitive to remaining outliers; not causal")
    plt.tight_layout()
    fig.savefig(out_dir / "relationships_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # Strongest off-diagonal pairs (upper triangle only, no mirrored duplicates).
    iu = np.triu_indices_from(corr, k=1)
    raw_r = corr.to_numpy()[iu]
    pairs = pd.DataFrame({
        "feature_a": corr.index.to_numpy()[iu[0]],
        "feature_b": corr.columns.to_numpy()[iu[1]],
        "pearson_r": raw_r,
        "abs_pearson_r": np.abs(raw_r),
    }).sort_values("abs_pearson_r", ascending=False).head(10)
    pairs.to_csv(out_dir / "top_correlations.csv", index=False)
    return pairs


def _plot_relationship_views(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = _domain_filtered(df)

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


def analyze_missingness_mechanisms(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Simple check whether missingness correlates with observed features."""
    cat_preds = [c for c in ["position", "gender", "fitness_level", "nationality"] if c in df.columns]
    num_preds = [c for c in ["age", "height", "weight", "games_missed_due_to_injury", "contracts_signed"] if c in df.columns]
    # include any source/merge indicator columns
    for c in df.columns:
        if ("source" in c.lower() or "merge" in c.lower()) and c not in cat_preds + num_preds:
            if pd.api.types.is_numeric_dtype(df[c]):
                num_preds.append(c)
            else:
                cat_preds.append(c)

    miss_pct = df.isna().mean() * 100
    candidates = [c for c in df.columns if 0 < miss_pct[c] < 100]

    rows = []
    # also note 100% missing columns
    for c in df.columns:
        if miss_pct[c] == 100:
            rows.append({"missing_column": c, "missing_percent": 100.0,
                         "best_related_feature": "", "relation_type": "", "score": np.nan,
                         "interpretation": "100% missing: unusable column, mechanism cannot be tested"})

    for col in candidates:
        is_miss = df[col].isna()
        n_miss, n_obs = int(is_miss.sum()), int((~is_miss).sum())
        if min(n_miss, n_obs) < 10:
            rows.append({"missing_column": col, "missing_percent": round(miss_pct[col], 2),
                         "best_related_feature": "", "relation_type": "", "score": np.nan,
                         "interpretation": "cannot test: too few missing/non-missing rows"})
            continue

        best_feat, best_score, best_type = "", 0.0, ""

        for p in cat_preds:
            if df[p].isna().all():
                continue
            rates = df.groupby(p)[col].apply(lambda s: s.isna().mean())
            diff = rates.max() - rates.min()
            if diff > best_score:
                best_score, best_feat, best_type = diff, p, "categorical_rate_diff"

        for p in num_preds:
            s = df[p]
            if s.isna().all() or s.std() == 0:
                continue
            med_miss = s[is_miss].median()
            med_obs = s[~is_miss].median()
            std = s.std()
            if pd.isna(med_miss) or pd.isna(med_obs) or std == 0:
                continue
            norm_diff = abs(med_miss - med_obs) / std
            if norm_diff > best_score:
                best_score, best_feat, best_type = norm_diff, p, "numeric_median_diff"

        threshold = 0.20 if best_type == "categorical_rate_diff" else 0.50
        if best_score >= threshold:
            interp = "possible MAR: missingness changes by observed feature"
        else:
            interp = "no strong observed relation found; consistent with MCAR but not proof"

        rows.append({"missing_column": col, "missing_percent": round(miss_pct[col], 2),
                     "best_related_feature": best_feat, "relation_type": best_type,
                     "score": round(best_score, 4), "interpretation": interp})

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "missingness_mechanism_check.csv", index=False)
    return result


def run_phase1_eda(df: pd.DataFrame, out_dir: Path = OUT_DIR) -> None:
    """Run compact EDA: graphs + summary CSVs + findings + slide plan + validation."""
    out_dir.mkdir(exist_ok=True, parents=True)

    _plot_missingness(df, out_dir)
    outlier_summary = _plot_outliers(df, out_dir)
    make_phase1_processed_data(df, out_dir)
    top_pairs = _relationship_corr_and_pairs(df, out_dir)
    _plot_relationship_views(df, out_dir)
    analyze_missingness_mechanisms(df, out_dir)

    print("\n" + "=" * 60)
    print("EDA SUMMARY")
    print("=" * 60)
    print("Saved outputs in", out_dir)
    if not outlier_summary.empty:
        print("\nTop outlier-domain issues:")
        print(outlier_summary.head(5).to_string(index=False))
    if not top_pairs.empty:
        print("\nTop absolute correlations (Pearson, linear only):")
        print(top_pairs.head(5).to_string(index=False))