from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from processing import DOMAIN_RANGES, NON_NEGATIVE, domain_filtered, make_processed_dataset

OUT_DIR = Path("phase1_outputs")
ID_COLS = {"international_id", "medical_id", "physician_signature", "jersey_number"}
DATE_COLS = {"return_date"}
TEXT_COLS = {
    "first_name", "last_name", "birth_city", "nationality",
    "medical_information", "scout_notes", "personal_notes",
    "stuffed_animal_name", "favourite_board_game",
    "favourite_child_info", "favourite_tv_show", "favourite_food",
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
    for _, row in miss_tbl.iterrows():
        col, feature_type, missing_percent = row["column"], row["feature_type"], row["missing_percent"]
        if missing_percent == 0:
            problem, action, warn = "none", "no action", ""
        elif missing_percent == 100:
            problem, action, warn = (
                "unusable column",
                "drop column (no information)",
                "do not use in Phase II",
            )
        elif missing_percent > 80:
            problem, action, warn = (
                "mostly missing",
                "drop column or treat as rare flag",
                "high risk of leakage / overfit if imputed",
            )
        elif feature_type == "numeric":
            problem, action, warn = (
                "missing numeric",
                "median imputation (fit on train split only)",
                "impute AFTER train/test split to avoid leakage",
            )
        elif feature_type == "categorical":
            problem, action, warn = (
                "missing categorical",
                "add 'Unknown' category",
                "do not mode-impute blindly",
            )
        elif feature_type == "date":
            problem, action, warn = (
                "missing date",
                "keep NaT + add `<col>_missing` flag",
                "naive date imputation is not meaningful",
            )
        elif feature_type == "text":
            problem, action, warn = (
                "missing free-text",
                "keep NaN + optional missing flag",
                "not a direct model feature",
            )
        elif feature_type == "id_meta":
            problem, action, warn = (
                "missing id/meta",
                "keep as-is, do not impute",
                "not a predictive feature",
            )
        else:
            problem, action, warn = "missing", "review manually", ""
        rows.append({
            "column": col,
            "missing_percent": missing_percent,
            "feature_type": feature_type,
            "problem_type": problem,
            "recommended_phase1_handling": action,
            "phase2_warning": warn,
        })
    return pd.DataFrame(rows)


def _plot_missingness(df: pd.DataFrame, out_dir: Path) -> None:
    miss_tbl = _missingness_table(df)
    miss_tbl.to_csv(out_dir / "missingness_summary.csv", index=False)
    _missingness_strategy(miss_tbl).to_csv(out_dir / "missingness_strategy.csv", index=False)


def _outlier_summary(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_outliers = int(((series < low) | (series > high)).sum())

        domain_low, domain_high = DOMAIN_RANGES.get(col, (None, None))
        domain_violations = 0
        if domain_low is not None:
            domain_violations += int((series < domain_low).sum())
        if domain_high is not None:
            domain_violations += int((series > domain_high).sum())
        if col in NON_NEGATIVE:
            domain_violations += int((series < 0).sum())

        rows.append({
            "column": col,
            "iqr_outliers": iqr_outliers,
            "iqr_outlier_pct": round(iqr_outliers / len(series) * 100, 2),
            "domain_violations": domain_violations,
            "min": float(series.min()),
            "max": float(series.max()),
        })

    if not rows:
        return pd.DataFrame(columns=["column", "iqr_outliers", "iqr_outlier_pct", "domain_violations", "min", "max"])
    return pd.DataFrame(rows).sort_values(["domain_violations", "iqr_outlier_pct"], ascending=False)


def _plot_outliers(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    numeric_cols = _numeric_feature_cols(df)
    summary = _outlier_summary(df, numeric_cols)
    summary.to_csv(out_dir / "outlier_summary.csv", index=False)

    return summary


def _relationship_corr_and_pairs(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    corr_cols = [
        c for c in [
            "goals", "assists", "num_of_shots", "shot_speed", "shooting_percentage",
            "time_on_ice", "puck_touches", "passes_completed", "pass_completion_rate",
            "penalty_minutes", "games_missed_due_to_injury", "age", "height", "weight",
            "body_fat_percentage",
        ] if c in df.columns and c not in ID_COLS
    ]

    clean = domain_filtered(df)
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
    plot_df = domain_filtered(df)

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
    for col in df.columns:
        if ("source" in col.lower() or "merge" in col.lower()) and col not in cat_preds + num_preds:
            if pd.api.types.is_numeric_dtype(df[col]):
                num_preds.append(col)
            else:
                cat_preds.append(col)

    miss_pct = df.isna().mean() * 100
    candidates = [c for c in df.columns if 0 < miss_pct[c] < 100]

    rows = []
    for col in df.columns:
        if miss_pct[col] == 100:
            rows.append({
                "missing_column": col,
                "missing_percent": 100.0,
                "best_related_feature": "",
                "relation_type": "",
                "score": np.nan,
                "interpretation": "100% missing: unusable column, mechanism cannot be tested",
            })

    for col in candidates:
        is_missing = df[col].isna()
        missing_count = int(is_missing.sum())
        observed_count = int((~is_missing).sum())
        if min(missing_count, observed_count) < 10:
            rows.append({
                "missing_column": col,
                "missing_percent": round(miss_pct[col], 2),
                "best_related_feature": "",
                "relation_type": "",
                "score": np.nan,
                "interpretation": "cannot test: too few missing/non-missing rows",
            })
            continue

        best_feature, best_score, best_type = "", 0.0, ""

        for pred in cat_preds:
            if df[pred].isna().all():
                continue
            rates = df.groupby(pred)[col].apply(lambda s: s.isna().mean())
            diff = rates.max() - rates.min()
            if diff > best_score:
                best_score, best_feature, best_type = diff, pred, "categorical_rate_diff"

        for pred in num_preds:
            series = df[pred]
            if series.isna().all() or series.std() == 0:
                continue
            median_missing = series[is_missing].median()
            median_observed = series[~is_missing].median()
            std = series.std()
            if pd.isna(median_missing) or pd.isna(median_observed) or std == 0:
                continue
            norm_diff = abs(median_missing - median_observed) / std
            if norm_diff > best_score:
                best_score, best_feature, best_type = norm_diff, pred, "numeric_median_diff"

        threshold = 0.20 if best_type == "categorical_rate_diff" else 0.50
        if best_score >= threshold:
            interpretation = "possible MAR: missingness changes by observed feature"
        else:
            interpretation = "no strong observed relation found; consistent with MCAR but not proof"

        rows.append({
            "missing_column": col,
            "missing_percent": round(miss_pct[col], 2),
            "best_related_feature": best_feature,
            "relation_type": best_type,
            "score": round(best_score, 4),
            "interpretation": interpretation,
        })

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "missingness_mechanism_check.csv", index=False)
    return result


def run_eda(df: pd.DataFrame, out_dir: Path = OUT_DIR) -> pd.DataFrame:
    """Run compact EDA and return the processed dataset snapshot."""
    out_dir.mkdir(exist_ok=True, parents=True)

    _plot_missingness(df, out_dir)
    outlier_summary = _plot_outliers(df, out_dir)
    processed = make_processed_dataset(df, out_dir)
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
    return processed
