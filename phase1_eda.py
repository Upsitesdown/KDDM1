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


def _plot_missingness(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
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
    return miss_tbl


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


def _write_findings(miss_tbl: pd.DataFrame, outliers: pd.DataFrame,
                    pairs: pd.DataFrame, df: pd.DataFrame, out_dir: Path) -> None:
    biggest_miss = miss_tbl[miss_tbl["missing_percent"] > 0].head(3)
    worst = outliers[outliers["column"] != "age_in_years"].sort_values("domain_violations", ascending=False).head(3)
    strongest = pairs.head(3)

    lines = ["# Phase I – Most important findings\n"]
    lines.append("## Integration / preprocessing")
    lines.append(f"- 7 source files in 5 formats merged into {len(df):,} rows × {len(df.columns)} cols.")
    lines.append("- Two identity tables with overlapping IDs (Roman numerals + medical IDs); coalesced after rename.")
    lines.append("- `moms_notes.json` had no ID column; matched on `(first_name, last_name)`, ambiguous duplicates resolved by closest `age`.\n")

    lines.append("## Missingness (largest issues)")
    for _, r in biggest_miss.iterrows():
        lines.append(f"- `{r['column']}`: {r['missing_percent']}% missing ({r['feature_type']}).")
    lines.append("- Strategy is documented per column in `missingness_strategy.csv`; values are NOT yet imputed.")
    lines.append("- We cannot prove from Phase I whether missingness is MCAR, MAR, or NMAR; imputation is proposed cautiously by feature type.")
    lines.append("- Missingness mechanism check: columns whose missingness is related to observed features are treated as MAR-like candidates. If no relation is found, this is only consistent with MCAR, not proof. NMAR cannot be proven from observed values alone.\n")

    lines.append("## Outliers / domain validity")
    for _, r in worst.iterrows():
        lines.append(f"- `{r['column']}`: {int(r['domain_violations'])} domain violations; raw range [{r['min']:.2f}, {r['max']:.2f}].")
    lines.append("- These are likely data-entry errors (e.g. age 455, height > 3 m) and should be replaced with NaN before Phase II.")
    lines.append("- IQR outliers are only flagged as unusual; only domain-impossible values are candidates for conversion to NaN.")
    lines.append("- Rare but plausible players should not be deleted automatically.")
    lines.append("- Distinguish: missing vs unknown vs domain-invalid vs rare-but-plausible values.\n")

    lines.append("## Strongest linear relationships")
    for _, r in strongest.iterrows():
        lines.append(f"- `{r['feature_a']}` and `{r['feature_b']}`: strong linear association (r\u2009=\u2009{r['pearson_r']:+.2f}).")
    lines.append("- Pearson is **linear only**, sensitive to outliers, and **does not imply causation**.")
    lines.append("- Highly correlated counters (shots / attempts / passes) indicate multicollinearity to handle later.\n")

    lines.append("## Recommended next steps before Phase II")
    lines.append("- Replace domain-impossible values with NaN, then impute per type AFTER the train/test split.")
    lines.append("- Decide multicollinearity handling for redundant counters.")
    lines.append("- Treat free-text columns separately; not direct model features.")
    (out_dir / "findings_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_slide_plan(out_dir: Path) -> None:
    md = """# Phase I – Slide Plan (Ice Hockey Talent Scouting)

Exactly 8 slides. Phase II / target variable is NOT used.

## Slide 1 — Approach
- Category: Approach
- Outputs: `merged_dataset.csv`, `missingness_summary.csv`
- Bullets:
  - 7 sources in 5 formats (TSV, CSV, XLSX, PKL, JSON).
  - Outer joins on `international_id` + `medical_id`; coalesced duplicates.
  - `moms_notes.json` has no ID → matched on `(first_name, last_name)`, ambiguity resolved by closest `age`.

## Slide 2 — Preprocessing I: Loading & Standardization
- Category: Preprocessing
- Outputs: `merged_dataset.csv`
- Bullets:
  - Latin → English column names in `identity_card_0.tsv`.
  - Roman numeral `international_id` → integer.
  - Typo fix: `penality_minutes` → `penalty_minutes`.
  - Empty strings in `moms_notes.json` → `NaN`.

## Slide 3 — Preprocessing II: Type & Unit Conversion
- Category: Preprocessing
- Outputs: `merged_dataset.csv`
- Bullets:
  - `height` parsed from cm/m strings → metres (float).
  - `weight` parsed from kg strings → float; `body_fat_percentage` from `%` strings → float.
  - `return_date` parsed → `datetime` (invalid → `NaT`).
  - `fitness_level` normalised; feature types classified (numeric / categorical / date / text / id_meta).

## Slide 4 — Relationships I: Numeric Correlations
- Category: Data relationships
- Outputs: `relationships_correlation_heatmap.png`, `top_correlations.csv`
- Bullets:
  - Pearson heatmap on numeric features after **domain filtering**.
  - Top |r| pairs flag possible multicollinearity.
  - Pearson is linear only and sensitive to remaining outliers.

## Slide 5 — Relationships II: Scatter & Grouped Views
- Category: Data relationships
- Outputs: `relationships_scatter_grouped.png`
- Bullets:
  - Age vs goals, weight vs shot speed (numeric scatter).
  - Time on ice by position (grouped boxplot).
  - Visual association only — no causal claim.

## Slide 6 — Cleaning I: Missing Values
- Category: Data cleaning / imputation
- Outputs: `missingness_top20.png`, `missingness_summary.csv`, `missingness_strategy.csv`, `missingness_mechanism_check.csv`
- Bullets:
  - Per-column missing count and percent quantified.
  - Strategy documented per type: drop / 'Unknown' category / median (post-split) / NaT + flag.
  - Missing values are NOT blindly imputed in Phase I.
  - Missingness mechanism (MCAR / MAR / NMAR) cannot be determined from Phase I; imputation is proposed cautiously.

## Slide 7 — Cleaning II: Outliers & Invalid Values
- Category: Data cleaning / imputation
- Outputs: `outliers_boxplots.png`, `outlier_summary.csv`
- Bullets:
  - IQR outliers + domain plausibility violations per numeric column.
  - Domain rules: e.g. age ∈ [15, 50], height ∈ [1.4, 2.3] m, percentages ∈ [0, 100], counters ≥ 0.
  - Plan: replace domain-impossible values with NaN; flag rather than delete rare-but-plausible values.
  - IQR outliers are flagged, not removed; only domain-impossible values are NaN candidates.

## Slide 8 — Most Important Findings / Discussion
- Category: Findings
- Outputs: `findings_summary.md`
- Bullets:
  - Biggest integration, missingness, and outlier issues.
  - Strongest associations + caveats (linear only, not causal).
  - Next cleaning steps before Phase II.
"""
    (out_dir / "phase1_slide_plan.md").write_text(md, encoding="utf-8")


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


TARGET_LIKE = {"target", "label", "y", "is_talent", "prospect", "prospect_grade"}


def validate_phase1_outputs(out_dir: Path, df: pd.DataFrame) -> None:
    """Print and save a Phase I validation checklist."""
    required = [
        "missingness_top20.png", "outliers_boxplots.png",
        "relationships_correlation_heatmap.png", "relationships_scatter_grouped.png",
        "missingness_summary.csv", "missingness_strategy.csv",
        "outlier_summary.csv", "top_correlations.csv",
        "findings_summary.md", "phase1_slide_plan.md",
    ]
    no_target = not (set(df.columns) & TARGET_LIKE)

    plan_text = (out_dir / "phase1_slide_plan.md").read_text(encoding="utf-8") \
        if (out_dir / "phase1_slide_plan.md").exists() else ""
    n_slides = sum(1 for line in plan_text.splitlines() if line.startswith("## Slide "))

    checks = [
        ("No obvious target column used in EDA", no_target),
        ("All required output files exist", all((out_dir / f).exists() for f in required)),
        ("Exactly 8 slide sections in phase1_slide_plan.md", n_slides == 8),
        ("Missingness analysis exists", (out_dir / "missingness_summary.csv").exists()),
        ("Missingness strategy exists", (out_dir / "missingness_strategy.csv").exists()),
        ("Outlier summary exists", (out_dir / "outlier_summary.csv").exists()),
        ("Relationship plots exist",
         (out_dir / "relationships_correlation_heatmap.png").exists()
         and (out_dir / "relationships_scatter_grouped.png").exists()),
        ("Findings summary exists", (out_dir / "findings_summary.md").exists()),
    ]

    lines = ["PHASE I VALIDATION CHECKLIST", "=" * 60]
    for label, ok in checks:
        lines.append(f"  [{'x' if ok else ' '}] {label}")
    lines.append("")
    lines.append(f"Missing files: {[f for f in required if not (out_dir / f).exists()]}")
    lines.append(f"Slide sections found: {n_slides}")
    lines.append("")
    lines.append("Notes (not auto-checked):")
    lines.append("  - Correlation heatmap uses domain-filtered copy (see _domain_filtered).")
    lines.append("  - Verify end-to-end run by checking all output files above exist.")
    text = "\n".join(lines)
    print("\n" + text)
    (out_dir / "validation_checklist.txt").write_text(text, encoding="utf-8")


def run_phase1_eda(df: pd.DataFrame, out_dir: Path = OUT_DIR) -> None:
    """Run compact EDA: graphs + summary CSVs + findings + slide plan + validation."""
    out_dir.mkdir(exist_ok=True, parents=True)

    miss_tbl = _plot_missingness(df, out_dir)
    outlier_summary = _plot_outliers(df, out_dir)
    top_pairs = _relationship_corr_and_pairs(df, out_dir)
    _plot_relationship_views(df, out_dir)
    analyze_missingness_mechanisms(df, out_dir)
    _write_findings(miss_tbl, outlier_summary, top_pairs, df, out_dir)
    _write_slide_plan(out_dir)

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

    validate_phase1_outputs(out_dir, df)
