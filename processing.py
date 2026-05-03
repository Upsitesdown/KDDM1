from pathlib import Path

import numpy as np
import pandas as pd

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


def domain_filtered(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with domain-impossible numeric values set to NaN."""
    out = df.copy()
    for col, (lo, hi) in DOMAIN_RANGES.items():
        if col in out.columns:
            if lo is not None:
                out.loc[out[col] < lo, col] = np.nan
            if hi is not None:
                out.loc[out[col] > hi, col] = np.nan
    return out


def make_processed_dataset(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Replace invalid values with NaN, add flags, and save the processed snapshot."""
    processed = df.copy()
    summary_rows = []

    for col, (lo, hi) in DOMAIN_RANGES.items():
        if col not in processed.columns or not pd.api.types.is_numeric_dtype(processed[col]):
            continue

        series = processed[col]
        mask = pd.Series(False, index=processed.index)
        if lo is not None:
            mask |= series < lo
        if hi is not None:
            mask |= series > hi
        mask &= series.notna()

        invalid_count = int(mask.sum())
        if invalid_count > 0:
            processed[f"{col}_invalid_flag"] = mask.astype(int)
            summary_rows.append({
                "column": col,
                "rule": f"valid range: {lo} to {hi}",
                "invalid_count": invalid_count,
                "invalid_percent": round(invalid_count / len(processed) * 100, 2),
                "min_before": float(series.min()),
                "max_before": float(series.max()),
            })
            processed.loc[mask, col] = np.nan

    for col in NON_NEGATIVE:
        if col in DOMAIN_RANGES or col not in processed.columns:
            continue
        if not pd.api.types.is_numeric_dtype(processed[col]):
            continue

        series = processed[col]
        mask = (series < 0) & series.notna()
        invalid_count = int(mask.sum())
        if invalid_count > 0:
            processed[f"{col}_invalid_flag"] = mask.astype(int)
            summary_rows.append({
                "column": col,
                "rule": "must be non-negative",
                "invalid_count": invalid_count,
                "invalid_percent": round(invalid_count / len(processed) * 100, 2),
                "min_before": float(series.min()),
                "max_before": float(series.max()),
            })
            processed.loc[mask, col] = np.nan

    processed.to_csv(out_dir / "phase1_processed_dataset.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "invalid_value_summary.csv", index=False)

    print(f"\nProcessed dataset: {len(processed)} rows, {len(processed.columns)} cols")
    print(f"Invalid values replaced with NaN: {sum(r['invalid_count'] for r in summary_rows)} across {len(summary_rows)} columns")
    return processed