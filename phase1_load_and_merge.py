"""
KDDM1 – Ice Hockey Talent Scouting
Phase I: Data Loading, Merging & Initial EDA
============================================
Run with:  python phase1_load_and_merge.py
Requires:  pandas, numpy, openpyxl, matplotlib, seaborn
           pip install pandas numpy openpyxl matplotlib seaborn
"""

import json
import pickle
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── adjust this if your files live somewhere else ──────────────────────────────
DATA_DIR = Path(".")   # folder containing all the data files
# ──────────────────────────────────────────────────────────────────────────────


# =============================================================================
# 1.  LOADING
# =============================================================================

def roman_to_int(s: str) -> int | None:
    """
    Convert a Roman numeral string to an integer.
    identity_card_0 stores international_id as Roman numerals (e.g. 'CCLXXV' → 275).
    This is a key preprocessing quirk worth noting in your presentation.
    """
    val = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    try:
        s = str(s).strip().upper()
        result, prev = 0, 0
        for ch in reversed(s):
            v = val.get(ch, 0)
            result += v if v >= prev else -v
            prev = v
        return result
    except Exception:
        return None


def load_identity_card_0(data_dir: Path) -> pd.DataFrame:
    """
    identity_card_0.tsv  –  two quirks:
      1. Column names are in Latin → rename to English.
      2. international_id is stored as Roman numerals (e.g. 'CCLXXV')
         → convert to integers so we can join with the other tables.
    Both identity cards carry the same info; we'll merge them to get a
    single clean identity table.
    """
    latin_to_english = {
        "numerus_internationalis_ad_identitatem": "international_id",
        "numerus_identificationis_ad_medicinam":  "medical_id",
        "praenomen":   "first_name",
        "cognomen":    "last_name",
        "sexus":       "gender",
        "aetas_annorum": "age",
        "urbs_natalis":  "birth_city",
        "natio":         "nationality",
    }
    df = pd.read_csv(data_dir / "identity_card_0.tsv", sep="\t")
    df.rename(columns=latin_to_english, inplace=True)

    # Convert Roman numerals to int (⚠ key preprocessing step for slide!)
    df["international_id"] = df["international_id"].apply(roman_to_int)

    print(f"[identity_card_0]  {df.shape[0]:,} rows  |  {df.shape[1]} cols")
    print("  NOTE: international_id was stored as Roman numerals – converted to int.")
    return df


def load_identity_card_1(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "identity_card_1.csv")
    print(f"[identity_card_1]  {df.shape[0]:,} rows  |  {df.shape[1]} cols")
    return df


def load_performance(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "performance.tsv", sep="\t")
    # Fix the typo in the raw file
    df.rename(columns={"penality_minutes": "penalty_minutes"}, inplace=True)
    print(f"[performance]      {df.shape[0]:,} rows  |  {df.shape[1]} cols")
    return df


def load_medical(data_dir: Path) -> pd.DataFrame:
    """
    medical_information.xlsx has a junk row 0 (date/quote) and a split header
    across rows 1-2.  We read without a header and reconstruct it manually.
    Raw layout:
        row 0  → junk metadata string
        row 1  → [NaN, 'height', 'weight', ..., 'physician_signature']
        row 2  → ['medical_id', NaN, NaN, ...]
        row 3+ → actual data
    """
    raw = pd.read_excel(data_dir / "medical_information.xlsx", header=None)

    # Build the column name list from rows 1 and 2
    col_names = ["medical_id"] + raw.iloc[1, 1:].tolist()

    # Data starts at row 3
    df = raw.iloc[3:].reset_index(drop=True)
    df.columns = col_names

    # ── height: strip units, convert to float (metres) ──────────────────────
    def parse_height(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        # e.g. "1.79m"  or  "180.1135cm"
        if s.endswith("cm"):
            return float(s[:-2]) / 100
        if s.endswith("m"):
            return float(s[:-1])
        try:
            return float(s)
        except ValueError:
            return np.nan

    # ── weight: strip units, convert to float (kg) ──────────────────────────
    def parse_weight(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower().replace("kg", "")
        try:
            return float(s)
        except ValueError:
            return np.nan

    # ── body_fat_percentage: strip %-sign ────────────────────────────────────
    def parse_pct(val):
        if pd.isna(val):
            return np.nan
        try:
            return float(str(val).replace("%", "").strip())
        except ValueError:
            return np.nan

    # ── fitness_level: normalise capitalisation ───────────────────────────────
    FITNESS_MAP = {
        "excellent": "Excellent",
        "good":      "Good",
        "average":   "Average",
        "poor":      "Poor",
    }

    df["height"]              = df["height"].apply(parse_height)
    df["weight"]              = df["weight"].apply(parse_weight)
    df["body_fat_percentage"] = df["body_fat_percentage"].apply(parse_pct)
    df["fitness_level"]       = (
        df["fitness_level"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(FITNESS_MAP)
    )
    df["return_date"] = pd.to_datetime(df["return_date"], errors="coerce")

    print(f"[medical]          {df.shape[0]:,} rows  |  {df.shape[1]} cols")
    return df


def load_scouting_notes(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "hockey_scouting_notes.csv")
    print(f"[scouting_notes]   {df.shape[0]:,} rows  |  {df.shape[1]} cols")
    return df


def load_contracts(data_dir: Path) -> pd.DataFrame:
    """
    contracts_competition.pkl is corrupted – every byte >= 0x80 was replaced
    with the UTF-8 replacement character (EF BF BD), making recovery impossible.

    Strategy: attempt to load; on failure return an empty stub with the known
    column schema so the rest of the pipeline still runs.  Replace this stub
    with the real data once a clean copy of the file is available.
    """
    known_columns = [
        "international_id", "contracts_signed", "salary", "captain",
        "won_championship", "jersey_number", "draft_year",
        "number_of_previous_teams",
    ]
    path = data_dir / "contracts_competition.pkl"
    try:
        df = pd.read_pickle(path)
        print(f"[contracts]        {df.shape[0]:,} rows  |  {df.shape[1]} cols")
        return df
    except Exception:
        # Attempt byte-patching (replaces 0xEFBFBD → 0x80) – rarely works
        # when multiple distinct high-bytes were all mapped to the same value
        try:
            raw = path.read_bytes().replace(b"\xef\xbf\xbd", b"\x80")
            df = pickle.loads(raw)
            print(f"[contracts] (patched)  {df.shape[0]:,} rows  |  {df.shape[1]} cols")
            return df
        except Exception:
            print(
                "[contracts]  WARNING – file is corrupted and could not be loaded.\n"
                "             Returning empty stub with correct column schema.\n"
                "             Replace 'contracts_competition.pkl' with a clean copy."
            )
            return pd.DataFrame(columns=known_columns)


def load_moms_notes(path: str | Path) -> pd.DataFrame:
    """
    moms_notes.json  –  array of player objects with personal/lifestyle info.
    NOTE: the file is NOT in the project folder (too large to upload).
    Pass the correct path when calling this function.

    Columns expected:
        first_name, last_name, age, birth_city, school_grade,
        years_in_usa, eye_color, stuffed_animal_name, favourite_board_game,
        personal_notes, favourite_child_info, favourite_tv_show, favourite_food
    """
    path = Path(path)
    if not path.exists():
        print(
            f"[moms_notes]  WARNING – file not found at '{path}'.\n"
            "              Returning empty stub.  Update the path when the file is available."
        )
        return pd.DataFrame(columns=[
            "first_name", "last_name", "age", "birth_city", "school_grade",
            "years_in_usa", "eye_color", "stuffed_animal_name",
            "favourite_board_game", "personal_notes", "favourite_child_info",
            "favourite_tv_show", "favourite_food",
        ])

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.json_normalize(data)

    # Replace empty strings with NaN for consistency
    df.replace("", np.nan, inplace=True)

    print(f"[moms_notes]       {df.shape[0]:,} rows  |  {df.shape[1]} cols")
    return df


# =============================================================================
# 2.  MERGING
# =============================================================================

def merge_all(
    id0:      pd.DataFrame,
    id1:      pd.DataFrame,
    perf:     pd.DataFrame,
    medical:  pd.DataFrame,
    scouting: pd.DataFrame,
    contracts: pd.DataFrame,
    moms:     pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge strategy
    ──────────────
    • id0 and id1 are two versions of the same identity table.
      Combine them with an outer join (on international_id + medical_id) to
      maximise coverage, then coalesce duplicate columns.
    • All other tables join on international_id (or medical_id for medical).
    • outer merge throughout so no player is silently dropped.
    • moms_notes has no ID column – it matches on (first_name, last_name).
      This is fuzzy: if a clean version of the file includes an id, prefer that.
    """

    # ── 2a. Combine the two identity cards ───────────────────────────────────
    shared_keys = ["international_id", "medical_id"]
    identity = pd.merge(id0, id1, on=shared_keys, how="outer", suffixes=("_id0", "_id1"))

    # Coalesce columns that exist in both (first_name, last_name, gender, age, …)
    for col in ["first_name", "last_name", "gender", "age", "birth_city", "nationality"]:
        col0, col1 = f"{col}_id0", f"{col}_id1"
        if col0 in identity.columns and col1 in identity.columns:
            identity[col] = identity[col0].combine_first(identity[col1])
            identity.drop(columns=[col0, col1], inplace=True)

    print(f"\nIdentity combined: {identity.shape[0]:,} rows")

    # ── 2b. Add performance data ──────────────────────────────────────────────
    df = pd.merge(identity, perf, on="international_id", how="outer")
    print(f"After +performance:  {df.shape[0]:,} rows")

    # ── 2c. Add medical data ──────────────────────────────────────────────────
    df = pd.merge(df, medical, on="medical_id", how="outer")
    print(f"After +medical:      {df.shape[0]:,} rows")

    # ── 2d. Add scouting notes ────────────────────────────────────────────────
    df = pd.merge(df, scouting, on="international_id", how="outer")
    print(f"After +scouting:     {df.shape[0]:,} rows")

    # ── 2e. Add contracts (stub if corrupted) ─────────────────────────────────
    if not contracts.empty:
        df = pd.merge(df, contracts, on="international_id", how="outer")
        print(f"After +contracts:    {df.shape[0]:,} rows")
    else:
        print("Contracts skipped (empty stub).")

    # ── 2f. Add mom's notes (match on name) ───────────────────────────────────
    if not moms.empty:
        df = pd.merge(
            df, moms,
            on=["first_name", "last_name"],
            how="left",     # left: keep all players, add mom info where matched
            suffixes=("", "_mom"),
        )
        # Coalesce age / birth_city if both sources have them
        for col in ["age", "birth_city"]:
            col_mom = f"{col}_mom"
            if col_mom in df.columns:
                df[col] = df[col].combine_first(df[col_mom])
                df.drop(columns=[col_mom], inplace=True)
        print(f"After +moms_notes:   {df.shape[0]:,} rows")
    else:
        print("Moms notes skipped (empty stub).")

    return df


# =============================================================================
# 3.  INITIAL EDA
# =============================================================================

def run_eda(df: pd.DataFrame, out_dir: Path = Path("eda_plots")) -> None:
    out_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("BASIC INFO")
    print("=" * 60)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB\n")

    # ── 3a. Missing values ────────────────────────────────────────────────────
    print("=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    miss_df = pd.DataFrame({"missing_count": missing, "missing_%": missing_pct})
    miss_df = miss_df[miss_df["missing_count"] > 0].sort_values("missing_%", ascending=False)
    print(miss_df.to_string())

    # Plot top-20 missing
    fig, ax = plt.subplots(figsize=(10, 6))
    top20 = miss_df.head(20)
    sns.barplot(x=top20["missing_%"], y=top20.index, ax=ax, color="steelblue")
    ax.set_title("Top 20 columns by % missing values")
    ax.set_xlabel("Missing (%)")
    plt.tight_layout()
    fig.savefig(out_dir / "missing_values.png", dpi=150)
    plt.close()
    print(f"\n→ Saved: {out_dir / 'missing_values.png'}")

    # ── 3b. Data types ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DATA TYPES")
    print("=" * 60)
    print(df.dtypes.value_counts())

    # ── 3c. Numeric summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("NUMERIC SUMMARY (first 10 columns)")
    print("=" * 60)
    print(df.describe().T.head(10).to_string())

    # ── 3d. Outlier detection – IQR method ───────────────────────────────────
    print("\n" + "=" * 60)
    print("OUTLIER COUNTS (IQR method, per numeric column)")
    print("=" * 60)
    numeric_cols = df.select_dtypes(include="number").columns
    outlier_report = {}
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        if n_out > 0:
            outlier_report[col] = n_out
    out_series = pd.Series(outlier_report).sort_values(ascending=False)
    print(out_series.to_string())

    # ── 3e. Correlation heatmap (performance features) ────────────────────────
    perf_cols = [
        "goals", "assists", "num_of_shots", "shot_speed", "shooting_percentage",
        "time_on_ice", "puck_touches", "passes_completed", "pass_completion_rate",
        "penalty_minutes", "games_missed_due_to_injury",
    ]
    perf_cols = [c for c in perf_cols if c in df.columns]
    if perf_cols:
        fig, ax = plt.subplots(figsize=(12, 9))
        corr = df[perf_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5, ax=ax,
        )
        ax.set_title("Correlation – Performance Features")
        plt.tight_layout()
        fig.savefig(out_dir / "correlation_performance.png", dpi=150)
        plt.close()
        print(f"\n→ Saved: {out_dir / 'correlation_performance.png'}")

    # ── 3f. Age distribution ──────────────────────────────────────────────────
    if "age" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        df["age"].dropna().plot.hist(bins=30, ax=ax, color="steelblue", edgecolor="white")
        ax.set_title("Age distribution")
        ax.set_xlabel("Age")
        plt.tight_layout()
        fig.savefig(out_dir / "age_distribution.png", dpi=150)
        plt.close()
        print(f"→ Saved: {out_dir / 'age_distribution.png'}")

    print("\n✓ EDA complete – plots written to:", out_dir.resolve())


# =============================================================================
# 4.  MAIN
# =============================================================================

if __name__ == "__main__":

    print("Loading files …\n")
    id0       = load_identity_card_0(DATA_DIR)
    id1       = load_identity_card_1(DATA_DIR)
    perf      = load_performance(DATA_DIR)
    medical   = load_medical(DATA_DIR)
    scouting  = load_scouting_notes(DATA_DIR)
    contracts = load_contracts(DATA_DIR)

    # ── Update this path once you have the file ───────────────────────────────
    MOMS_PATH = DATA_DIR / "moms_notes.json"
    moms = load_moms_notes(MOMS_PATH)
    # ─────────────────────────────────────────────────────────────────────────

    print("\nMerging …")
    df = merge_all(id0, id1, perf, medical, scouting, contracts, moms)

    print(f"\n✓ Master DataFrame shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("Columns:", df.columns.tolist())

    print("\nRunning EDA …")
    run_eda(df)

    # Optional: save the merged DataFrame for later notebooks
    df.to_csv("merged_dataset.csv", index=False)
    print("\n✓ Saved merged_dataset.csv")
