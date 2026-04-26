"""
KDDM1 – Ice Hockey Talent Scouting
Phase I: Data Loading, Merging & Initial EDA
"""

import json
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

DATA_DIR = Path("./Data-20260425")


def roman_to_int(s: str) -> int | None:
    """Convert Roman numeral string to integer."""
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
    """Load identity_card_0.tsv (Latin column names, Roman numerals for IDs)."""
    mapping = {
        "numerus_internationalis_ad_identitatem": "international_id",
        "numerus_identificationis_ad_medicinam": "medical_id",
        "praenomen": "first_name",
        "cognomen": "last_name",
        "sexus": "gender",
        "aetas_annorum": "age",
        "urbs_natalis": "birth_city",
        "natio": "nationality",
    }
    df = pd.read_csv(data_dir / "identity_card_0.tsv", sep="\t")
    df.rename(columns=mapping, inplace=True)
    df["international_id"] = df["international_id"].apply(roman_to_int)

    print("Success: load identity card 0")
    print(f"[identity_card_0]  {len(df):,} rows")
    return df


def load_identity_card_1(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "identity_card_1.csv")
    print("Success: load identity card 1")
    print(f"[identity_card_1]  {len(df):,} rows")
    return df


def load_performance(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "performance.tsv", sep="\t")
    df.rename(columns={"penality_minutes": "penalty_minutes"}, inplace=True)
    print("Success: load performance")
    print(f"[performance]      {len(df):,} rows")
    return df


def load_medical(data_dir: Path) -> pd.DataFrame:
    """Load medical info from Excel (skip junk rows, reconstruct header)."""
    raw = pd.read_excel(data_dir / "medical_information.xlsx", header=None)
    col_names = ["medical_id"] + raw.iloc[1, 1:].tolist()
    df = raw.iloc[3:].reset_index(drop=True)
    df.columns = col_names

    def parse_height(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        if s.endswith("cm"):
            return float(s[:-2]) / 100
        if s.endswith("m"):
            return float(s[:-1])
        try:
            return float(s)
        except ValueError:
            return np.nan

    def parse_weight(val):
        if pd.isna(val):
            return np.nan
        try:
            return float(str(val).strip().lower().replace("kg", ""))
        except ValueError:
            return np.nan

    def parse_pct(val):
        if pd.isna(val):
            return np.nan
        try:
            return float(str(val).replace("%", "").strip())
        except ValueError:
            return np.nan

    df["height"] = df["height"].apply(parse_height)
    df["weight"] = df["weight"].apply(parse_weight)
    df["body_fat_percentage"] = df["body_fat_percentage"].apply(parse_pct)

    fitness_map = {"excellent": "Excellent", "good": "Good", "average": "Average", "poor": "Poor"}
    df["fitness_level"] = df["fitness_level"].astype(str).str.strip().str.lower().map(fitness_map)
    df["return_date"] = pd.to_datetime(df["return_date"], errors="coerce")

    print("Success: load medical")
    print(f"[medical]          {len(df):,} rows")
    return df


def load_scouting_notes(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "hockey_scouting_notes.csv")
    print("Success: load scouting notes")
    print(f"[scouting_notes]   {len(df):,} rows")
    return df


def load_contracts(data_dir: Path) -> pd.DataFrame:
    df = pd.read_pickle(data_dir / "contracts_competition.pkl")
    print("Success: load contracts")
    print(f"[contracts]        {len(df):,} rows")
    return df


def load_moms_notes(path: str | Path) -> pd.DataFrame:
    """Load mom's notes from JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    df.replace("", np.nan, inplace=True)
    print(f"Success: load moms notes ({len(df):,} rows)")
    return df


def merge_all(id0, id1, perf, medical, scouting, contracts, moms) -> pd.DataFrame:
    """Merge all datasets into a single master table."""
    # Merge identity tables
    identity = pd.merge(id0, id1, on=["international_id", "medical_id"], how="outer", suffixes=("_id0", "_id1"))

    # Coalesce duplicate columns
    for col in ["first_name", "last_name", "gender", "age", "birth_city", "nationality"]:
        col0, col1 = f"{col}_id0", f"{col}_id1"
        if col0 in identity.columns and col1 in identity.columns:
            identity[col] = identity[col0].combine_first(identity[col1])
            identity.drop(columns=[col0, col1], inplace=True)

    print(f"After merging identity: {len(identity):,} rows")

    # Add other tables
    df = identity
    df = pd.merge(df, perf, on="international_id", how="outer")
    print(f"After adding performance: {len(df):,} rows")

    df = pd.merge(df, medical, on="medical_id", how="outer")
    print(f"After adding medical: {len(df):,} rows")

    df = pd.merge(df, scouting, on="international_id", how="outer")
    print(f"After adding scouting: {len(df):,} rows")

    df = pd.merge(df, contracts, on="international_id", how="outer")
    print(f"After adding contracts: {len(df):,} rows")

    # Merge moms notes with intelligent name+age matching
    # For ambiguous names (appear >1 time), pick closest age match
    matched_rows = []
    for _, main_row in df.iterrows():
        fname = main_row['first_name']
        lname = main_row['last_name']
        main_age = main_row['age']

        moms_matches = moms[(moms['first_name'] == fname) & (moms['last_name'] == lname)]

        if len(moms_matches) > 0:
            if len(moms_matches) == 1:
                # Unambiguous: use it
                best_match = moms_matches.iloc[0]
            else:
                # Ambiguous: pick closest age
                if pd.notna(main_age) and main_age < 100:
                    valid_ages = moms_matches[moms_matches['age'].notna() & (moms_matches['age'] < 100)]
                    if len(valid_ages) > 0:
                        age_diffs = (valid_ages['age'] - main_age).abs()
                        best_match = valid_ages.loc[age_diffs.idxmin()]
                    else:
                        best_match = moms_matches.iloc[0]
                else:
                    best_match = moms_matches.iloc[0]

            # Add moms columns to main row
            for col in moms.columns:
                if col not in ['first_name', 'last_name']:
                    main_row[col] = best_match[col]

        matched_rows.append(main_row)

    df = pd.DataFrame(matched_rows)
    print(f"After adding moms notes: {len(df):,} rows")

    return df


def run_eda(df: pd.DataFrame, out_dir: Path = Path("eda_plots")) -> None:
    """Run exploratory data analysis and save plots."""
    out_dir.mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    print(f"Shape: {len(df):,} rows × {len(df.columns)} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB\n")

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    miss_df = pd.DataFrame({"count": missing, "percent": missing_pct})
    miss_df = miss_df[miss_df["count"] > 0].sort_values("percent", ascending=False)
    print("Missing values (top 20):")
    print(miss_df.head(20).to_string())

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=miss_df.head(20)["percent"], y=miss_df.head(20).index, ax=ax, color="steelblue")
    ax.set_title("Missing Values")
    ax.set_xlabel("Percent (%)")
    plt.tight_layout()
    fig.savefig(out_dir / "missing_values.png", dpi=150)
    plt.close()
    print(f"\nSaved: missing_values.png")

    # Data types
    print("\n" + "="*60)
    print("DATA TYPES")
    print("="*60)
    print(df.dtypes.value_counts())

    # Numeric summary
    print("\n" + "="*60)
    print("NUMERIC SUMMARY")
    print("="*60)
    print(df.describe().T.head(10).to_string())

    # Correlation heatmap
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
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation – Performance Features")
        plt.tight_layout()
        fig.savefig(out_dir / "correlation_performance.png", dpi=150)
        plt.close()
        print(f"Saved: correlation_performance.png")

    # Age distribution
    if "age" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        df["age"].dropna().plot.hist(bins=30, ax=ax, color="steelblue", edgecolor="white")
        ax.set_title("Age Distribution")
        ax.set_xlabel("Age")
        plt.tight_layout()
        fig.savefig(out_dir / "age_distribution.png", dpi=150)
        plt.close()
        print(f"Saved: age_distribution.png")

    print(f"\nEDA complete – plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    print("Loading data...\n")
    id0 = load_identity_card_0(DATA_DIR)
    id1 = load_identity_card_1(DATA_DIR)
    perf = load_performance(DATA_DIR)
    medical = load_medical(DATA_DIR)
    scouting = load_scouting_notes(DATA_DIR)
    contracts = load_contracts(DATA_DIR)
    moms = load_moms_notes(DATA_DIR / "moms_notes.json")

    print("\nMerging data...")
    df = merge_all(id0, id1, perf, medical, scouting, contracts, moms)
    print(f"\nFinal shape: {len(df):,} rows × {len(df.columns)} columns")

    print("\nRunning EDA...")
    run_eda(df)

    df.to_csv("merged_dataset.csv", index=False)
    print("\nSaved: merged_dataset.csv")
