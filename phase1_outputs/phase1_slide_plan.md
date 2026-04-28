# Phase I – Slide Plan (Ice Hockey Talent Scouting)

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
