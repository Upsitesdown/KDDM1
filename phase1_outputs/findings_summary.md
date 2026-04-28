# Phase I – Most important findings

## Integration / preprocessing
- 7 source files in 5 formats merged into 10,000 rows × 68 cols.
- Two identity tables with overlapping IDs (Roman numerals + medical IDs); coalesced after rename.
- `moms_notes.json` had no ID column; matched on `(first_name, last_name)`, ambiguous duplicates resolved by closest `age`.

## Missingness (largest issues)
- `favourite_tv_show`: 100.0% missing (text).
- `favourite_food`: 100.0% missing (text).
- `medical_information`: 85.18% missing (text).
- Strategy is documented per column in `missingness_strategy.csv`; values are NOT yet imputed.

## Outliers / domain validity
- `age_in_years`: 130 domain violations; raw range [18.00, 455.00].
- `age`: 129 domain violations; raw range [18.00, 455.00].
- `height`: 61 domain violations; raw range [1.32, 3.09].
- These are likely data-entry errors (e.g. age 455, height > 3 m) and should be set to NaN before Phase II.
- Distinguish: missing vs unknown vs domain-invalid vs rare-but-plausible values.

## Strongest linear relationships
- `passes_completed` ↔ `pass_completion_rate`: |r| = 0.75.
- `puck_touches` ↔ `passes_completed`: |r| = 0.71.
- `height` ↔ `body_fat_percentage`: |r| = 0.70.
- Pearson is **linear only**, sensitive to outliers, and **does not imply causation**.
- Highly correlated counters (shots / attempts / passes) indicate multicollinearity to handle later.

## Recommended next steps before Phase II
- Apply domain-rule clipping → NaN, then impute per type AFTER the train/test split.
- Decide multicollinearity handling for redundant counters.
- Treat free-text columns separately; not direct model features.