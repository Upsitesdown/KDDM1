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
- We cannot prove from Phase I whether missingness is MCAR, MAR, or NMAR; imputation is proposed cautiously by feature type.
- Missingness mechanism check: columns whose missingness is related to observed features are treated as MAR-like candidates. If no relation is found, this is only consistent with MCAR, not proof. NMAR cannot be proven from observed values alone.

## Outliers / domain validity
- `age`: 129 domain violations; raw range [18.00, 455.00].
- `height`: 61 domain violations; raw range [1.32, 3.09].
- `body_fat_percentage`: 14 domain violations; raw range [13.99, 37.34].
- These are likely data-entry errors (e.g. age 455, height > 3 m) and should be replaced with NaN before Phase II.
- IQR outliers are only flagged as unusual; only domain-impossible values are candidates for conversion to NaN.
- Rare but plausible players should not be deleted automatically.
- Distinguish: missing vs unknown vs domain-invalid vs rare-but-plausible values.

## Strongest linear relationships
- `passes_completed` and `pass_completion_rate`: strong linear association (r = +0.75).
- `puck_touches` and `passes_completed`: strong linear association (r = +0.71).
- `height` and `body_fat_percentage`: strong linear association (r = -0.70).
- Pearson is **linear only**, sensitive to outliers, and **does not imply causation**.
- Highly correlated counters (shots / attempts / passes) indicate multicollinearity to handle later.

## Processed dataset
- Domain-impossible values are replaced with NaN in `phase1_processed_dataset.csv`, and corresponding `<column>_invalid_flag` columns preserve where invalid values occurred. IQR outliers are not removed automatically.

## Recommended next steps before Phase II
- Impute NaN values per type AFTER the train/test split.
- Decide multicollinearity handling for redundant counters.
- Treat free-text columns separately; not direct model features.