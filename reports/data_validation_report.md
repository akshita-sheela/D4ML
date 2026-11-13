# Data Validation Report

Reference date: `2026-01-01`. Invalid rows are written to `data/errors/<table>_errors.csv` with an `error_reason` column (a row can violate several rules).

| Table | Total | Valid | Invalid | % invalid |
|---|---:|---:|---:|---:|
| users | 2010 | 1994 | 16 | 0.796 |
| items | 804 | 797 | 7 | 0.871 |
| interactions | 54529 | 52267 | 2262 | 4.148 |

## users

| Rule | Rows violating |
|---|---:|
| duplicate_user_id | 10 |
| invalid_age_range | 6 |

Warnings (rows kept):

- missing_age (imputed in preparation): 20
- missing_country (imputed in preparation): 21

## items

| Rule | Rows violating |
|---|---:|
| duplicate_item_id | 4 |
| invalid_release_year | 3 |

Warnings (rows kept):

- missing_category (imputed 'Unknown'): 8

## interactions

| Rule | Rows violating |
|---|---:|
| exact_duplicate_row | 535 |
| orphan_item_id | 371 |
| orphan_user_id | 327 |
| invalid_rating_range | 270 |
| future_timestamp | 162 |
| missing_rating | 161 |
| invalid_timestamp_format | 161 |
| invalid_dtype_user_id | 109 |
| invalid_dtype_rating | 108 |
| timestamp_before_signup | 107 |
