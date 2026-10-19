# Workflow: Temporal Behavioral Analysis for Mental Health Risk Detection

## 1. Workflow summary

For the full research rationale, reported metrics, statistical model, limitations, and data dictionary, see [`METHODOLOGY.md`](METHODOLOGY.md).

This document describes the order and purpose of the major stages in the notebook. The project workflow includes a local preprocessing step in VS Code, copying prepared CSVs into Google Drive, and then running the Colab notebook. This is a workflow map, not a guarantee that every stage runs from a clean GitHub clone: the standalone VS Code preprocessing script(s), all source/prepared data, checkpoints, and intermediate row-level artifacts are not included in the public repository files inspected for this update.

**Interpretation boundary:** the outputs are model-estimated emotion/risk signals. They are not verified mental-health states, diagnoses, or validated predictions of future outcomes. Statistical associations are not causal effects.

## 2. Inputs and outputs at a glance

| Stage | Primary inputs | Outputs / artifacts | Compute type |
|---|---|---|---|
| GoEmotions setup and inference | `data/processed/rmhd_emotion_input.csv`; `SamLowe/roberta-base-go_emotions` | `results/predictions/goemotions/checkpoints/checkpoint_*.csv`; `results/predictions/goemotions/rmhd_goemotions_predictions.csv` | Transformer inference; expensive |
| Prediction quality and alignment | Combined prediction CSV, prepared input | Integrity checks; probability diagnostics; 200-row alignment check with `random_state=42` | Diagnostics; alignment check performs additional inference |
| Text-length/truncation analysis | Input text, GoEmotions tokenizer, saved predictions | Token-length table and truncation diagnostics | Tokenization plus statistics; selected sensitivity cells may rerun inference |
| Emotion distribution and stability diagnostics | Emotion probabilities and timestamps | Correlation, variability, ICC, baseline stability, shuffle, monthly drift, and subreddit diagnostics | Statistical analysis |
| Temporal feature engineering | Post timestamps and emotion probabilities | `temporal_features_90d_180d.csv`, `post_level_temporal_90d_180d.csv`, configuration JSON files | Aggregation and statistical calculations |
| C-SSRS classifier | `data/CSSRS/Copy of cssrs_train.parquet`, `Copy of cssrs_val.parquet`, `Copy of cssrs_test.parquet`; `distilbert-base-uncased` | Trained model/checkpoint, validation/test predictions and metrics under `models/` | Fine-tuning and inference; expensive |
| Leakage audit | C-SSRS split files | Exact-match, shared-user and near-duplicate audit tables | Text normalization, hashing, TF-IDF similarity; no model training |
| Temporal risk analysis | E2 predictions, post-level temporal data, user-level feature tables | User-level risk changes, emotion–risk correlations, regression and sensitivity results | Statistical analysis |
| Reporting | Aggregate results and saved statistical tables | `significant_temporal_emotion_summary.csv`, `consolidated_project_results.csv`, figures | Summary/visualization |
| Bootstrap and robustness | Combined user-level feature table | Bootstrap CIs, Cook's-excluded and length-adjusted regression tables | Statistical computation; bootstrap uses 5,000 requested resamples |

## 3. Detailed stages

### Stage 0 — Local preprocessing and Drive handoff

The author reports running preprocessing scripts in VS Code, producing CSV files, and copying those CSVs into Google Drive for use by Colab. The current notebook then reads prepared inputs such as `data/processed/rmhd_emotion_input.csv` and `data/processed/rmhd_680_clean.csv`. The older `NLP(1).ipynb` also reads `rmhd_688_selected.csv`, but that belongs to a legacy workflow with different cohort counts and scoring logic; it is not interchangeable with the current cleaned notebook's 680-author corpus.

**Reproducibility boundary:** the standalone local VS Code script(s) were not available among the project files inspected for this update. Do not claim a fresh clone can regenerate the prepared CSVs until those exact scripts are added and verified. When adding them, document their raw inputs, output filenames, row counts, column schemas, and execution command. Do not publish sensitive row-level data just to demonstrate the pipeline.


### Stage A — Load prepared RMHD input and initialize GoEmotions

**Notebook headings:** `Load Dataset and Initialize GoEmotions`, `GoEmotions Inference`, `Export GoEmotions Predictions`.

- Input: `data/processed/rmhd_emotion_input.csv`.
- Required columns explicitly checked by the notebook: `post_id` and `text`; the code also expects `author` for unique-user counts. Available metadata columns may include `post_id`, `author`, `timestamp`, `created_utc`, `subreddit`, and `score`.
- Model identifier: `SamLowe/roberta-base-go_emotions`.
- The model returns 28 emotion probabilities, using sigmoid over logits.
- Inference parameters: `BATCH_SIZE = 32`, `CHECKPOINT_SIZE = 1000`, `MAX_LENGTH = 512`.
- Existing checkpoints are detected by `post_id` so interrupted inference can resume. The export step combines checkpoint files, removes duplicate `post_id` rows keeping the last, sorts by `post_id`, and writes `results/predictions/goemotions/rmhd_goemotions_predictions.csv`.
- Expected validation: 83,516 rows, 83,516 unique `post_id` values, 28 emotion columns, finite probabilities in `[0, 1]`.
- Cost: expensive when all rows must be inferred. The notebook does not record a stable wall-clock benchmark; runtime depends on available hardware and whether cached checkpoints exist.

The checkpoint and prediction files contain row-level identifiers and should not be committed to a public repository.

### Stage B — Prediction validation and alignment

**Notebook headings:** `GoEmotions Prediction Validation`, `Prediction Alignment Verification`, `Dataset and Timestamp Integrity`.

Checks include prediction uniqueness, emotion-column count, missing values, finite values and probability range. The alignment check samples `200` rows with `random_state=42` and reruns inference with max length `512` to compare against saved predictions. This is a validation operation, not a full model retraining step.

### Stage C — Text length, truncation, and prediction diagnostics

**Notebook headings:** `Text Length and Truncation Analysis`, `Prediction Quality Diagnostics`, `Relationship Between Text Length and Emotion Scores`, `Distribution of Truncated Posts Across Users`, `Calculate Token Lengths for Truncation Analysis`, `User-Level Truncation Analysis`, `Truncation Sensitivity Analysis`.

The notebook loads the GoEmotions tokenizer, measures token lengths, and checks posts exceeding the `512`-token inference limit. A token-length output is referenced as `results/predictions/goemotions/token_lengths.csv`. Some later cells compare token-length changes with risk-score changes or rerun predictions for affected posts. These are diagnostics; they do not establish that length causes the model scores.

The documented output paths are the paths written in the relevant cells; successful completion of every optional sensitivity cell is not guaranteed by the notebook alone.

### Stage D — Emotion quality and temporal diagnostics

The notebook includes analyses of emotion probability distributions, emotion–emotion correlations, within-user and between-user variability, intraclass correlation, temporal baseline stability, shuffled-baseline comparisons, monthly drift, subreddit composition/metadata quality, and truncation-related differences.

These stages mostly consume saved predictions and timestamps. They are descriptive diagnostics and do not by themselves train a model or validate clinical meaning.

### Stage E — Freeze configuration and engineer temporal features

**Notebook headings:** `Temporal Window Coverage Analysis`, `Temporal Feature Engineering — Window Coverage`, `User-Level Temporal Emotion Deviations`, `Freeze Analysis Configuration`, `Temporal Feature Quality Checks`, `Build Post-Level Temporal Dataset`.

The primary temporal design recorded in the notebook is:

| Parameter | Value in notebook | Meaning |
|---|---:|---|
| Anchor | Each user's latest observed post | Relative time origin for that user's windows |
| Recent window | 90 days | Posts after the recent-window start and through the anchor, as coded |
| Baseline window | 180 days preceding the recent window | Historical comparison window, from 270 to 90 days before the anchor |
| Minimum recent posts | 5 | Eligibility threshold |
| Minimum baseline posts | 15 | Eligibility threshold |
| Strict sensitivity minimum recent posts | 8 | Alternative eligibility threshold |
| Strict sensitivity minimum baseline posts | 20 | Alternative eligibility threshold |
| Emotion probability columns | 28 | Model outputs summarized by window |
| Multiple-testing method | Benjamini–Hochberg FDR | Used for specified families of statistical tests |
| FDR alpha | 0.05 | Threshold in frozen configuration |
| Token truncation limit | 512 | GoEmotions input limit used in inference |
| Truncation-sensitive user criterion | Share at least 0.10 truncated posts | Configuration threshold for targeted sensitivity work |

The primary temporal feature output is `results/temporal_features_90d_180d.csv`. Its columns include `author`, `recent_posts`, `baseline_posts`, `baseline_<emotion>`, `recent_<emotion>`, and `delta_<emotion>` for the 28 model emotion outputs. The post-level output is `results/post_level_temporal_90d_180d.csv`. Both are row-level/user-level and must remain private.

The code drops rows with invalid timestamps after parsing them as UTC, and only retains users meeting the minimum-post thresholds. Temporal cells do not express every interval boundary identically. The intended analysis definition is `[0,90)` recent and `[90,270)` baseline; a strict reproduction should follow the executed feature-building cell and note the edge convention used.

### Stage F — Statistical nulls and population-level temporal tests

The notebook contains statistical null calibration, within-user permutation and autocorrelation-preserving block/circular-shift null procedures, population tests, alternative window analyses, and sensitivity checks. Exact iteration counts for all such tests are not consistently summarized in a single parameter table.

The notebook contains several null/permutation procedures with settings defined in their own cells; it does not provide a single consolidated parameter ledger for every procedure. The bootstrap procedure described below is explicitly recorded as 5,000 draws with seed 20261019.

### Stage G — C-SSRS-related classifier training and evaluation

**Notebook headings:** `Inspect C-SSRS Splits` and subsequent tokenizer, training, evaluation, E2 prediction, and comparison cells.

- Split files referenced by the notebook:
  - `data/CSSRS/Copy of cssrs_train.parquet`
  - `data/CSSRS/Copy of cssrs_val.parquet`
  - `data/CSSRS/Copy of cssrs_test.parquet`
- Base model: `distilbert-base-uncased`.
- Training seed: `42`.
- Training batch size: `8`; evaluation batch size: `16`.
- Learning rate: `2e-5`.
- Max token length in the classifier tokenization/training path: `256`.
- Other E2 inference/evaluation cells use max length `512`; do not assume one setting applies to every cell.
- The notebook contains training cells and later cells that load trained checkpoints for inference/evaluation. The training cells set `EPOCHS = 8`; the saved test metrics should be associated with the `e2_test_predictions.csv` artifact generated by the test-inference section. Checkpoint directory names vary by section.
- Split sizes reported in the notebook: train `1,815`, validation `482`, test `443`, total `2,740`.

Recorded metrics in the notebook's result history:

| Split | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Validation | 0.6515 | 0.3374 | 0.6450 |
| Test | 0.6388 | 0.5079 | 0.6231 |

The notebook's analysis notes report under-detection of `behavior` and `attempt` by default high-risk predictions. Do not use the test set for model/threshold tuning. Confirm that the saved metrics correspond to the checkpoint and run being documented before publication.

### Stage H — Cross-split leakage audits

**Notebook headings:** `Near-Duplicate Leakage Audit`, `Inspect C-SSRS Splits`, `Cross-Split Leakage Audit`, and `C-SSRS Data Leakage Audit-conclusion`.

The audit checks normalized exact-text matches, shared users, and near-duplicate candidate pairs using character-level TF-IDF and cosine similarity with threshold `≥ 0.85`. The recorded result states that no exact normalized-text matches, shared users, or candidate near-duplicate pairs were found across the tested splits. This conclusion applies only to the specific matching methods and threshold used; it does not prove all forms of leakage are impossible. Do not publish row-level audit tables or examples.

### Stage I — Risk-score and temporal association analysis

The notebook defines an E2-derived risk score using the sum of the `behavior` and `attempt` class probabilities in the recorded project notes. The temporal analysis joins this score with baseline/recent windows and six emotion changes: nervousness, fear, disappointment, neutral, embarrassment, and curiosity.

The primary rate-adjusted OLS model uses `mean_risk_change` as outcome and seven predictors: the six emotion deltas plus `relative_posting_rate_change`. The relative posting-rate change is computed from recent posts per 90 days and baseline posts per 180 days, divided by baseline posting rate. The primary regression uses HC3 robust standard errors. The multiple-testing correction for the seven predictor p-values uses Benjamini–Hochberg in the specified regression cells.

Recorded primary-model result: disappointment coefficient `0.42338`, HC3 robust p-value `0.000450`, BH-adjusted p-value `0.00315`. The recorded model has `n=258`, R-squared `0.1622`, and adjusted R-squared `0.1388`. This is an exploratory association in model-derived measurements; it does not show causation or clinical validity.

Sensitivity analyses include adding change in mean token length and excluding influential observations based on Cook's distance. The full-sample model is the primary result; the influential-observation-excluded model is a sensitivity analysis and should not replace the primary result solely because its significance pattern differs.

### Stage J — Summary outputs and bootstrap uncertainty

The notebook creates aggregate result files including `significant_temporal_emotion_summary.csv` and `consolidated_project_results.csv`. It also creates user-level or row-level files that should not be published, including `user_level_e2_temporal_risk_changes.csv`, `temporal_user_emotion_differences.csv`, and per-post prediction/diagnostic outputs.

The bootstrap cell resamples users with replacement for `B = 5000` requested draws and uses `np.random.default_rng(20261019)`. It saves `user_bootstrap_emotion_risk_correlations.csv` and `user_bootstrap_disappointment_coefficient.csv`. These are computed from the combined user-level feature table. Confirm successful completion and valid draw counts in the saved output before quoting the intervals.

## 4. Output privacy classification

| Output type | Examples | Public release guidance |
|---|---|---|
| Aggregate project summary | `consolidated_project_results.csv` | Candidate for publication after checking all rows and interpretations. |
| Aggregate emotion summary | `significant_temporal_emotion_summary.csv` | Candidate for publication after confirming no user-level rows and verifying statistics. |
| Per-post emotion predictions | `rmhd_goemotions_predictions.csv`, `checkpoint_*.csv` | Do not publish. Contains post identifiers/metadata and model scores. |
| Per-user temporal features | `temporal_features_90d_180d.csv`, `combined_temporal_behavioural_features.csv`, `final_user_risk_indicator_summary.csv` | Do not publish. Individual-level features remain potentially sensitive even if author IDs are removed. |
| Raw/processed text | `rmhd_680_clean.csv`, `rmhd_emotion_input.csv` | Do not publish as-is. Contains text and identifiers/metadata. |
| Model checkpoints and optimizer state | Local `models/` artifacts | Do not publish unless redistribution is reviewed and approved. |
| Aggregate metrics and figures | Classification metrics, regression summary, approved aggregate plots | Can be considered after checking provenance, small-cell disclosure, and consistency with the notebook. |

## 5. Reproduction checklist

- [ ] Obtain each dataset from the source page and record the version/revision and terms.
- [ ] Create the project directory and place files at the exact paths used in the notebook.
- [ ] Reproduce or obtain the processed RMHD intermediate files using documented transformation code.
- [ ] Verify the C-SSRS parquet split schemas and row counts.
- [ ] Confirm model identifiers and checkpoint directories.
- [ ] Run data and schema checks before inference/training.
- [ ] Confirm GoEmotions prediction count, unique IDs, 28 emotion columns, finite values and `[0, 1]` range.
- [ ] Confirm C-SSRS split sizes and metrics for the exact checkpoint used.
- [ ] Confirm temporal eligibility counts and window settings.
- [ ] Re-run or verify the aggregate statistical tables and multiple-testing correction.
- [ ] Keep row-level data, raw text, user identifiers and model checkpoints out of public GitHub.

## 6. Reproducibility and release notes

The current repository is documentation plus a notebook, not a fully self-contained data release. The notebook consumes locally prepared data and saved artifacts. To reproduce the complete analysis from raw sources, a researcher needs the upstream preparation script and exact source versions, neither of which is fully specified in the current notebook. This is a documented project boundary, not a guarantee of end-to-end reproduction.

The older `NLP(1).ipynb` is a legacy version: it records 83,895 posts / 688 users and a chronological 80/20 split with a weighted composite scoring approach. Those settings are not the current cleaned notebook's experiment and must not be mixed with its 83,516-post / 680-author cohort, E2 `behavior + attempt` score, or 90/180-day temporal design.

Before a research release, retain a local provenance record containing dataset record/version and terms; Python/library versions; model revisions and checkpoint hashes; the exact upstream preprocessing script; row counts before/after filters; and the artifact versions corresponding to reported results. Keep any access paths or identifiers in that private record out of Git.

Public release checklist:

- [ ] README links match the actual notebook filename in the repository.
- [ ] `requirements.txt` is described as a starter dependency list unless a clean-environment lockfile has been tested.
- [ ] No raw text, author identifiers, post IDs, per-post predictions, per-user tables, or private audit rows are committed.
- [ ] Both aggregate CSVs are inspected for accidental row-level fields and checked against the cited notebook run.
- [ ] Results are described as exploratory model-derived associations, not diagnoses, prevalence estimates, or causal effects.
- [ ] Results from the legacy notebook are kept distinct from the current cleaned notebook results.


## Local preprocessing status

The local-VS-Code-to-Google-Drive step is part of the described project workflow. This documentation records it explicitly, but does not invent script names or transformations. The repository becomes fully reproducible from raw source data only after the original local preprocessing script(s) are committed (or otherwise made available with exact instructions) and checked against the current prepared-input counts and schemas.
