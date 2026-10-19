# Setup Guide: Temporal Behavioral Analysis for Mental Health Risk Detection

## 1. Purpose and scope

See [`METHODOLOGY.md`](METHODOLOGY.md) for the research question, analysis design, model evaluation, statistical results, and data dictionary.

This repository contains an exploratory NLP research workflow for analyzing emotion-related signals and model-estimated risk scores in Reddit text over time. The notebook combines GoEmotions-based emotion inference, a C-SSRS-related text classifier, temporal feature construction, statistical association analyses, and robustness/data-quality checks.

**Safety statement:** This is not a clinical or diagnostic system. Model outputs are estimates derived from text, not verified mental-health states. Associations do not establish causation. Do not use the outputs to diagnose, label, rank, or make decisions about individuals.

## 2. What the public repository contains

The repository currently contains the notebook, README, dependency starter file, `.gitignore`, and selected aggregate result CSVs. Confirm exact filenames in the GitHub repository before using these instructions.

The repository deliberately should not contain:

- Raw Reddit posts or source dataset dumps.
- Usernames/author identifiers, post IDs, or other row-level data.
- Per-post emotion/risk predictions or per-user temporal feature tables.
- Model checkpoints, optimizer states, or other large trained artifacts unless distribution is explicitly approved.
- API keys, access tokens, credentials, or private Drive paths that grant access.

The notebook currently references files under `/content/drive/MyDrive/mental-health-analysis`, which is the original author's Google Drive location. A new user will not have access to that private directory. They must obtain the datasets and create their own project directory.

## 3. Source datasets and references

Use the source pages below as starting points. Before downloading or redistributing any material, read the dataset record, licence, terms of use, privacy/ethics guidance, and citation instructions. A public download page does not automatically grant permission to republish source rows or derived row-level data.

| Resource | Source link | Role in this project | Access / terms |
|---|---|---|---|
| Reddit Mental Health Dataset (RMHD) | https://zenodo.org/records/3941387 | Source corpus for timestamped Reddit post analysis. | Check the Zenodo record for its current files, licence, usage restrictions, and citation instructions before use. The source record is the starting point; the exact downloaded version and its licence/terms must be retained with the researcher’s local data provenance record because they are not embedded in the notebook. |
| Reddit C-SSRS Suicide Dataset | https://zenodo.org/records/2667859 | Source of text examples associated with C-SSRS-related categories for classifier development/evaluation. | Check the record for current files, terms, licence, and citation requirements. The exact downloaded version and its licence/terms are not recorded in the notebook; consult the source record and record the version used locally. |
| GoEmotions (Google Research) | https://github.com/google-research/google-research/tree/master/goemotions | Emotion-label dataset/resource. | Read the repository's licence and dataset documentation; cite the dataset paper. The notebook does not pin a repository commit/revision. |
| GoEmotions paper | https://arxiv.org/abs/2005.00547 | Dataset and benchmark description. | Cite in project documentation and any report. |
| GoEmotions Hugging Face dataset | https://huggingface.co/datasets/google-research-datasets/go_emotions | The notebook checks/uses the `google-research-datasets/go_emotions` dataset with configuration `simplified` in evaluation-related cells. | Confirm the precise revision and the data loader version when reproducing. |
| GoEmotions model used for inference | https://huggingface.co/SamLowe/roberta-base-go_emotions | The notebook's `MODEL_NAME` for GoEmotions inference is `SamLowe/roberta-base-go_emotions`. | Model card and terms should be checked before use. The model ID is recorded, but the original model revision/commit is not pinned. |
| C-SSRS instrument background | https://cssrs.columbia.edu/the-columbia-scale-c-ssrs/cssrs-for-research/ | Background about the instrument; not a replacement for the Reddit research dataset documentation. | Follow the instrument owner's terms. Do not describe this text classifier as administering or scoring a clinical C-SSRS assessment. |

## 4. Environment prerequisites

### Python and libraries

The notebook metadata identifies a Python 3 kernel. The notebook does **not** pin a Python minor version or exact package versions. It imports or uses, across cells, packages including:

- `pandas`, `numpy`
- `torch` (PyTorch)
- `transformers`
- `scipy`
- `scikit-learn` (including TF-IDF/similarity and metrics)
- `statsmodels`
- `matplotlib`
- `pyarrow` or another compatible Parquet engine for `.parquet` inputs
- IPython/Jupyter display utilities

The notebook does not record a complete tested Python/package matrix. The repository's `requirements.txt` is a starter list, not a lockfile; this documentation does not claim bit-for-bit environment reproducibility.

### Compute and runtime

- The notebook metadata requests a GPU runtime with a T4 accelerator in Colab, but a T4 is not guaranteed for every user.
- GoEmotions inference uses batch size `32`, checkpoint chunks of `1000` posts, and a maximum token length of `512`.
- The C-SSRS DistilBERT training cells use batch size `8`, evaluation batch size `16`, and max token length `256` in the tokenization/training path. Other E2 inference/evaluation cells use a max token length of `512`.
- Runtime depends on hardware, network/model download time, dataset version, and whether inference/training is repeated. The notebook does not record reproducible wall-clock or peak-memory benchmarks; runtime depends on hardware, network/model downloads, and whether saved outputs are reused.
- Full GoEmotions inference over the recorded 83,516-post table and classifier training can be expensive. Do not run all cells blindly.

## 5. Directory layout

The original notebook repeatedly defines the project root as `/content/drive/MyDrive/mental-health-analysis`. The following structure reflects paths and files referenced in the notebook; not every file is included in the public repository.

```text
mental-health-analysis/
├── configs/
│   ├── batch3_analysis_config.json
│   └── temporal_analysis_config.json       # referenced by analysis workflow; verify exact presence
├── data/
│   ├── processed/
│   │   ├── rmhd_680_clean.csv               # private/local intermediate; do not publish as-is
│   │   └── rmhd_emotion_input.csv            # private/local intermediate; do not publish as-is
│   └── CSSRS/
│       ├── Copy of cssrs_train.parquet
│       ├── Copy of cssrs_val.parquet
│       └── Copy of cssrs_test.parquet
├── models/
│   ├── e2_test_predictions.csv               # row-level; keep private
│   ├── e2_validation_predictions.csv         # row-level; keep private
│   ├── e2_rmhd_predictions.csv                # row-level; keep private
│   └── [trained model/checkpoint directories] # keep private unless redistribution is approved
└── results/
    ├── predictions/
    │   └── goemotions/
    │       ├── checkpoints/                  # per-post predictions; keep private
    │       └── rmhd_goemotions_predictions.csv # per-post predictions; keep private
    ├── temporal_features_90d_180d.csv         # user-level; keep private
    ├── post_level_temporal_90d_180d.csv       # post-level; keep private
    ├── combined_temporal_behavioural_features.csv # user-level; keep private
    ├── multivariable_temporal_rate_adjusted_results.csv
    ├── multivariable_temporal_length_adjusted_results.csv
    ├── multivariable_temporal_cooks_excluded_BH_results.csv
    ├── significant_temporal_emotion_summary.csv # aggregate candidate; review before publication
    └── consolidated_project_results.csv        # aggregate candidate; review before publication
```

The tree is illustrative, not a claim that every listed file is distributed. Preserve exact spelling and capitalization where the notebook uses them.

## 6. Path variables used by the notebook

The notebook uses several local variables that resolve to the same hard-coded root. The table lists the principal path definitions found in code; individual cells may define additional aliases such as `ROOT`, `BASE`, `PROJECT_DIR`, `PROJECT_ROOT`, `RESULTS`, `MODELS`, `CSSRS`, `DATA_DIR`, `INPUT`, `OUTPUT_PATH`, and `MODEL_PATH`.

| Variable / expression | Notebook path or example | What it points to / what a new user must do |
|---|---|---|
| `PROJECT_DIR` | `/content/drive/MyDrive/mental-health-analysis` | Root directory used by the first GoEmotions inference section. Change to the user's own project root. |
| `PROJECT_ROOT` | `/content/drive/MyDrive/mental-health-analysis` | Root used by temporal feature/configuration cells. Change consistently if running outside the original Drive. |
| `ROOT` | `/content/drive/MyDrive/mental-health-analysis` | Root used by later risk, regression, audit, and summary cells. Change consistently. |
| `BASE` | Typically assigned to the project root in later cells | Root alias used by later analysis; verify each definition before execution. |
| `DATA_PATH` | `data/processed/rmhd_emotion_input.csv` under `PROJECT_DIR` | Prepared post/text input for GoEmotions inference; must be created from permitted source data. |
| `CHECKPOINT_DIR` | `results/predictions/goemotions/checkpoints` | Stores resumable GoEmotions prediction chunks. Contains row-level data; keep private. |
| `OUTPUT_PATH` in GoEmotions export | `results/predictions/goemotions/rmhd_goemotions_predictions.csv` | Combined 28-emotion prediction output; row-level, do not publish. |
| `DATA_DIR` in C-SSRS cells | `data/CSSRS` under the project root | Must contain `Copy of cssrs_train.parquet`, `Copy of cssrs_val.parquet`, and `Copy of cssrs_test.parquet`. |
| `RESULTS` | `results/` under the project root | Directory for analysis outputs. Keep individual-level and per-post files private. |
| `MODELS` / `MODEL_PATH` | `models/` or a local trained-model directory, depending on cell | Model checkpoints or prediction files. Exact path varies by cell; inspect each assignment. Path aliases vary by section; follow each cell’s explicit path assignment and keep the root consistent. |

## 7. Environment setup

### Option A — Google Colab

1. Open `Google-Colab-notebook.ipynb` in the repository (or use the exact notebook filename currently shown in GitHub).
2. Open it in Colab and choose **File → Save a copy in Drive**. Use your own Drive copy; the original author's Drive is private.
3. Run the notebook's Drive-mount cell, which calls `drive.mount('/content/drive')`, and authorize access to your own Google Drive.
4. Create your own project folder at `/content/drive/MyDrive/mental-health-analysis`, or change the root path variables to match your chosen folder.
5. Download permitted source datasets and prepare the intermediate files described in Section 8. Place files at the exact paths expected by the notebook.
6. Run file-existence and schema checks before expensive model operations.
7. Select a GPU runtime if available and appropriate. Availability and GPU model depend on Colab allocation.
8. Execute sections selectively and in order. Do not rerun training or full inference unless intended.

### Option B — local computer

1. Clone or download the repository.
2. Create a project data directory on the local computer and obtain the permitted datasets.
3. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# macOS/Linux
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install jupyter
```

4. Install a PyTorch build appropriate for the operating system and CPU/GPU from the official selector: https://pytorch.org/get-started/locally/.
5. Install a compatible Parquet engine if needed, for example `pyarrow`, after confirming the tested version. The notebook does not pin a tested Parquet-engine version; use a compatible engine for the chosen environment.
6. Change the notebook's hard-coded Colab root paths to the local project root. Do this consistently in every cell that defines `ROOT`, `BASE`, `PROJECT_DIR`, `PROJECT_ROOT`, or related path aliases.
7. Start Jupyter and open the notebook:

```bash
jupyter notebook
```

A local CPU run may be substantially slower than a GPU run. The notebook was authored in Colab and is not yet fully path-portable.

**Important notebook-specific distinction:** GoEmotions inference uses `MAX_LENGTH = 512`, while the C-SSRS DistilBERT training/tokenization path uses `MAX_LENGTH = 256`. Later E2 inference/evaluation cells also use 256 in some paths and 512 in the separate RMHD GoEmotions pipeline; do not collapse these into one global maximum length.

## 8. Intermediate data preparation

The notebook consumes intermediate CSVs from the original project folder. It does not, in the public repository, provide a complete independently verified raw-to-processed recipe for every file. The known schemas below come from the project inventory and code references; exact row filters and transformations must be confirmed from the relevant preparation cells.

| File | Known schema / role | Created by / used in | Publication status |
|---|---|---|---|
| `data/processed/rmhd_680_clean.csv` | Known columns: `author`, `created_utc`, `score`, `selftext`, `subreddit`, `title`, `timestamp`. | Prepared RMHD table used downstream. The current notebook consumes this prepared table; the upstream creation recipe and 680-author selection logic are not included in the current notebook. | Do not publish as-is: includes author identifiers and text fields. |
| `data/processed/rmhd_emotion_input.csv` | Known columns: `post_id`, `author`, `created_utc`, `score`, `selftext`, `subreddit`, `title`, `timestamp`, `text`. | Input to GoEmotions inference (`DATA_PATH`). The `text` field is the model input. The current notebook does not define the upstream text-construction and post-ID-generation recipe. | Do not publish as-is: includes identifiers and raw text. |
| `data/CSSRS/Copy of cssrs_train.parquet` | Schema is inspected by notebook cells; not restated here because exact full schema should be confirmed from the file. | Training split for DistilBERT classifier. | Do not publish without permission and privacy review. |
| `data/CSSRS/Copy of cssrs_val.parquet` | Schema is inspected by notebook cells; verify exact columns. | Validation/model-selection split. | Do not publish without permission and privacy review. |
| `data/CSSRS/Copy of cssrs_test.parquet` | Schema is inspected by notebook cells; verify exact columns. | Held-out evaluation split. | Do not publish without permission and privacy review. |

### Safe preparation documentation

To make preprocessing reproducible, document these steps from the actual source code rather than inferring them from output CSVs:

1. Record the raw input filenames and dataset version/revision.
2. Record all row filters and exclusion rules.
3. Record how missing author, text, timestamp, title, and subreddit values are handled.
4. Record the exact deduplication key and duplicate-handling rule.
5. Record timestamp parsing, timezone, and invalid-date handling.
6. Record the exact construction of `text` in `rmhd_emotion_input.csv`.
7. Report row counts and unique-author counts before and after every transformation.
8. Save a schema/data dictionary that contains column names and meanings, but no row-level records.

Do not publish `rmhd_680_clean.csv` or `rmhd_emotion_input.csv` in their current form. If publishing any derived row-level data is considered, first review the original data terms, platform terms, ethics/privacy requirements, and re-identification risk. A safer release is preprocessing code plus aggregate counts and synthetic examples.

### Legacy notebook note

The older `NLP(1).ipynb` belongs to an earlier iteration. It records a separate cohort of 83,895 posts / 688 users and a chronological 80/20 split with a weighted composite score. Those figures and that scoring method are not the current experiment. The current notebook uses the 83,516-post / 680-author prepared corpus and the E2 `behavior + attempt` probability score; do not combine results from the two notebook versions.

## 9. Execution order and expected outputs

The cleaned notebook attachment reviewed for this documentation contains 244 cells, with repeated analysis and audit sections. The order below groups the actual headings into practical phases; a phase may read files generated earlier or files that existed in the original Drive.

| Phase / notebook headings | Inputs | Main output or effect | Computation |
|---|---|---|---|
| Load Dataset and Initialize GoEmotions; GoEmotions Inference; Export GoEmotions Predictions | `data/processed/rmhd_emotion_input.csv`; model `SamLowe/roberta-base-go_emotions` | `results/predictions/goemotions/checkpoints/checkpoint_*.csv`; `results/predictions/goemotions/rmhd_goemotions_predictions.csv` | Full transformer inference; potentially expensive. Batch size `32`, checkpoint chunk `1000`, max length `512`. |
| GoEmotions Prediction Validation; Prediction Alignment Verification; Dataset and Timestamp Integrity | Saved predictions and prepared input | Validation checks and diagnostics; alignment sample uses `random_state=42`, `n=200` | Mostly diagnostics; alignment reruns inference for 200 sampled rows. |
| Text Length and Truncation Analysis; Prediction Quality Diagnostics; Relationship Between Text Length and Emotion Scores; Distribution of Truncated Posts Across Users; Calculate Token Lengths for Truncation Analysis; User-Level Truncation Analysis; Truncation Sensitivity Analysis | Input text, saved predictions, tokenizer | Token-length and truncation diagnostics, including `results/predictions/goemotions/token_lengths.csv` | Tokenizer pass; some sensitivity cells may recompute predictions for affected posts. |
| Emotion distribution/correlation/variability, ICC, temporal baseline stability, shuffled baseline, subreddit diagnostics, monthly drift | Prepared input and saved GoEmotions predictions | Diagnostic summaries and plots/tables | Primarily analysis; no new full-model training in these sections. |
| Temporal Window Coverage Analysis; Temporal Feature Engineering — Window Coverage; User-Level Temporal Emotion Deviations; Freeze Analysis Configuration; Temporal Feature Quality Checks; Build Post-Level Temporal Dataset | Post timestamps and emotion predictions | `results/temporal_features_90d_180d.csv`, config files, `results/post_level_temporal_90d_180d.csv`, temporal diagnostics | User-level aggregation; primary recent window `90` days, baseline `180` days, minimum `5` recent and `15` baseline posts. Strict sensitivity uses `8` recent and `20` baseline posts. |
| Statistical Null Calibration; temporal emotion differences/population tests and related sensitivity cells | Temporal features/post-level temporal dataset | Null calibration, population tests, sensitivity CSVs | Statistical simulations/permutations may be time-consuming. Iteration counts differ by procedure and are defined locally in the relevant cells; there is no single global iteration count for all null/sensitivity analyses. |
| Inspect C-SSRS Splits; classifier tokenization/training/evaluation; E2 predictions and comparisons | Three C-SSRS parquet splits; model base `distilbert-base-uncased`; local checkpoint paths | Classifier checkpoints and evaluation/prediction CSVs under `models/` | Includes model training in cells 101/103 and inference/evaluation in later cells. Training: seed `42`, batch size `8`, eval batch size `16`, learning rate `2e-5`; maximum 8 epochs, with early stopping patience 2. Max token length `256` in training tokenization. |
| Cross-Split Leakage Audit; Near-Duplicate Leakage Audit; User-Level Risk Indicator Design; indicator activity sensitivity | C-SSRS splits; saved predictions and temporal features | Exact-match, shared-user, near-duplicate audit CSVs; user-level indicators and sensitivity summaries | Audit and analysis; do not publish row-level audit examples or user-level tables. Near-duplicate candidate threshold is cosine similarity `≥ 0.85`. |
| Temporal Emotion–Risk Association Review; Supporting Temporal Emotion Signals; Consolidated Experimental Results; Results and Discussion | Saved aggregate analysis tables and private user-level intermediates | Aggregate summaries such as `significant_temporal_emotion_summary.csv` and `consolidated_project_results.csv` | Statistical summary and reporting. |
| Post Length and Truncation Diagnostic; multiple-testing correction; Cook's-distance sensitivity; user-level bootstrap confidence intervals | Combined user-level temporal table, token-length changes, primary regression features | `post_length_risk_diagnostic.csv`; regression sensitivity CSVs; bootstrap CSVs | Statistical analysis; bootstrap uses `B=5000`, seed `20261019`. |

**Important:** This table describes the notebook's observed workflow, not a guarantee that all phases can run from a clean clone. The notebook currently depends on local files and model artifacts that are not in GitHub.

## 10. Sanity checks and headline results

The notebook's GoEmotions validation cell explicitly asserts:

- `83,516` rows and `83,516` unique `post_id` values in the combined prediction file.
- `28` emotion columns.
- All probabilities are finite and within `[0, 1]`.

The notebook's C-SSRS split audit describes `2,740` examples: train `1,815`, validation `482`, test `443`. The recorded E2 test output reports accuracy `0.6388`, macro-F1 `0.5079`, and weighted-F1 `0.6231`. The recorded validation output reports accuracy `0.6515`, macro-F1 `0.3374`, and weighted-F1 `0.6450`. Confirm these values against the saved notebook outputs and the exact checkpoint before quoting them in a report.

The temporal analysis design uses a recent window of `90` days and baseline window of `180` days, anchored to each user's latest observed post, with at least `5` recent and `15` baseline posts. The recorded temporal cohort contains `258` users and `16,639` posts; the larger RMHD-derived set is recorded as `680` users and `83,516` posts. These are project-specific cohort counts, not prevalence estimates.

The recorded primary rate-adjusted multivariable OLS model uses HC3 robust standard errors, six emotion-change predictors and relative posting-rate change. The notebook's saved/recorded results show disappointment coefficient `0.42338`, robust p-value `0.000450`, and BH-adjusted p-value `0.00315`. Interpret this as an exploratory association, not causation or clinical validity. These are the recorded project results; they should be tied to the same saved output files/checkpoint when cited in a publication.

## 11. Reproducibility notes

- The C-SSRS classifier training code sets seed `42` for Python/NumPy/PyTorch/Trainer-related settings shown in the cells.
- The GoEmotions alignment check samples `200` posts with `random_state=42`.
- The user-level bootstrap uses `5,000` requested resamples and NumPy seed `20261019`.
- The analysis configuration applies Benjamini–Hochberg false-discovery-rate correction with alpha `0.05`.
- Tokenizer truncation limit is `512` for GoEmotions inference. DistilBERT classifier tokenization uses max length `256` in the training path; later E2 inference cells also use `512` in some operations. Follow the setting in the specific cell being run.
- Results may vary with dataset revisions, split files, model revision, library versions, GPU/CPU kernels, preprocessing, timestamp handling, missing data, and randomness not fully controlled by the notebook.
- The test set should remain held out. Do not tune model/threshold choices on the test split.
- The notebook contains repeated exploratory/diagnostic cells and later refinements. The notebook contains multiple exploratory and sensitivity sections; interpret the primary 90/180-day configuration separately from alternative-window and strict-eligibility results.

## 12. Troubleshooting

| Symptom | Likely cause | Suggested fix |
|---|---|---|
| `FileNotFoundError` for a CSV or Parquet file | The user's local folder does not match the original Drive layout, or the preprocessing step has not been run. | Confirm the project root and exact filename; create required intermediate files from the documented source data/code. |
| Drive mount or authorization error | Colab is not connected to the user's Drive. | Rerun the Drive mount cell and authorize the user's own account. |
| Model download/load failure | Network restriction, incorrect model path, or missing private checkpoint. | Verify the model identifier and model-card access; distinguish public base models from project-trained checkpoints. |
| GPU out-of-memory | Batch size or sequence length exceeds available VRAM. | Use a suitable GPU, reduce batch size only if methodologically acceptable, and document the changed setting. |
| Inference is slow | CPU runtime, large corpus, model downloads, or full rerun. | Use GPU if available; reuse verified outputs only if permitted and present; do not rerun inference unnecessarily. |
| Parquet read error | Missing/incompatible Parquet engine or damaged file. | Install a compatible engine such as `pyarrow` and verify the downloaded split files. Record the chosen Parquet engine version in any reproduced environment. |
| Row counts differ from the expected counts | Different source release, preprocessing rules, filters, duplicates, or incomplete checkpoints. | Compare source versions and intermediate counts; inspect checkpoint completeness and deduplication. |
| GoEmotions output has duplicate post IDs | Multiple or stale checkpoint files overlap. | Rerun the export/validation logic and confirm one row per `post_id`; do not publish per-post predictions. |
| A regression result differs | Different eligible cohort, missing values, predictor columns, window settings, or package versions. | Compare the exact input table, eligibility rules, model output versions, and regression cell settings. |

## 13. Pre-commit checklist

Before committing any new file to this public repository:

- [ ] No raw Reddit text, titles, selftext, or identifiable quotations.
- [ ] No `author`, username, `post_id`, user-level row, per-post prediction, or row-level leakage-audit example.
- [ ] No model weights, checkpoint directories, optimizer state, or large private artifacts.
- [ ] No API keys, tokens, passwords, private URLs, or credentials.
- [ ] `.gitignore` excludes local data, checkpoints, environment files, and generated row-level outputs.
- [ ] No restricted dataset has been copied into GitHub contrary to its terms.
- [ ] All published CSVs have been inspected in full, not only by column names or first rows.
- [ ] The README and setup guide use the exact notebook filename present in GitHub.
- [ ] Aggregate findings have been checked against notebook outputs and are described as exploratory.
- [ ] The README and documentation clearly state that the current notebook starts from prepared CSVs and does not include a complete raw-to-prepared-data script.
