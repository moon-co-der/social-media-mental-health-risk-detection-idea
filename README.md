# Temporal Behavioral Analysis for Mental Health Risk Detection

**Author:** Shivani Shree  
**Year:** 2026

Exploratory NLP research investigating whether changes in emotion-related language in Reddit posts are associated with changes in a model-estimated risk score over time.

> **Research use only.** This project is not a clinical or diagnostic tool. Model outputs are estimates, not verified mental-health states or validated predictions of future behaviour. Statistical associations do not establish causation. Do not use this work to diagnose, label, rank, monitor, or make consequential decisions about individuals.

## Overview

- **Emotion inference:** Generates 28 GoEmotions probability scores per post using [`SamLowe/roberta-base-go_emotions`](https://huggingface.co/SamLowe/roberta-base-go_emotions).
- **Risk-related text classification:** Trains and evaluates a six-class DistilBERT classifier using a C-SSRS-related Reddit dataset. Labels are `attempt`, `behavior`, `ideation`, `indicator`, `supportive`, and `uninformative`.
- **Temporal analysis:** Compares each eligible user's recent 90-day window with the preceding 180-day baseline window.
- **Statistical analysis:** Uses Spearman correlations and rate-adjusted OLS regression with HC3 robust standard errors, Benjamini–Hochberg correction, bootstrap analysis, and sensitivity checks.
- **Quality checks:** Examines text truncation, temporal robustness, and potential train/validation/test leakage.

The model-derived score is defined as:

```text
risk_score = P(behavior) + P(attempt)
```

This is an operational research variable, **not a validated clinical scale or a probability of future behaviour**.

## Key Results

These are the project's recorded results and should be interpreted as exploratory findings.

| Measure | Recorded result |
|---|---:|
| Prepared corpus | 83,516 posts from 680 authors |
| Temporal analysis cohort | 258 eligible users across 16,639 posts |
| Classifier test set | 443 examples |
| Test accuracy | 0.639 |
| Test macro-F1 | 0.508 |
| Test weighted-F1 | 0.623 |
| Disappointment-change coefficient in the primary regression | 0.423 |
| Benjamini–Hochberg-adjusted p-value for that coefficient | 0.00315 |

An increase in the model-estimated disappointment score was positively associated with an increase in the model-derived score in the primary rate-adjusted regression. These are model-dependent associations, not evidence of causation or clinical validity. Recall was limited for the less frequent `behavior` and `attempt` classes; the classifier is not suitable for screening or individual risk assessment. See [METHODOLOGY.md](METHODOLOGY.md) for the detailed results and limitations.

## Documentation

| Document | Purpose |
|---|---|
| [METHODOLOGY.md](METHODOLOGY.md) | Research design, methods, evaluation, statistical results, limitations, and data dictionary |
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | Environment setup, dataset requirements, paths, and execution guidance |
| [WORKFLOW.md](WORKFLOW.md) | Pipeline stages, inputs, outputs, and reproducibility boundary |

## Workflow and Reproducibility

The RMHD cohort was prepared by filtering the original dataset at the **user level**: retain users with **at least 50 posts** and a timeline spanning **at least 365 days**, then keep the posts belonging to those selected users. This produced the current prepared CSV, `rmhd_680_clean.csv`, containing **83,516 posts from 680 users**. The CSV was prepared locally in VS Code and copied to Google Drive for the Colab analysis. The downstream emotion workflow then prepares `rmhd_emotion_input.csv`, runs GoEmotions inference, and saves predictions. The six reviewed helper scripts cover those downstream tasks; they are not the original RMHD cohort-filtering script itself. The older `NLP(1).ipynb` records a legacy 688-user/83,895-post cohort and a different scoring design, so do not mix those historical figures with the current analysis. See [METHODOLOGY.md](METHODOLOGY.md), [SETUP_GUIDE.md](SETUP_GUIDE.md), and [WORKFLOW.md](WORKFLOW.md).

### Reviewed Python helper scripts

The accompanying scripts cover the downstream GoEmotions stage:

- `download_goemotions_model.py` — downloads `SamLowe/roberta-base-go_emotions`.
- `test_goemotions.py` — verifies the local model exposes 28 labels and runs a sample prediction.
- `prepare_emotion_data.py` — creates `rmhd_emotion_input.csv` from an existing `rmhd_680_clean.csv`.
- `run_emotion_inference.py` — primary resumable batch inference script; writes 28 sigmoid probability columns with metadata.
- `predict_rmhd_emotions.py` — alternative older inference implementation; its output omits `post_id`, so it requires extra alignment checks.
- `generate_rmhd_dataset_summary.py` — writes `results/dataset_summary.json`; it does not preprocess raw data.

**Cohort selection:** the current prepared CSV represents the result of filtering the original RMHD dataset to users with at least 50 posts and at least 365 days between their first and last observed posts. The six reviewed scripts are the supporting downstream scripts for emotion preparation/inference and summary generation; they do not contain the cohort-filtering implementation. See [WORKFLOW.md](WORKFLOW.md) for the pipeline narrative.

### Run the Notebook

1. Open [Google-Colab-notebook.ipynb](Google-Colab-notebook.ipynb) in Google Colab.
2. Obtain the source datasets from their original sources and review their terms. See [SETUP_GUIDE.md](SETUP_GUIDE.md).
3. Place the prepared `data/processed/rmhd_680_clean.csv` in the project data folder. It represents the RMHD subset filtered to users with at least 50 posts and a timeline of at least 365 days (current cohort: 680 users and 83,516 posts). Run `prepare_emotion_data.py` to create `rmhd_emotion_input.csv` if this stage is needed.
4. Copy the prepared files into the expected Google Drive project folders and update notebook paths if your folder structure differs.
5. Mount Google Drive in Colab and run notebook sections in order. Some model inference and training stages are computationally expensive.

## Repository Structure

```text
social-media-mental-health-risk-detection-idea/
├── README.md
├── LICENSE
├── Google-Colab-notebook.ipynb
├── METHODOLOGY.md
├── SETUP_GUIDE.md
├── WORKFLOW.md
├── requirements.txt
├── .gitignore
└── results/
    ├── consolidated_project_results.csv
    └── significant_temporal_emotion_summary.csv
```

Confirm that these filenames and paths match the files actually committed to GitHub.

## Results Files

- `results/consolidated_project_results.csv` — Summary of selected project results and interpretations.
- `results/significant_temporal_emotion_summary.csv` — Aggregate summary of selected temporal emotion changes.

Only reviewed aggregate results should be public. Do not publish raw text, author identifiers, post IDs, individual-level features, per-post predictions, or model checkpoints.

## Datasets and Models

**Datasets**
- [Reddit Mental Health Dataset (RMHD)](https://zenodo.org/records/3941387)
- [Reddit C-SSRS Suicide Dataset](https://zenodo.org/records/2667859)
- [GoEmotions dataset and research repository](https://github.com/google-research/google-research/tree/master/goemotions)
- [GoEmotions research paper](https://arxiv.org/abs/2005.00547)

**Models**
- [GoEmotions RoBERTa model](https://huggingface.co/SamLowe/roberta-base-go_emotions)
- [DistilBERT base uncased](https://huggingface.co/distilbert/distilbert-base-uncased)

Third-party datasets and models retain their own licensing, usage, and attribution requirements.

## Limitations and Responsible Use

- Model scores may not accurately represent a person's actual emotional or mental-health state.
- The temporal cohort is selected by posting-history and data-availability requirements and is not representative of the general population by default.
- Language, posting activity, text length, sampling, and model error may influence associations.
- The risk score is not a validated clinical measure; statistical associations do not establish causation.
- Performance varies across classes, with lower recall for rare categories.
- The current cohort definition is at least 50 posts per user and at least 365 days between the user's first and last observed posts; the resulting prepared CSV contains 680 users and 83,516 posts. The six reviewed helper scripts cover downstream emotion setup/inference and summary generation, not the original cohort-filtering implementation.
- Do not publish raw or processed sensitive posts, author/post IDs, per-user tables, private audit examples, credentials, or access tokens.

## Licence

The original code and documentation are licensed under the [MIT License](LICENSE), provided the repository owner has the right to license those materials. Third-party datasets, pretrained models, and external materials remain subject to their own terms. The MIT License does not grant additional rights to those materials.

## Citation

Shree, S. (2026). *Temporal Behavioral Analysis for Mental Health Risk Detection*. GitHub repository: https://github.com/moon-co-der/social-media-mental-health-risk-detection-idea

## Contact

For questions or collaboration, visit [Shivani Shree's GitHub profile](https://github.com/moon-co-der) or open an issue in this repository.
