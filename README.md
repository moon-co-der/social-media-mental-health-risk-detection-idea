# Temporal Behavioral Analysis for Mental Health Risk Detection

**Author:** Shivani Shree  
**Year:** 2026

Exploratory NLP research investigating whether changes in emotion-related language in Reddit posts are associated with changes in a model-estimated risk score over time.

> **Research use only.** This project is not a clinical or diagnostic tool. Model outputs are estimates, not verified mental-health states or validated predictions of future behaviour. Statistical associations do not establish causation. Do not use this work to diagnose, label, rank, monitor, or make consequential decisions about individuals.

## Overview

This project combines emotion inference, text classification, temporal feature engineering, and statistical analysis to explore changes in language over time.

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

**Main finding:** An increase in the model-estimated disappointment score was positively associated with an increase in the model-derived score in the primary rate-adjusted regression.

Recall was limited for the less frequent `behavior` and `attempt` classes. The classifier is not suitable for screening or individual risk assessment. These results are model-dependent associations, not evidence of causation or clinical validity. See [METHODOLOGY.md](METHODOLOGY.md) for detailed metrics, confidence intervals, sensitivity analyses, and limitations.

## Documentation

| Document | Purpose |
|---|---|
| [METHODOLOGY.md](METHODOLOGY.md) | Research design, methods, evaluation, statistical results, limitations, and data dictionary |
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | Environment setup, dataset requirements, paths, and execution guidance |
| [WORKFLOW.md](WORKFLOW.md) | Notebook stages, inputs, outputs, and computational requirements |

## Getting Started

### Requirements

- Python 3 and compatible package versions
- Google Colab or a suitable local Python environment
- Access to the required source datasets and prepared input files
- Adequate memory and compute for transformer inference and classifier training; a GPU is recommended for computationally intensive stages

### Run the Notebook

1. Open [Google-Colab-notebook.ipynb](Google-Colab-notebook.ipynb) in Google Colab and save a copy to your own Google Drive.
2. Obtain the datasets from their original sources and review their licences and terms. See [SETUP_GUIDE.md](SETUP_GUIDE.md).
3. Update the notebook's project paths for your environment. The notebook references `/content/drive/MyDrive/mental-health-analysis`, which is specific to the original author's Drive.
4. Place required files in the expected locations, following the setup guide.
5. Run the notebook sections in order and check that each stage produces the expected intermediate outputs.

Emotion inference and classifier training can be computationally expensive.

### Reproducibility Note

The notebook relies on prepared CSV files, intermediate outputs, and model artifacts that are not all included in this public repository. The raw-to-prepared preprocessing script is not currently included, so a clean clone cannot reproduce the entire pipeline end to end without obtaining or recreating those inputs.

Exact package versions, dataset revisions, and preprocessing details should be verified before claiming an exact reproduction. See [SETUP_GUIDE.md](SETUP_GUIDE.md) and [WORKFLOW.md](WORKFLOW.md).

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

This is the intended structure. Confirm that names and paths match the files actually committed to GitHub.

### Results Files

- `results/consolidated_project_results.csv` — Summary of selected project results and interpretations.
- `results/significant_temporal_emotion_summary.csv` — Aggregate summary of selected temporal emotion changes.

Only reviewed aggregate results should be published. Do not publish raw text, author identifiers, post IDs, individual-level features, per-post predictions, or model checkpoints.

## Datasets and Models

### Datasets

- [Reddit Mental Health Dataset (RMHD)](https://zenodo.org/records/3941387)
- [Reddit C-SSRS Suicide Dataset](https://zenodo.org/records/2667859)
- [GoEmotions dataset and research repository](https://github.com/google-research/google-research/tree/master/goemotions)
- [GoEmotions research paper](https://arxiv.org/abs/2005.00547)

### Models

- [GoEmotions RoBERTa model](https://huggingface.co/SamLowe/roberta-base-go_emotions)
- [DistilBERT base uncased](https://huggingface.co/distilbert/distilbert-base-uncased)

Each dataset and model has its own licensing, usage, and attribution requirements. Consult the original source documentation before downloading, using, or redistributing data or model artifacts.

## Limitations

- Model scores may not accurately represent a person's actual emotional or mental-health state.
- The temporal cohort is selected according to posting-history and data-availability requirements and should not be treated as representative of the general population.
- Language, posting activity, text length, sampling, and model error may influence observed associations.
- The risk score is not a validated clinical measure.
- Statistical associations do not establish causation.
- Performance varies across classes, with lower recall for rare categories.
- Complete reproduction requires data and artifacts not included in the public repository.

## Responsible Use and Privacy

This project concerns sensitive subject matter and social media text.

- Do not publish raw or processed posts containing sensitive or identifiable content.
- Do not commit author IDs, post IDs, per-user feature tables, or individual predictions.
- Do not publish private audit examples, credentials, access tokens, or private data paths.
- Do not upload model checkpoints unless redistribution and privacy implications have been reviewed.
- Review public result files for sensitive information and consistency with notebook outputs.
- Follow source dataset terms and applicable ethical requirements.

This project is intended for research and education, not individual diagnosis, screening, surveillance, ranking, or consequential decision-making.

## License

The original code and documentation are licensed under the [MIT License](LICENSE), provided that you have the right to license those materials. Third-party datasets, pretrained models, and external materials remain subject to their own licences and terms. The MIT License does not grant additional rights to those materials.

## Citation

Shree, S. (2026). *Temporal Behavioral Analysis for Mental Health Risk Detection*. GitHub repository: https://github.com/moon-co-der/social-media-mental-health-risk-detection-idea

## Contact

For questions or collaboration, visit [Shivani Shree's GitHub profile](https://github.com/moon-co-der) or open an issue in this repository.

---

*This README describes the recorded project design and results. Verify the published aggregate CSVs and final notebook outputs before using the reported metrics in a formal publication.*
