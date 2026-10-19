# Temporal Behavioral Analysis for Mental Health Risk Detection

An exploratory NLP research project that examines how emotion-related signals and model-estimated mental health risk scores in social media text change over time.

> **Important:** This is a research and educational project. It is **not** a clinical diagnostic or decision-making tool. See [Ethical Considerations and Limitations](#ethical-considerations-and-limitations).

---

## Overview

The project combines four components:

1. **Emotion classification** on social media text (GoEmotions-based predictions)
2. **Risk-score estimation** using a classifier trained on C-SSRS-related categories
3. **Temporal feature engineering** comparing historical and recent observation windows
4. **Statistical analysis** of how changes in emotion signals relate to changes in model-estimated risk

## Objectives

- Analyze emotion-related patterns in social media text.
- Evaluate a classifier for risk-related text categories.
- Build temporal features from historical and recent observation windows.
- Investigate associations between changes in emotion signals and changes in model-estimated risk scores.
- Assess the robustness and limitations of the findings.

## Methodology

| Stage | Description |
|---|---|
| **Emotion analysis** | GoEmotions-based predictions provide emotion-related signals for each text. |
| **Risk classification** | A classifier trained and evaluated on C-SSRS-related categories; performance reported with accuracy, macro-F1 and weighted-F1. |
| **Temporal analysis** | Historical vs. recent observation windows are compared to derive changes in emotion and risk signals, then tested for association. |
| **Robustness checks** | Data-quality diagnostics, sensitivity analyses and cross-split leakage audits to surface bias and uncertainty. |

## Key Findings (Exploratory)

Some emotion-related changes were associated with changes in model-estimated risk scores. In the primary multivariable analysis, **disappointment** showed a positive association, which remained after the specified multiple-testing correction.

These results are exploratory. They do not establish causality, verify any individual's mental health state, or demonstrate clinical predictive validity.

## Repository Contents

```
.
├── Temporal_Behavioral_Analysis_Mental_Health_Risk_Cleaned.ipynb   # Full analysis workflow
└── README.md
```

Aggregate results, documentation and reproducibility files may be added after review.

## Getting Started

### Prerequisites

- Python 3.9+ and Jupyter (or Google Colab)
- A GPU is recommended; some steps are computationally heavy
- Python dependencies: install the packages imported in the notebook's first cells (a `requirements.txt` is planned)

### Setup

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```
2. Obtain the required datasets under their applicable terms and licenses (the data and model artifacts are **not** bundled with this repo).
3. Open the notebook and update the file paths. The notebook currently points to files in Google Drive, so edit those paths (or mount your own Drive folder) to match your setup.
4. Run the notebook cells in order.

## Datasets and Reproducibility

- Datasets and trained model artifacts are not included in this repository.
- Users are responsible for obtaining data legally and complying with each dataset's terms of use, including any access or ethics requirements for sensitive mental-health-related data.
- Results may vary with data splits, random seeds and library versions.

## Ethical Considerations and Limitations

- Model outputs are **estimates**, not verified mental health states.
- Classification errors and class imbalance may affect results.
- Language and online behavior vary across individuals, communities and contexts.
- Statistical associations do not imply causation.
- This analysis **must not** be used to diagnose, label, or make consequential decisions about individuals.

## Future Work

- Improve reproducibility and dependency management
- Evaluate model generalization and calibration
- Explore additional temporal windows and robustness checks
- Strengthen privacy safeguards and responsible-research documentation

## Disclaimer

This project is for educational and research purposes only. It is not a substitute for professional mental health assessment or care. If you or someone you know is struggling, please reach out to a qualified mental health professional or local crisis service.

## License

_Add your chosen license here (e.g., MIT) and note any restrictions inherited from the datasets used._

## Contact

_Add your name and preferred contact or GitHub profile here._
