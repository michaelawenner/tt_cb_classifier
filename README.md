# LLM-Based Classification of Technology Transfer and Capacity Building Activities

This repository contains the code for a pilot study evaluating the use of large language models (LLMs) to classify development cooperation activities as contributing to **Technology Transfer (TT)** or **Capacity Building (CB)**, based on Swiss definitions under the UNFCCC's Enhanced Transparency Framework (ETF).

The pipeline supports multiple model providers through a single OpenAI-compatible interface. The current production model is **Apertus 70B** (the fully open Swiss LLM from EPFL, ETH Zurich and CSCS), served via the [Public AI Inference Utility](https://platform.publicai.co). Earlier iterations used Meta-Llama-3.3-70B (via Hugging Face) and Mistral models; see [Model Selection](#-model-selection) for how the current model was chosen.

---

## 🔍 Project Overview

- **Goal:** Improve data quality and consistency in international climate finance reporting.
- **Input:** OECD CRS-based bilateral activity data filtered by climate policy markers.
- **Method:** Few-shot prompting using curated examples and national definitions for TT and CB.
- **Output:** Binary classification (`0` or `1`) of whether an activity includes TT or CB.

---

## 📁 Project Structure

```
tt_cb_classifier/
│
├── data/                        # XLSX/CSV input data (not tracked by git)
│   ├── ClimateData_2023_2024.csv      # Deduplicated production input
│   ├── Definitions.xlsx               # TT/CB definitions (official ETF text)
│   └── Final_TT_CB_Labels.xlsx        # Train / example / dev / test labels
│
├── results/                     # Output CSVs (not tracked by git)
│
├── src/                         # Core logic
│   ├── load_data.py             # Load input files and definitions
│   ├── prompts.py               # Prompt and role builder
│   ├── classifier.py            # Multi-provider API call logic
│   └── evaluate.py              # Scoring and confusion matrix
│
├── .env                         # API tokens (not tracked)
├── .gitignore
├── requirements.txt
├── run_classification.py        # Validation pipeline (labelled data + scoring)
├── run_production.py            # Production pipeline (unlabelled data, checkpointed)
└── README.md
```

---

## 🧩 Data Preparation

Production input is **deduplicated** before classification. Swiss project numbers of the
form `7F-XXXXX.YY.ZZ` (and `UR_`/`UX_` variants) often repeat the same title and
description across phases (`.YY`) and tranches (`.ZZ`). Deduplication collapses rows with
identical normalised text so each unique project description is classified once, then the
resulting labels are joined back to all underlying activity rows.

Text is normalised (accents, apostrophes, whitespace and punctuation) only for the purpose
of matching; the original description text is preserved for classification and output.

---

## 🤖 Model Selection

Several models were evaluated on the same held-out test set (2021/2022 labelled data),
using an identical prompt so that only the model varied.

**Final held-out test results:**

| Model    | Task | Accuracy | Precision | Recall | F1    |
|----------|------|----------|-----------|--------|-------|
| Small (Mistral) | TT | 0.911 | 0.733 | 0.733 | 0.733 |
| Small (Mistral) | CB | 0.822 | 0.821 | 0.676 | 0.742 |
| **Apertus 70B** | **TT** | **0.933** | **1.000\*** | **0.600** | **0.750** |
| **Apertus 70B** | **CB** | **0.833** | **0.852** | **0.676** | **0.754** |

\* No false positives were observed on the test set. This should be read as
*highest precision observed*, not as literally perfect classification; the exact value
is expected to fall slightly below 1.00 on larger production data.

**Why Apertus was chosen for production:**

1. **Precision priority.** The primary risk to avoid in climate finance reporting is
   **over-reporting** (claiming TT/CB where it is not present). Apertus is the more
   conservative classifier: it labels TT/CB less readily, and when it does, it is more
   likely to be correct. On the test set it achieved the highest precision on both tasks,
   with no observed false positives on TT.
2. **Accepted trade-off.** The conservative approach carries lower recall (TT recall 0.60),
   meaning some genuine TT/CB activities are not identified. This under-identification is
   the deliberately accepted direction: under-reporting is preferable to over-reporting for
   an official submission. See [Methodology & Limitations](#-methodology--limitations).
3. **Governance and reproducibility.** Apertus is a fully open Swiss model (open weights,
   training data and recipe), which supports transparency and reproducibility of the
   reporting method in a way a closed, hosted model does not.

The model identifier should be **pinned to a dated version** for each reporting cycle so
results are reproducible.

---

## 🛠️ Setup Instructions

1. **Clone the repository** and navigate into the folder:
    ```bash
    git clone https://github.com/your-username/tt_cb_classifier.git
    cd tt_cb_classifier
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    venv\Scripts\activate   # Windows
    ```

3. **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Add your API token(s)** to a `.env` file. Only the token for the provider you use is
   required:
    ```
    PUBLICAI_API_TOKEN=your_token_here    # for Apertus (production)
    MISTRAL_API_TOKEN=your_token_here     # for Mistral (optional, comparison)
    ```

5. **Place your data files** into the `data/` folder (files are not included for privacy).

---

## 🚀 Running the Pipeline

**Validation** (labelled data, with scoring):

```bash
python run_classification.py
```

- Builds prompts from labelled examples and definitions
- Classifies each project via the selected provider/model
- Saves predictions to `results/`
- Prints evaluation metrics (confusion matrix, accuracy, precision, recall, F1)

**Production** (unlabelled data, checkpointed):

```bash
python run_production.py
```

- Classifies the deduplicated production input
- Checkpoints progress so interrupted runs can resume
- Flags any unresolved (failed/unparseable) rows rather than defaulting them to `0`

Provider and model are set in a single config block at the top of each script.

---

## 📊 Output Example

```
Evaluation for Technology Transfer:
Confusion Matrix:
[[TN, FP],
 [FN, TP]]
Accuracy:  0.93
Precision: 1.00
Recall:    0.60
F1 Score:  0.75
```

---

## 📐 Methodology & Limitations

- **Provider abstraction.** The classifier calls providers through a single
  OpenAI-compatible interface, so switching between Apertus (Public AI), and Mistral
  is a configuration change, not a code change.
- **Deterministic output.** Classification runs at temperature 0 so that the same input
  yields the same label, supporting reproducibility.
- **Failure handling.** API or parsing failures return an explicit "unresolved" value and
  are counted separately; they are **never** silently coerced to `0`, which would understate
  the reported totals.
- **Precision over recall (explicit choice).** The production model was selected to
  prioritise precision over recall, deliberately accepting under-identification of TT/CB
  activities in order to avoid over-reporting. On the held-out test set this yielded, for
  TT, precision 1.00 / recall 0.60.
- **Small evaluation set.** Metrics are computed on a limited labelled set; individual
  figures (especially minority-class precision/recall) have wide confidence intervals and
  should be read as directional.
- **Label noise.** The manual test labels are not guaranteed to be 100% correct. Small
  differences between models should not be over-interpreted; the model choice rests on the
  robust signal (precision advantage) and the reporting-policy priority, not on exact
  second-decimal differences.
- **Definitions are used verbatim.** The official ETF TT/CB definitions are reformatted for
  readability in the prompt but their wording is never altered.
- **Data handling.** Production classification is performed via an external inference
  gateway. Confirm the provider's data-handling terms are compatible with the sensitivity
  of the input data before running. 

---

## 📄 License

This project is released under the **MIT License** – free to use, share, and modify.

---

## ✉️ Contact

For questions or collaboration, please reach out via GitHub or contact the project lead directly.
