# LLM-Based Classification of Technology Transfer and Capacity Building Activities

This repository contains the code for a pilot study conducted to evaluate the use of large language models (LLMs) to classify development cooperation activities as contributing to **Technology Transfer (TT)** or **Capacity Building (CB)**, based on Swiss definitions under the UNFCCC's Enhanced Transparency Framework (ETF).

The approach uses the [Meta-Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) model via Hugging Face’s inference API to assess project titles and descriptions.

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
├── data/                        # XLSX input data (not tracked by git)
│   ├── ClimateData_2021_2022.xlsx
│   ├── Definitions.xlsx
│   └── Final_TT_CB_Labels.xlsx
│
├── results/                     # Output CSVs
│
├── src/                         # Core logic
│   ├── load_data.py             # Load and clean input files
│   ├── prompts.py               # Prompt and role builder
│   ├── classifier.py            # API call logic
│   └── evaluate.py              # Scoring and confusion matrix
│
├── .env                         # Stores HF_API_TOKEN (not tracked)
├── .gitignore
├── requirements.txt
├── run_classification.py        # Main pipeline script
└── README.md
```

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
    venv\Scripts\activate  # Windows
    ```

3. **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Add your Hugging Face API token** in a `.env` file:
    ```
    HF_API_TOKEN=your_token_here
    ```

5. **Place your data files** into the `data/` folder (files are not included for privacy).

---

## 🚀 Running the Pipeline

To classify a test set and evaluate results:

```bash
python run_classification.py
```

This will:
- Build prompts based on labeled examples
- Classify each project using the LLM
- Save predictions to `results/test_7.csv`
- Print evaluation metrics to the console

---

## 📊 Output Example

```
📊 Evaluation for Technology Transfer:
Confusion Matrix:
[[25, 2],
 [ 4, 9]]
Accuracy:  0.85
Precision: 0.82
Recall:    0.69
F1 Score:  0.75
```

---

## 📄 License

This project is released under the **MIT License** – free to use, share, and modify.

---

## ✉️ Contact

For questions or collaboration, feel free to reach out via GitHub or contact the project lead directly.