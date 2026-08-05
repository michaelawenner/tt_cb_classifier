import pandas as pd
from src.load_data import load_definitions, load_labeled_examples
from src.prompts import build_prompts
from src.classifier import classify_project
from src.evaluate import evaluate_predictions

# --- CONFIGURATION ---
TEST_ID = 1
NUM_EXAMPLES = 9           # few-shot examples in prompt     
NUM_TEST_CASES = 91         # Number of test rows to classify
OUTPUT_PATH = f"results/test_{TEST_ID}.csv"
PRED_TT_COL = f"is_technology_transfer_{TEST_ID}"
PRED_CB_COL = f"is_capacity_building_{TEST_ID}"

# --- STEP 1: Load inputs ---
print("📥 Loading data...")
definitions = load_definitions()
train_df = load_labeled_examples()
test_df_full = pd.read_excel("data/Final_TT_CB_Labels.xlsx", sheet_name="Test").copy()
test_df = test_df_full.head(NUM_TEST_CASES).copy()
print(test_df.head())


# --- STEP 2: Build prompts ---
print("🧠 Building prompts...")
context_tt, role_tt, context_cb, role_cb = build_prompts(train_df, definitions, num_examples=NUM_EXAMPLES)

# --- STEP 3: Run classification ---
print("🚀 Classifying test set...")

MODELS = {
    #"small-4": ("mistral,","mistral-small-latest", 1.5),
    #"large-3": ("mistral","mistral-large-latest", 15.0),
    "apertus": ("apertus","swiss-ai/apertus-v1.5-70b", 1)
}

# ... load definitions, train_df, build prompts as you already do ...

all_metrics = []
for model_name, (provider, model_id, delay) in MODELS.items():
    print(f"\n🚀 Running {model_name} ({provider}: {model_id})...")
    df = test_df.copy()
    tt_col = f"is_tt_{model_name}"
    cb_col = f"is_cb_{model_name}"

    df[tt_col] = df.apply(
        lambda r: classify_project(r["Title"], r["Description"], context_tt, role_tt, "TT", provider=provider, model=model_id, delay=delay),
        axis=1,
    )
    df[cb_col] = df.apply(
        lambda r: classify_project(r["Title"], r["Description"], context_cb, role_cb, "CB", provider=provider, model=model_id, delay=delay),
        axis=1,
    )

    df.to_csv(f"results/compare_{model_name}.csv", index=False)

    # drop unparsed rows before scoring so metrics aren't polluted
    tt_ok = df[df[tt_col].notna()].copy()
    cb_ok = df[df[cb_col].notna()].copy()
    n_tt_skip = len(df) - len(tt_ok)
    n_cb_skip = len(df) - len(cb_ok)
    if n_tt_skip or n_cb_skip:
        print(f"  ⚠️ skipped rows — TT:{n_tt_skip} CB:{n_cb_skip} (investigate before trusting F1)")

    m_tt = evaluate_predictions(tt_ok, "TT Manual", tt_col, f"TT / {model_name}")
    m_cb = evaluate_predictions(cb_ok, "CB Manual", cb_col, f"CB / {model_name}")
    all_metrics.append({"model": model_name, "task": "TT", **{k: m_tt[k] for k in ["accuracy","precision","recall","f1"]}})
    all_metrics.append({"model": model_name, "task": "CB", **{k: m_cb[k] for k in ["accuracy","precision","recall","f1"]}})

pd.DataFrame(all_metrics).to_csv("results/comparison_summary.csv", index=False)
print("\n", pd.DataFrame(all_metrics).to_string(index=False))