import pandas as pd
from src.load_data import load_definitions, load_labeled_examples
from src.prompts import build_prompts
from src.classifier import classify_project
from src.evaluate import evaluate_predictions

# --- CONFIGURATION ---
TEST_ID = 1
NUM_EXAMPLES = 9           # few-shot examples in prompt     
NUM_TEST_CASES = 2         # Number of test rows to classify
OUTPUT_PATH = f"results/test_{TEST_ID}.csv"
PRED_TT_COL = f"is_technology_transfer_{TEST_ID}"
PRED_CB_COL = f"is_capacity_building_{TEST_ID}"

# --- STEP 1: Load inputs ---
print("📥 Loading data...")
definitions = load_definitions()
train_df = load_labeled_examples()
test_df_full = pd.read_excel("data/Final_TT_CB_Labels.xlsx", sheet_name="Test").copy()
test_df = test_df_full.head(NUM_TEST_CASES)


# --- STEP 2: Build prompts ---
print("🧠 Building prompts...")
context_tt, role_tt, context_cb, role_cb = build_prompts(train_df, definitions, num_examples=NUM_EXAMPLES)

# --- STEP 3: Run classification ---
print("🚀 Classifying test set...")

test_df[PRED_TT_COL] = test_df.apply(
    lambda row: classify_project(row["Title"], row["Description"], context_tt, role_tt, tool="TT"),
    axis=1
)

test_df[PRED_CB_COL] = test_df.apply(
    lambda row: classify_project(row["Title"], row["Description"], context_cb, role_cb, tool="CB"),
    axis=1
)

# --- STEP 4: Save results ---
print(f"💾 Saving predictions to {OUTPUT_PATH}")
test_df.to_csv(OUTPUT_PATH, index=False)

# --- STEP 5: Evaluate ---
print("📈 Evaluating predictions...")
evaluate_predictions(test_df, label_col="TT Manual", prediction_col=PRED_TT_COL, label_name="Technology Transfer")
evaluate_predictions(test_df, label_col="CB Manual", prediction_col=PRED_CB_COL, label_name="Capacity Building")