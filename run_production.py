import os
import pandas as pd
from src.load_data import load_definitions, load_labeled_examples
from src.prompts import build_prompts
from src.classifier import classify_project

# --- CONFIG ---
MODEL_ID    = "mistral-small-latest"   # the chosen model; pin the dated ID for the cycle
MODEL_DELAY = 1.5
NUM_EXAMPLES = 9
INPUT_PATH  = "data/llm_input.csv"
OUTPUT_PATH = "results/classified_projects.csv"
CHECKPOINT  = "results/classified_checkpoint.csv"

# --- Load inputs ---
print("📥 Loading data...")
definitions = load_definitions()
train_df    = load_labeled_examples()

df = pd.read_csv(INPUT_PATH)

# --- Build prompts (same examples/definitions as validation) ---
context_tt, role_tt, context_cb, role_cb = build_prompts(train_df, definitions, num_examples=NUM_EXAMPLES)

# --- Resume from checkpoint if present ---
if os.path.exists(CHECKPOINT):
    done = pd.read_csv(CHECKPOINT)
    print(f"⏩ Resuming: {len(done)} rows already classified")
    done_keys = set(done["label"])
    todo = df[~df["label"].isin(done_keys)].copy()
    results = [done]
else:
    todo = df.copy()
    results = []

print(f"🚀 Classifying {len(todo)} projects with {MODEL_ID}...")

# --- Classify, checkpointing every N rows ---
CHECKPOINT_EVERY = 25
buffer = []
for i, (_, row) in enumerate(todo.iterrows(), 1):
    tt = classify_project(row["Title"], row["Description"], context_tt, role_tt, "TT",
                          model=MODEL_ID, delay=MODEL_DELAY)
    cb = classify_project(row["Title"], row["Description"], context_cb, role_cb, "CB",
                          model=MODEL_ID, delay=MODEL_DELAY)
    buffer.append({**row.to_dict(), "is_TT": tt, "is_CB": cb})

    if i % CHECKPOINT_EVERY == 0 or i == len(todo):
        chunk = pd.DataFrame(buffer)
        combined = pd.concat(results + [chunk], ignore_index=True) if results else chunk
        combined.to_csv(CHECKPOINT, index=False)
        print(f"  💾 checkpoint: {len(combined)} done")

# --- Finalize ---
final = pd.read_csv(CHECKPOINT)
final.to_csv(OUTPUT_PATH, index=False)
n_fail = final["is_TT"].isna().sum() + final["is_CB"].isna().sum()
print(f"✅ Done. {len(final)} projects → {OUTPUT_PATH}")
if n_fail:
    print(f"⚠️ {n_fail} unresolved (None) predictions — re-run to retry those rows")